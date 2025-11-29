
import os
import io
import re
import time
import json
import base64
import mimetypes
import asyncio
import logging
import urllib.parse
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import httpx
import pandas as pd
import pdfplumber
from playwright.async_api import async_playwright

# --------------------------------------------------------
# CONFIGURATION FROM ENVIRONMENT
# --------------------------------------------------------
AUTH_TOKEN = os.getenv("SECRET")
LANGUAGE_MODEL_ENDPOINT = os.getenv("OPENAI_BASE_URL")
LANGUAGE_MODEL_KEY = os.getenv("OPENAI_API_KEY")
SPEECH_TO_TEXT_ENDPOINT = os.getenv("WHISPER_API_URL")
SPEECH_TO_TEXT_KEY = os.getenv("WHISPER_API_KEY")
FILE_SIZE_LIMIT = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))
REQUEST_TIMEOUT = int(os.getenv("TOTAL_TIMEOUT", "170"))
RESPONSE_SIZE_LIMIT = int(os.getenv("FINAL_JSON_LIMIT", str(1 * 1024 * 1024)))
BROWSER_TIMEOUT = int(os.getenv("PLAYWRIGHT_PAGE_TIMEOUT", "60000"))
FETCH_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))

# --------------------------------------------------------
# APPLICATION INITIALIZATION
# --------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("quiz-automation")
CORS(app)

# --------------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------------
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

def create_error_response(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code

# --------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return "Quiz Automation Service — operational"

@app.route("/task", methods=["POST"])
def handle_task():
    start_time = time.time()
    
    try:
        request_data = request.get_json(force=True)
    except:
        return create_error_response("malformed json payload", 400)
    
    try:
        quiz_request = QuizRequest(**request_data)
    except ValidationError as validation_error:
        return create_error_response("schema validation failed: " + str(validation_error), 400)
    
    if quiz_request.secret != ";.wujsdfbjsubdf;bs;eoudfeUUBsU:BSfJosjBDO;>OJD":
        return create_error_response("authentication failed", 403)
    
    try:
        outcome = asyncio.run(asyncio.wait_for(
            process_quiz_workflow(quiz_request),
            timeout=REQUEST_TIMEOUT
        ))
    except asyncio.TimeoutError:
        return jsonify({"status": "timeout"}), 200
    except Exception as error:
        return jsonify({"status": "error", "msg": str(error)}), 200
    
    outcome["elapsed_s"] = time.time() - start_time
    return jsonify(outcome), 200

# --------------------------------------------------------
# MAIN WORKFLOW PROCESSOR
# --------------------------------------------------------
async def process_quiz_workflow(quiz_request: QuizRequest):
    execution_log = {"log": [], "steps": []}
    collected_answers = []
    active_url = quiz_request.url
    
    async with async_playwright() as playwright_instance:
        browser_instance = await playwright_instance.chromium.launch(headless=True)
        browser_context = await browser_instance.new_context()
        web_page = await browser_context.new_page()
        
        try:
            while True:
                execution_log["log"].append(f"loading:{active_url}")
                execution_log["steps"].append(active_url)
                
                await web_page.goto(active_url, timeout=BROWSER_TIMEOUT)
                
                try:
                    await web_page.wait_for_load_state("networkidle", timeout=20000)
                except:
                    execution_log["log"].append("network-idle-timeout")
                
                await asyncio.sleep(0.3)
                
                page_html = await web_page.content()
                try:
                    page_text = await web_page.inner_text("body", timeout=5000)
                except:
                    page_text = re.sub(r"<[^>]*>", " ", page_html)
                
                # Locate submission endpoint
                submission_endpoint = await locate_submission_url(web_page, page_html, page_text)
                execution_log["submit_url"] = submission_endpoint
                
                # Extract solution or continuation URL
                extracted_answer, extraction_info, continuation_url = await extract_solution_data(web_page, page_html, page_text)
                execution_log.update(extraction_info)
                
                # Handle multi-stage workflow
                if continuation_url:
                    execution_log["log"].append(f"workflow_continuation:{continuation_url}")
                    active_url = urllib.parse.urljoin(active_url, continuation_url)
                    continue
                
                # Process single-stage answer
                if extracted_answer is None:
                    execution_log["log"].append("extraction_unsuccessful")
                    
                    if LANGUAGE_MODEL_ENDPOINT:
                        extracted_answer = await query_language_model(page_text, active_url)
                        execution_log["via"] = "llm"
                    else:
                        return {"status": "unable_to_extract", "meta": execution_log}
                
                collected_answers.append(extracted_answer)
                
                # Prepare submission payload
                submission_data = {
                    "email": quiz_request.email,
                    "secret": quiz_request.secret,
                    "url": active_url,
                    "answer": extracted_answer
                }
                
                if len(json.dumps(submission_data).encode()) > RESPONSE_SIZE_LIMIT:
                    return {"status": "answer_too_large", "meta": execution_log}
                
                if not submission_endpoint:
                    return {"status": "no_submit_url", "meta": execution_log}
                
                execution_log["log"].append(f"posting_answer:{active_url}")
                
                async with httpx.AsyncClient(timeout=30) as http_client:
                    response = await http_client.post(submission_endpoint, json=submission_data)
                    try:
                        server_response = response.json()
                    except:
                        server_response = {"status_code": response.status_code, "text": response.text}
                
                # Check for workflow continuation
                continuation_url = server_response.get("url")
                
                if not continuation_url:
                    return {
                        "status": "multi-step-done",
                        "final_response": server_response,
                        "answers": collected_answers,
                        "meta": execution_log
                    }
                
                execution_log["log"].append(f"next_stage:{continuation_url}")
                active_url = continuation_url
        
        finally:
            await browser_instance.close()

# --------------------------------------------------------
# SUBMISSION URL LOCATOR
# --------------------------------------------------------
async def locate_submission_url(web_page, page_html, page_text):
    try:
        form_action = await web_page.eval_on_selector("form", "el => el.action", strict=False)
        if form_action:
            return form_action
    except:
        pass
    
    try:
        preformatted_content = await web_page.eval_on_selector("pre", "el => el.textContent", strict=False)
        if preformatted_content:
            try:
                parsed_json = json.loads(preformatted_content)
                if "submit" in parsed_json:
                    return parsed_json["submit"]
            except:
                pass
    except:
        pass
    
    url_pattern = re.search(r"https?://[^\s\"']*submit[^\s\"']*", page_html)
    if url_pattern:
        return url_pattern.group(0)
    
    return None

# --------------------------------------------------------
# SOLUTION EXTRACTION ENGINE
# --------------------------------------------------------
async def extract_solution_data(web_page, page_html, page_text):
    extraction_info = {"steps": []}
    continuation_url = None
    
    # Check for JSON in preformatted block
    try:
        preformatted_content = await web_page.eval_on_selector("pre", "el => el.textContent", strict=False)
    except:
        preformatted_content = None
    
    if preformatted_content:
        try:
            parsed_json = json.loads(preformatted_content)
            
            if "next" in parsed_json:
                extraction_info["steps"].append("found_next_url")
                continuation_url = parsed_json["next"]
                return None, extraction_info, continuation_url
            
            if "answer" in parsed_json:
                extraction_info["steps"].append("pre_json")
                return parsed_json["answer"], extraction_info, None
        except:
            pass
    
    # Calculate table sum
    try:
        computed_sum = await web_page.eval_on_selector_all("table", """
        (tables) => {
            function normalize(s){return (s||'').toString().trim().toLowerCase();}
            for(const table of tables){
                const headers = Array.from(table.querySelectorAll('th')).map(x=>normalize(x.innerText));
                const valueIndex = headers.indexOf('value');
                if(valueIndex >= 0){
                    let sum = 0;
                    for(const row of table.querySelectorAll('tbody tr')){
                        const cells = Array.from(row.querySelectorAll('td'));
                        if(cells.length <= valueIndex) continue;
                        const cleanedValue = cells[valueIndex].innerText.replace(/[^0-9+\-.,]/g,'');
                        const numericValue = parseFloat(cleanedValue);
                        if(!isNaN(numericValue)) sum += numericValue;
                    }
                    return sum;
                }
            }
            return null;
        }
        """)
        if computed_sum:
            extraction_info["steps"].append("html_table_sum")
            return computed_sum, extraction_info, None
    except:
        pass
    
    # Process PDF document
    pdf_link = None
    try:
        all_links = await web_page.eval_on_selector_all("a", "els => els.map(e => e.href)")
        for link in all_links:
            if link.lower().endswith(".pdf"):
                pdf_link = link
                break
    except:
        pass
    
    if pdf_link:
        extraction_info["steps"].append("pdf_link_found")
        pdf_content = await retrieve_file(pdf_link)
        if pdf_content:
            extraction_info["steps"].append("pdf_downloaded")
            computed_value = extract_pdf_table_sum(pdf_content, 2, "value")
            if computed_value is not None:
                extraction_info["steps"].append("pdf_parsed")
                return computed_value, extraction_info, None
    
    # Process spreadsheet files
    spreadsheet_match = re.search(r"https?://[^\s\"']+\.(csv|xlsx|xls)", page_text, flags=re.I)
    if spreadsheet_match:
        spreadsheet_url = spreadsheet_match.group(0)
        extraction_info["steps"].append("csv_link_found")
        file_content = await retrieve_file(spreadsheet_url)
        
        if file_content:
            try:
                dataframe = pd.read_csv(io.BytesIO(file_content))
            except:
                try:
                    dataframe = pd.read_excel(io.BytesIO(file_content))
                except:
                    dataframe = None
            
            if dataframe is not None:
                for column in dataframe.columns:
                    if str(column).strip().lower() == "value":
                        computed_total = dataframe[column].apply(pd.to_numeric, errors="coerce").sum()
                        extraction_info["steps"].append("csv_sum")
                        return float(computed_total), extraction_info, None
    
    # Process audio file
    audio_match = re.search(r"https?://[^\s\"']+\.(wav|mp3|ogg|m4a|flac|aac)", page_text, flags=re.I)
    if audio_match:
        audio_url = audio_match.group(0)
        extraction_info["steps"].append("audio_link_found")
        audio_content = await retrieve_file(audio_url)
        if audio_content:
            transcription = await convert_speech_to_text(audio_content, audio_url.split("/")[-1])
            if transcription:
                extraction_info["steps"].append("audio_transcribed")
                return transcription, extraction_info, None
    
    # Decode base64 embedded JSON
    base64_match = re.search(r"atob\(['\"]([A-Za-z0-9+/=\n\r]+)['\"]\)", page_text)
    if base64_match:
        try:
            decoded_content = base64.b64decode(base64_match.group(1))
            json_match = re.search(rb"{.*?}", decoded_content, flags=re.S)
            if json_match:
                parsed_json = json.loads(json_match.group(0).decode())
                if "answer" in parsed_json:
                    extraction_info["steps"].append("base64_json")
                    return parsed_json["answer"], extraction_info, None
        except:
            pass
    
    return None, extraction_info, None

# --------------------------------------------------------
# FILE RETRIEVAL
# --------------------------------------------------------
async def retrieve_file(file_url):
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as http_client:
            response = await http_client.get(file_url)
            if response.status_code != 200:
                return None
            if len(response.content) > FILE_SIZE_LIMIT:
                return None
            return response.content
    except:
        return None

# --------------------------------------------------------
# PDF TABLE EXTRACTION
# --------------------------------------------------------
def extract_pdf_table_sum(pdf_content, target_page, target_column):
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf_doc:
            pages_to_scan = []
            
            if target_page - 1 < len(pdf_doc.pages):
                pages_to_scan.append(pdf_doc.pages[target_page - 1])
            
            pages_to_scan.extend(pdf_doc.pages)
            
            for page in pages_to_scan:
                extracted_tables = page.extract_tables()
                if not extracted_tables:
                    continue
                
                for table in extracted_tables:
                    normalized_headers = [str(cell).strip().lower() for cell in table[0]]
                    if target_column.lower() in normalized_headers:
                        column_index = normalized_headers.index(target_column.lower())
                        accumulated_sum = 0
                        for data_row in table[1:]:
                            try:
                                accumulated_sum += float(str(data_row[column_index]).replace(",", ""))
                            except:
                                pass
                        return accumulated_sum
    except:
        return None
    
    return None

# --------------------------------------------------------
# SPEECH TRANSCRIPTION
# --------------------------------------------------------
async def convert_speech_to_text(audio_content, file_name="audio.wav"):
    if not SPEECH_TO_TEXT_ENDPOINT or not SPEECH_TO_TEXT_KEY:
        return None
    
    content_type = mimetypes.guess_type(file_name)[0] or "audio/wav"
    
    upload_files = {
        "file": (file_name, audio_content, content_type),
        "model": (None, "whisper-1"),
    }
    
    request_headers = {"Authorization": f"Bearer {SPEECH_TO_TEXT_KEY}"}
    
    try:
        async with httpx.AsyncClient(timeout=60) as http_client:
            response = await http_client.post(SPEECH_TO_TEXT_ENDPOINT, files=upload_files, headers=request_headers)
            response_data = response.json()
            return response_data.get("text", "").strip()
    except:
        return None

# --------------------------------------------------------
# LANGUAGE MODEL QUERY
# --------------------------------------------------------
async def query_language_model(content: str, source_url: str):
    if not LANGUAGE_MODEL_ENDPOINT or not LANGUAGE_MODEL_KEY:
        return None
    
    request_headers = {
        "Authorization": f"Bearer {LANGUAGE_MODEL_KEY}",
        "Content-Type": "application/json"
    }
    
    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Provide only the answer without explanation."
            },
            {
                "role": "user",
                "content": f"Source URL: {source_url}\n\nContent:\n{content[:6000]}"
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=30) as http_client:
        response = await http_client.post(LANGUAGE_MODEL_ENDPOINT, json=request_payload, headers=request_headers)
        try:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        except:
            return None

# --------------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------------
if __name__ == "__main__":
    server_port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=server_port, debug=False)