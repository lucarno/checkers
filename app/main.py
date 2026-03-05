import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import JournalRules
from .scraper import scrape_guidelines
from .rule_extractor import extract_rules
from .checker import check_manuscript
from .parsers.pdf_parser import parse_pdf
from .parsers.docx_parser import parse_docx
from .parsers.latex_parser import parse_latex

app = FastAPI(title="Manuscript Journal Validator")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class ExtractRulesRequest(BaseModel):
    url: str
    api_key: str


@app.post("/api/extract-rules")
async def api_extract_rules(request: ExtractRulesRequest):
    """Scrape journal guidelines and extract structured rules."""
    try:
        guidelines_text = scrape_guidelines(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape guidelines: {e}")

    if not guidelines_text.strip():
        raise HTTPException(status_code=400, detail="No text content found at the provided URL.")

    try:
        rules = extract_rules(guidelines_text, request.api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract rules: {e}")

    return rules.model_dump()


@app.post("/api/check")
async def api_check(file: UploadFile = File(...), rules: str = Form(...)):
    """Parse an uploaded manuscript and check it against rules."""
    # Parse rules JSON
    try:
        rules_data = json.loads(rules)
        journal_rules = JournalRules(**rules_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rules JSON: {e}")

    # Read file
    file_bytes = await file.read()
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Parse based on file type
    try:
        if ext == "pdf":
            metadata = parse_pdf(file_bytes, filename)
        elif ext == "docx":
            metadata = parse_docx(file_bytes, filename)
        elif ext == "tex":
            metadata = parse_latex(file_bytes, filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse manuscript: {e}")

    # Run checks
    result = check_manuscript(journal_rules, metadata)
    return result.model_dump()


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve static files from frontend directory
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
