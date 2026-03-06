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
from .journal_presets import get_presets_list, get_preset_rules
from .manuscript_analyzer import analyze_manuscript, apply_llm_analysis

app = FastAPI(title="Manuscript Journal Validator")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class ExtractRulesRequest(BaseModel):
    url: str = ""
    api_key: str
    guidelines_text: str = ""


@app.post("/api/extract-rules")
async def api_extract_rules(request: ExtractRulesRequest):
    """Scrape journal guidelines or use provided text, then extract structured rules."""
    if request.guidelines_text.strip():
        guidelines_text = request.guidelines_text.strip()
    elif request.url.strip():
        try:
            guidelines_text = scrape_guidelines(request.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape guidelines: {e}")

        if not guidelines_text.strip():
            raise HTTPException(status_code=400, detail="No text content found at the provided URL.")

        if len(guidelines_text.strip()) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"Very little text extracted ({len(guidelines_text.strip())} chars). "
                       "The site may be blocking automated access. Try pasting the guidelines text manually."
            )
    else:
        raise HTTPException(status_code=400, detail="Please provide either a URL or paste the guidelines text.")

    try:
        rules = extract_rules(guidelines_text, request.api_key)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to extract rules: {e}")

    return rules.model_dump()


@app.post("/api/check")
async def api_check(
    file: UploadFile = File(...),
    rules: str = Form(...),
    word_count_includes: str = Form(""),
    api_key: str = Form(""),
):
    """Parse an uploaded manuscript and check it against rules."""
    # Parse rules JSON
    try:
        rules_data = json.loads(rules)
        journal_rules = JournalRules(**rules_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rules JSON: {e}")

    # Parse word count inclusions (comma-separated; overrides rules if provided)
    includes = [s.strip() for s in word_count_includes.split(",") if s.strip()] or None

    # Read file
    file_bytes = await file.read()
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Parse based on file type (binary-level extraction: fonts, spacing, page count)
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

    # LLM analysis: when API key is provided, use Claude for precise text-level analysis
    llm_error = None
    if api_key.strip() and metadata.raw_text:
        try:
            analysis = analyze_manuscript(metadata.raw_text, api_key.strip())
            metadata = apply_llm_analysis(metadata, analysis)
        except Exception as e:
            llm_error = str(e)

    # Run checks
    result = check_manuscript(journal_rules, metadata, word_count_includes=includes)

    resp = result.model_dump()
    resp["llm_analyzed"] = metadata.llm_analyzed
    if llm_error:
        resp["llm_error"] = llm_error
    return resp


@app.post("/api/debug-parse")
async def api_debug_parse(file: UploadFile = File(...)):
    """Debug endpoint: return parsed metadata for a manuscript."""
    file_bytes = await file.read()
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        metadata = parse_pdf(file_bytes, filename)
    elif ext == "docx":
        metadata = parse_docx(file_bytes, filename)
    elif ext == "tex":
        metadata = parse_latex(file_bytes, filename)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported: .{ext}")
    return metadata.model_dump()


@app.get("/api/journal-presets")
async def api_journal_presets():
    """Return the list of available journal presets for the dropdown."""
    return get_presets_list()


@app.get("/api/journal-presets/{preset_id}")
async def api_journal_preset_rules(preset_id: str):
    """Return the full rules for a specific journal preset."""
    rules = get_preset_rules(preset_id)
    if rules is None:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")
    return rules


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve static files from frontend directory
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
