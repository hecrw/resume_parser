import os
import base64
import json
import time
import tempfile
from io import BytesIO
from functools import wraps
from typing import Dict, List, Optional

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from paddleocr import PPStructureV3
from pdf2image import convert_from_bytes
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict


# ── Config ──────────────────────────────────────────────────
NATIVE_TEXT_MIN_CHARS = 100
VISION_TARGET_PIXELS = 800_000
PDF_RASTER_DPI = 150
MODEL_NAME = "Qwen/Qwen3.5-4B"
LLM_BASE_URL = "http://0.0.0.0:8000/v1"


def log(msg: str) -> None:
    print(msg, flush=True)


def timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            log(f"[{func.__name__}] time taken: {time.time() - start:.2f}s")
    return wrapper


app = FastAPI()


# ── Pydantic models ─────────────────────────────────────────
class WorkExperience(BaseModel):
    jobTitle: Optional[str] = None
    company: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: Optional[str] = None
    fieldOfStudy: Optional[str] = None
    institution: str
    startDate: Optional[str] = None
    graduationDate: Optional[str] = None
    location: Optional[str] = None
    grade: Optional[str] = None


class Certification(BaseModel):
    name: str
    issuingOrganization: Optional[str] = None
    issueDate: Optional[str] = None
    expirationDate: Optional[str] = None
    credentialId: Optional[str] = None


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidateName: Optional[str] = None
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    workExperience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)

SYSTEM_PROMPT = """You are a precise resume parser. Extract information into the JSON schema.

Field guidance:
- candidateName: the person's full name (often at the top of the resume).
- email, phoneNumber: contact details, copied verbatim.
- summary: the candidate's profile/objective paragraph if present, otherwise null.
- skills: the explicit skills list. Do NOT put skills into 'hobbies'.
- workExperience: every job, internship, or role. Each entry has its own jobTitle,
  company, startDate, endDate. Do NOT collapse multiple roles into one.
- education: degrees and institutions.
- certifications: courses, trainings, and certifications. Only the fields in the
  schema — do not invent extra fields.
- hobbies: personal interests only. Leave empty if none are listed.

Rules:
- Preserve exact spellings of names, emails, phone numbers, and company names.
- If a field has no information, use null or an empty list — do not invent values.
- Do not duplicate work experiences when sub-headings appear under one role.
"""

# ── LLM client ──────────────────────────────────────────────
client = OpenAI(api_key="EMPTY", base_url=LLM_BASE_URL)


# ── PaddleOCR pipeline ──────────────────────────────────────
structure_pipeline = PPStructureV3(
    device="gpu",
    lang=["en", "ar"],

    # Layout detection
    layout_detection_model_name="PP-DocLayout_plus-L",
    layout_threshold=0.3,
    layout_nms=True,
    layout_unclip_ratio=1.05,
    layout_merge_bboxes_mode="large",

    # Text detection
    text_detection_model_name="PP-OCRv5_server_det",
    text_det_limit_side_len=1536,
    text_det_limit_type="max",
    text_det_thresh=0.2,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=2.0,

    # Text recognition
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_rec_score_thresh=0.0,

    # Preprocessing
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True,

    # Structure
    use_table_recognition=True,
    use_region_detection=True,

    # Disabled
    use_chart_recognition=False,
    use_formula_recognition=False,
    use_seal_recognition=False,
)


# ── PDF / image helpers ─────────────────────────────────────
def rasterize_pdf(pdf_bytes: bytes, dpi: int = PDF_RASTER_DPI) -> List[Image.Image]:
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    if not images:
        raise HTTPException(status_code=422, detail="Could not rasterize PDF")
    log(f"PDF has {len(images)} page(s)")
    return images


def extract_text_native(pdf_bytes: bytes) -> str:
    """Extract embedded text using pdfplumber. Returns '' on a scanned PDF."""
    pages = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        log(f"PDF has {len(pdf.pages)} pages (native)")
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            log(f"  Page {i}: {len(text)} chars (native)")
            if text:
                pages.append(text)
    return "\n\n---\n\n".join(pages)


def _parse_ocr_page(image: Image.Image) -> str:
    """Run PPStructureV3 on a single page and return its text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        page_path = f.name
    try:
        image.save(page_path)
        output = structure_pipeline.predict(input=page_path)

        lines = []
        for res in output:
            data = res.json
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            for block in data.get("parsing_res_list", []) or []:
                content = block.get("block_content", "")
                if isinstance(content, str) and content.strip():
                    lines.append(content.strip())
        return "\n".join(lines)
    finally:
        if os.path.exists(page_path):
            os.remove(page_path)


def extract_text_from_images(images: List[Image.Image]) -> str:
    pages = []
    for i, image in enumerate(images, start=1):
        page_text = _parse_ocr_page(image)
        log(f"  Page {i}: {len(page_text)} chars extracted")
        pages.append(page_text)

    full_text = "\n\n---\n\n".join(pages)
    log(full_text)
    return full_text


# ── Vision helpers ──────────────────────────────────────────
def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def normalize_for_vision(img: Image.Image, target_pixels: int = VISION_TARGET_PIXELS) -> Image.Image:
    """Downscale large pages to a pixel budget, preserving aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    if w * h <= target_pixels:
        return img
    scale = (target_pixels / (w * h)) ** 0.5
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


# ── LLM parsers ─────────────────────────────────────────────
def _call_llm(user_content) -> ResumeData:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ResumeData",
                "schema": ResumeData.model_json_schema()
            },
        }
    )
    raw = response.choices[0].message.content
    log(f"FINISH REASON: {response.choices[0].finish_reason}")
    log(f"RAW: {raw!r}")

    if not raw or not raw.strip():
        raise ValueError(
            f"LLM returned empty response. finish_reason={response.choices[0].finish_reason}"
        )
    log(raw)
    return ResumeData.model_validate_json(raw)

def llm_parser_from_text(resume_text: str) -> ResumeData:
    return _call_llm(f"Extract structured data from this resume text:\n\n{resume_text}")


def llm_parser_from_images(images: List[Image.Image]) -> ResumeData:
    """One image block per page — no stitching."""
    intro = (
        f"This resume has {len(images)} page(s). Extract structured data from "
        "all pages into the JSON schema. Treat the pages as one continuous "
        "document; do not duplicate entries that span page breaks."
    )
    blocks = [{"type": "text", "text": intro}] + [
        {"type": "image_url", "image_url": {"url": pil_to_data_url(normalize_for_vision(img))}}
        for img in images
    ]
    return _call_llm(blocks)


# ── Endpoint helpers ────────────────────────────────────────
async def _read_pdf(file: UploadFile) -> bytes:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return content


def _parse_text(resume_text: str) -> ResumeData:
    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF")
    log(f"Extracted {len(resume_text)} chars")
    resume = llm_parser_from_text(resume_text)
    log(str(resume))
    return resume


# ── Endpoints ───────────────────────────────────────────────
@app.post("/parse_resume_ocr/")
@timer
async def parse_resume_ocr(file: UploadFile = File(...)):
    content = await _read_pdf(file)
    images = rasterize_pdf(content)
    resume_text = extract_text_from_images(images)
    resume = _parse_text(resume_text)
    return {"message": "PDF parsed successfully (OCR)", "mode": "ocr", "content": resume}


@app.post("/parse_resume_vision/")
@timer
async def parse_resume_vision(file: UploadFile = File(...)):
    content = await _read_pdf(file)
    images = rasterize_pdf(content)
    resume = llm_parser_from_images(images)
    log(str(resume))
    return {"message": "PDF parsed successfully (vision)", "mode": "vision", "content": resume}


@app.post("/parse_resume_native/")
@timer
async def parse_resume_native(file: UploadFile = File(...)):
    """Native PDF text extraction with OCR fallback for scanned PDFs."""
    content = await _read_pdf(file)
    resume_text = extract_text_native(content)
    used_fallback = False

    if len(resume_text.strip()) < NATIVE_TEXT_MIN_CHARS:
        log(
            f"Native extraction returned {len(resume_text)} chars "
            f"(< {NATIVE_TEXT_MIN_CHARS}); falling back to OCR"
        )
        used_fallback = True
        images = rasterize_pdf(content)
        resume_text = extract_text_from_images(images)

    resume = _parse_text(resume_text)
    return {
        "message": "PDF parsed successfully (native)",
        "mode": "native",
        "used_ocr_fallback": used_fallback,
        "content": resume,
    }