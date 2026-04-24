import os
from functools import wraps
from typing import List, Dict, Optional
from pdf2image import convert_from_path
import tempfile
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PPStructureV3
from pydantic import BaseModel, Field
from ollama import chat
import time
import asyncio

def timer(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"time taken: {(end - start):.2f}s", flush=True)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"time taken: {(end - start):.2f}s", flush=True)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

app = FastAPI()
        
class WorkExperience(BaseModel):
    jobTitle: Optional[str] = None
    company: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)

class Education(BaseModel):
    degree: Optional[str] = None
    fieldOfStudy: Optional[str] = None
    institution: Optional[str] = None
    startDate: Optional[str] = None
    graduationDate: Optional[str] = None
    location: Optional[str] = None
    grade: Optional[str] = None

class Certification(BaseModel):
    name: Optional[str] = None
    issuingOrganization: Optional[str] = None
    issueDate: Optional[str] = None
    expirationDate: Optional[str] = None
    credentialId: Optional[str] = None

class ResumeData(BaseModel):
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
    additionalInfo: Dict[str, str] = Field(default_factory=dict)


SYSTEM_PROMPT = """You are an expert resume parser. Extract structured information from OCR-extracted resume text into JSON matching the provided schema.

## Core rules

1. **No hallucination.** Only extract what is explicitly in the text. If a field is absent, use null (strings) or [] (arrays). Never infer, guess, or fabricate.

2. **OCR noise tolerance.** The text comes from OCR of a multi-column PDF. Expect broken words, reordered sections, and stray characters. Reconstruct meaning where obvious, but do not invent missing data.

3. **Section synonyms** — map these headings to schema fields:
   - "Profile" / "About" / "Objective" / "Career Summary" → summary
   - "Technical Skills" / "Core Competencies" / "Tools" / "Technologies" / "Stack" → skills
   - "Employment" / "Professional Experience" / "Work History" / "Career" → workExperience
   - "Academic Background" / "Qualifications" → education
   - "Licenses" / "Credentials" / "Courses" / "Training" → certifications
   - "Interests" / "Activities" → hobbies

4. **Dates — normalize to YYYY-MM.** "Jan 2020" → "2020-01". Year only → "YYYY". Ongoing ("Present", "Current", "Now", "–") → "Present". Keep original if unparseable.

## Field-specific rules

- **degree vs fieldOfStudy**: split them. "Bachelor of Science in Computer Science" → degree: "Bachelor of Science", fieldOfStudy: "Computer Science". "MBA, Finance" → degree: "MBA", fieldOfStudy: "Finance".

- **description vs responsibilities**: `description` is a short paragraph summary of the role (if present). `responsibilities` is the bullet list — one item per bullet, stripped of bullet characters (•, -, *, →, ▪). Never duplicate content between the two. If the role only has bullets, leave description null.

- **skills**: one skill per list item. Split comma- or pipe-separated inline lists ("Python, Java, SQL" → three items). Strip proficiency levels ("Python (Expert)" → "Python").

- **languages**: spoken/written human languages only (English, Arabic, French). Programming languages go in `skills`.

- **location**: the candidate's own city/region (usually near the name/contact block), not employer or school locations.

- **phoneNumber**: preserve the original format including country code (e.g., "+20 100 123 4567").

- **additionalInfo**: use only for labeled items that don't fit elsewhere — LinkedIn URL, portfolio, GitHub, availability, visa status, nationality, date of birth, driver's license. Key is the label, value is the content.

5. **Ordering**: preserve the order items appear in the source text for workExperience, education, and certifications.

Return only the JSON object."""


def llm_parser(resume_text: str) -> ResumeData:
    response = chat(
        model="qwen2.5:7b",
        format=ResumeData.model_json_schema(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"RESUME TEXT:\n\n{resume_text}"},
        ],
        options={"temperature": 0, "num_ctx": 16384},
    )
    content = response.message.content
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")
    return ResumeData.model_validate_json(content)

structure_pipeline = PPStructureV3(
    device="gpu",
    lang="en",

    # ── LAYOUT DETECTION ─────────────────────────────────────
    # Best layout model — handles multi-column docs well
    layout_detection_model_name="PP-DocLayout_plus-L",
    layout_threshold=0.3,              # default 0.5; lower catches narrow sidebars
    layout_nms=True,
    layout_unclip_ratio=1.05,           # slight expansion so text near box edges isn't clipped
    layout_merge_bboxes_mode="large",   # default; "union" if you see sidebars being eaten

    # ── TEXT DETECTION ───────────────────────────────────────
    text_detection_model_name="PP-OCRv5_server_det",  # higher accuracy than mobile
    text_det_limit_side_len=1536,       # default 960; larger = catches small gutter text
    text_det_limit_type="max",
    text_det_thresh=0.2,                # default 0.3; lower catches faint text
    text_det_box_thresh=0.5,            # default 0.6; lower catches more boxes
    text_det_unclip_ratio=2.0,          # default; increase to 2.5 if text is getting cut off

    # ── TEXT RECOGNITION ─────────────────────────────────────
    # For pure English: en_PP-OCRv4_mobile_rec (70.39% English accuracy)
    # For mixed/unknown: PP-OCRv5_server_rec (64.70% English, but newer architecture)
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_rec_score_thresh=0.0,          # don't filter recognition results

    # ── PREPROCESSING ────────────────────────────────────────
    use_doc_orientation_classify=True,  # safety for rotated scans
    use_doc_unwarping=False,            # only for photographed/curved docs
    use_textline_orientation=True,      # catches sideways sidebar text

    # ── STRUCTURE ────────────────────────────────────────────
    use_table_recognition=True,         # for Education sections laid out as tables
    use_region_detection=True,

    # ── DISABLE UNUSED ───────────────────────────────────────
    use_chart_recognition=False,
    use_formula_recognition=False,
    use_seal_recognition=False,
)

def extract_text_from_pdf(pdf_path: str) -> str:
    page_images = convert_from_path(pdf_path, dpi=200)
    if not page_images:
        return ""
    
    print(f"PDF has {len(page_images)} pages", flush=True)
    
    all_pages_text = []
    temp_files = []
    
    try:
        for i, img in enumerate(page_images, start=1):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                page_path = f.name
                temp_files.append(page_path)
            img.save(page_path)
            
            output = structure_pipeline.predict(input=page_path)
            
            page_text = ""
            for res in output:
                md = res.markdown
                text = md.get("markdown_texts") or md.get("markdown_text") or ""
                page_text += text
            
            print(f"  Page {i}: {len(page_text)} chars extracted", flush=True)
            all_pages_text.append(page_text)
    
    finally:
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
    print("\n\n---\n\n".join(all_pages_text))
    return "\n\n---\n\n".join(all_pages_text)

@app.post("/parse_resume/")
@timer
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    temp_path = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name

        resume_text = extract_text_from_pdf(temp_path)
        print(f"Extracted {len(resume_text)} chars", flush=True)

        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")

        resume = llm_parser(resume_text)
        return {"message": "PDF parsed successfully", "content": resume}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
