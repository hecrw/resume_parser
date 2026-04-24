import uuid
import os
import base64
import shutil
from typing import List, Dict, Optional
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from ollama import chat
import pdfplumber

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


def extract_text_from_pdf(pdf_path: str) -> str:
    """extracts text from pdf and falls back to ocr if the page is empty"""
    text_parts = []
    needs_ocr: bool = False
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < 50:
                needs_ocr = True
            text_parts.append(page_text)
    return "\n\n--- PAGE BREAK ---\n\n".join(text_parts)


def convert_pdf_to_images(pdf_path: str) -> list[str]:
    """Convert PDF pages to JPEG images, return file paths."""
    output_dir = os.path.join("outputs", str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)
    convert_from_path(pdf_path, 500, output_folder=output_dir, fmt='JPEG')  
    return [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]


def encode_images_to_base64(image_paths: list[str]) -> list[str]:
    """Convert image file paths to base64 strings for Ollama."""
    encoded = []
    for path in image_paths:
        with open(path, "rb") as img_file:
            encoded.append(base64.b64encode(img_file.read()).decode("utf-8"))
    return encoded


def llm_parser(resume_text: str) -> ResumeData:
    prompt = f"""You are a resume parser. Extract all information from the resume text below into JSON matching the schema.
        Rules:
        - Use null for missing fields
        - Keep dates as written (e.g. "Jan 2020", "2020-01", "Present")
        - skills, languages, hobbies must be flat string arrays
        - Capture every work experience and education entry
        - responsibilities should be individual bullet points, not one long string

        RESUME TEXT:
        {resume_text}
    """

    response = chat(
        model="qwen2.5:7b",
        format=ResumeData.model_json_schema(),
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_ctx": 8192,
        },
    )

    content = response.message.content
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")

    return ResumeData.model_validate_json(content)



@app.post("/parse_resume/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        resume_text = extract_text_from_pdf(temp_path)

        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")

        resume = llm_parser(resume_text)
        return {"message": "PDF parsed successfully", "content": resume}

    finally:
        file.file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
