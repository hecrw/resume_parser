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


def llm_parser(image_paths: list[str]) -> ResumeData:
    base64_images = encode_images_to_base64(image_paths)

    prompt = """
    You are a resume parser. Extract all information from these resume images.
    You must respond with ONLY valid JSON matching the schema. No explanation, no markdown, no code blocks.
    Rules:
    - Use null for missing fields
    - Dates as strings (e.g. "Jan 2020", "2020-01", or as written)
    - List all skills, languages, hobbies as flat string arrays
    - Capture every work experience and education entry
    """

    response = chat(
        model="ministral-3:8b",
        format=ResumeData.model_json_schema(),
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": base64_images,
            }
        ],
        options={
            "temperature": 0.5,
            "num_ctx": 16384,
        },
    )

    content = response.message.content
    print(content)
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")

    return ResumeData.model_validate_json(content)


@app.post("/parse_resume/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    temp_path = None
    image_dir = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        image_paths = convert_pdf_to_images(temp_path)
        image_dir = os.path.dirname(image_paths[0]) if image_paths else None

        resume = llm_parser(image_paths)
        return {"message": "PDF parsed successfully", "content": resume}

    finally:
        file.file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if image_dir and os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)