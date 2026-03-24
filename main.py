import uuid
import os
import shutil
from typing import Annotated, List, Dict, Optional
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


def convert_pdf_2_images(pdf_path: str) -> list[str]:
    output = os.path.join("outputs", str(uuid.uuid4()))
    if not os.path.exists(output):
        os.makedirs(output)
    pages = convert_from_path(pdf_path, 500, output_folder=output, fmt='JPEG')
    paths = []
    for file in os.listdir(output):
        paths.append(os.path.join("outputs",file))
    return paths

def llm_parser(image_paths: list[str]):

    prompt = """
    You are a resume parser. Extract all information from this resume image.
    Return ONLY a valid JSON object matching this schema — no markdown, no explanation.

    Rules:
    - Use null for missing fields
    - Dates as strings (e.g. "Jan 2020", "2020-01", or as written)
    - List all skills, languages, hobbies as flat string arrays
    - Capture every work experience and education entry
    """
    response = chat(
        model="qwen3-vl:4b",
        format=ResumeData.model_json_schema(),
        messages=[
            {
                "role": "user",
                "content":prompt,
                "images": image_paths,
            }
        ],
        options={"temperature": 0},
    )

    resume = ResumeData.model_validate_json(response.message.content)
    return resume


@app.post("/parse_resume/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            image_paths = convert_pdf_2_images(temp_path)
            resume = llm_parser(image_paths)
        return {
            "message": "PDF uploaded successfully",
            "content": resume
        }

    finally:
        file.file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)