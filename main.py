import os
import json
import base64
import asyncio
import time
from io import BytesIO
from functools import wraps
from typing import List, Dict, Optional
from tempfile import NamedTemporaryFile
from PIL import Image

from pdf2image import convert_from_bytes
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI


def timer(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        print(f"time taken: {(time.time() - start):.2f}s", flush=True)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"time taken: {(time.time() - start):.2f}s", flush=True)
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


client = OpenAI(
    api_key="EMPTY",
    base_url="http://0.0.0.0:8000/v1",
)

MODEL_NAME = "numind/NuExtract-2.0-4B"

RESUME_TEMPLATE = {
    "candidateName": "verbatim-string",
    "email": "verbatim-string",
    "phoneNumber": "verbatim-string",
    "summary": "string",
    "location": "verbatim-string",
    "languages": ["verbatim-string"],
    "skills": ["verbatim-string"],
    "workExperience": [
        {
            "jobTitle": "verbatim-string",
            "company": "verbatim-string",
            "startDate": "verbatim-string",
            "endDate": "verbatim-string",
            "location": "verbatim-string",
            "description": "string",
            "responsibilities": ["string"],
        }
    ],
    "education": [
        {
            "degree": "verbatim-string",
            "fieldOfStudy": "verbatim-string",
            "institution": "verbatim-string",
            "startDate": "verbatim-string",
            "graduationDate": "verbatim-string",
            "location": "verbatim-string",
            "grade": "verbatim-string",
        }
    ],
    "certifications": [
        {
            "name": "verbatim-string",
            "issuingOrganization": "verbatim-string",
            "issueDate": "verbatim-string",
            "expirationDate": "verbatim-string",
            "credentialId": "verbatim-string",
        }
    ],
    "hobbies": ["verbatim-string"],
}


def pil_to_data_url(img, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URL for the OpenAI image_url block."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150):
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def stitch_vertical(images, gap: int = 20, bg=(255, 255, 255)) -> Image.Image:
    """Stack PIL images vertically into one tall image."""
    if len(images) == 1:
        return images[0]

    max_w = max(im.width for im in images)
    resized = []
    for im in images:
        if im.width != max_w:
            new_h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, new_h), Image.LANCZOS)
        resized.append(im.convert("RGB"))

    total_h = sum(im.height for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (max_w, total_h), bg)
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def llm_parser_from_images(images) -> ResumeData:
    combined = stitch_vertical(images)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(combined)}}
            ],
        }],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(RESUME_TEMPLATE, indent=4),
            },
        },
    )

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response.")
    return ResumeData.model_validate_json(raw)

@app.post("/parse_resume/")
@timer
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    images = pdf_to_images(content)
    if not images:
        raise HTTPException(status_code=422, detail="Could not rasterize PDF")

    print(f"PDF has {len(images)} pages", flush=True)

    resume = llm_parser_from_images(images)
    return {"message": "PDF parsed successfully", "content": resume}