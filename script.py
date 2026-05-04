import os
import sys
import json
import time
from pathlib import Path

import requests

URL_VISION = "http://127.0.0.1:8001/parse_resume_vision/"
URL_OCR = "http://127.0.0.1:8001/parse_resume_ocr/"
URL_NATIVE = "http://127.0.0.1:8001/parse_resume_native/"

TESTS_DIR = Path("tests")
RESULTS_DIR = Path("results")

MODEL = "Qwen/Qwen3.5-4B"


def call_endpoint(url: str, pdf_path: Path, model: str) -> dict:
    """POST a PDF to the endpoint. Raises on any failure (caller will halt)."""
    start = time.time()
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {"model": model}
        response = requests.post(url, files=files, data=data, timeout=600)
    elapsed = time.time() - start

    if not response.ok:
        raise RuntimeError(
            f"{url} -> {response.status_code} for {pdf_path.name}: {response.text[:500]}"
        )

    return {
        "status_code": response.status_code,
        "elapsed_seconds": round(elapsed, 2),
        "response": response.json(),
    }


def model_slug(model: str) -> str:
    """Filesystem-safe tag for the model name (e.g. NuExtract-2.0-2B)."""
    return model.split("/")[-1]


def main():
    if not TESTS_DIR.is_dir():
        print(f"ERROR: {TESTS_DIR}/ not found", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(p for p in TESTS_DIR.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"ERROR: no PDFs in {TESTS_DIR}/", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    slug = model_slug(MODEL)

    print(f"Found {len(pdfs)} PDFs. Model: {MODEL}\n")

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")

        # OCR
        print("  → OCR...", flush=True)
        ocr_result = call_endpoint(URL_OCR, pdf, MODEL)
        print(f"    done in {ocr_result['elapsed_seconds']}s")

        # Native (pdfplumber, with OCR fallback)
        # print("  → native...", flush=True)
        # native_result = call_endpoint(URL_NATIVE, pdf, MODEL)
        # fallback_note = (
        #     " (OCR fallback)"
        #     if native_result["response"].get("used_ocr_fallback")
        #     else ""
        # )
        # print(f"    done in {native_result['elapsed_seconds']}s{fallback_note}")

        # Vision
        # print("  → vision...", flush=True)
        # vision_result = call_endpoint(URL_VISION, pdf, MODEL)
        # print(f"    done in {vision_result['elapsed_seconds']}s")

        record = {
            "file": pdf.name,
            "model": MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ocr": ocr_result,
            # "native": native_result,
            # "vision": vision_result,
        }

        out_path = RESULTS_DIR / f"{pdf.stem}__{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  saved → {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()