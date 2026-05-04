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

# NuExtract-2.0 is single-image only — route via the OCR text endpoint, which
# now produces layout-aware markdown (titles / lists / tables) that NuExtract
# was trained to consume.
MODEL = "numind/NuExtract-2.0-2B"
URL = URL_OCR


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

    # Skip PDFs we've already processed for this model — makes re-runs cheap.
    for i, pdf in enumerate(pdfs, start=1):
        out_path = RESULTS_DIR / f"{pdf.stem}__{slug}.json"
        if out_path.exists():
            print(f"[{i}/{len(pdfs)}] {pdf.name}  (skip — already done)")
            continue

        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        mode = "ocr" if URL == URL_OCR else ("vision" if URL == URL_VISION else "native")
        print(f"  → {mode}...", flush=True)
        try:
            result = call_endpoint(URL, pdf, MODEL)
        except RuntimeError as e:
            print(f"    FAILED: {e}")
            continue
        print(f"    done in {result['elapsed_seconds']}s")

        record = {
            "file": pdf.name,
            "model": MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            mode: result,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  saved → {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()