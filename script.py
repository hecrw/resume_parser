import json
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8001/parse_resume/"
TESTS_DIR = Path("tests")
RESULTS_DIR = Path("results")
MODEL_SLUG = "gemini-2.5-flash"


def main():
    pdfs = sorted(p for p in TESTS_DIR.iterdir() if p.suffix.lower() == ".pdf")
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Found {len(pdfs)} PDFs.\n")

    for i, pdf in enumerate(pdfs, start=1):
        out_path = RESULTS_DIR / f"{pdf.stem}__{MODEL_SLUG}.json"
        if out_path.exists():
            print(f"[{i}/{len(pdfs)}] {pdf.name}  (skip)")
            continue

        print(f"[{i}/{len(pdfs)}] {pdf.name} …", end="", flush=True)
        start = time.time()
        with open(pdf, "rb") as f:
            r = requests.post(URL, files={"file": (pdf.name, f, "application/pdf")}, timeout=300)
        elapsed = round(time.time() - start, 2)

        if not r.ok:
            print(f" FAILED ({r.status_code}): {r.text[:200]}")
            continue
        print(f" {elapsed}s")

        out_path.write_text(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
