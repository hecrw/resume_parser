import sys
import requests

url = "http://127.0.0.1:8001/parse_resume/"

if len(sys.argv) < 2:
    print("Usage: python script.py <path-to-pdf>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files)

print(response.json())