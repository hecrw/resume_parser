import requests

url = "http://127.0.0.1:8000/parse_resume/"
file_path = "test_resume.pdf"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files)

print(response.json())