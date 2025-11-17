import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

def analyze_image_with_hf_api(file):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/hogiahien/cxr_health_multi_label"
    file.seek(0)  # на случай если файл уже читался
    response = requests.post(API_URL, headers=headers, files={"file": file})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text, "status_code": response.status_code}
