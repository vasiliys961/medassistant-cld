import requests

def analyze_image(uploaded_file):
    api_url = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}
    resp = requests.post(api_url, headers=headers, files={"file": uploaded_file})
    info = resp.json()
    labels = [x.get('label','') for x in info.get('predictions',[])]
    return f"Обнаружены: {', '.join(labels)}", info
