import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_bytes
import io

def ocr_and_parse_lab(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pages = convert_from_bytes(uploaded_file.read())
        img = pages[0]  # берём первую страницу
    else:
        img = Image.open(uploaded_file)
    text = pytesseract.image_to_string(img, lang='rus+eng')
    rows = []
    for line in text.strip().split('\n'):
        if ',' in line:
            row = [w.strip() for w in line.split(',')]
            if len(row) >= 3: rows.append(row)
    if rows:
        df = pd.DataFrame(rows, columns=["Параметр","Значение","Единица","Норма"][:len(rows[0])])
        return df
    return None

def ocr_and_parse_ecg_img(uploaded_file):
    # Для MVP: просто сообщаем что нельзя (production: подключать img2signal пайплайны)
    return None, None
