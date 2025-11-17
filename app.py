import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from modules.image_analysis import analyze_image_with_hf_api
from modules.ocr_tools import ocr_and_parse_ecg_img, ocr_and_parse_lab
from modules.ecg_analysis import analyze_ecg
from modules.lab_analysis import analyze_lab_results

st.set_page_config(page_title="MedAssistant", layout="wide")

st.title("MedAssistant")
st.write("Загрузите изображение ЭКГ, лабораторные анализы или снимок для анализа.")

uploaded_file = st.file_uploader("Загрузите файл (PDF/JPG/PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_name = getattr(uploaded_file, "name", "unnamed")
    st.write(f"Работаем с файлом: {file_name}")
    try:
        if "ecg" in file_name.lower():
            signal = ocr_and_parse_ecg_img(uploaded_file)
            if signal is None:
                st.error("Извлечь сигнал из ЭКГ не удалось.")
            else:
                result = analyze_ecg(signal)
                st.write(result)

        elif "lab" in file_name.lower():
            lab_data = ocr_and_parse_lab(uploaded_file)
            if lab_data is None:
                st.error("Результаты анализов не распознаны.")
            else:
                result = analyze_lab_results(lab_data)
                st.write(result)

        else:
            analysis = analyze_image_with_hf_api(uploaded_file)
            st.write(analysis)
    except Exception as ex:
        st.exception(ex)
