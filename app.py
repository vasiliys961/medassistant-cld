import streamlit as st
import os
from dotenv import load_dotenv
import anthropic

# Локальная загрузка .env (ничего не меняет при деплое!)
load_dotenv()

# Сначала пробуем ключ из окружения, потом (при необходимости) из секрета Streamlit
API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("anthropic_key")
if not API_KEY:
    st.error("Anthropic API ключ не найден! Добавьте ANTHROPIC_API_KEY в .env или Secret.")
    st.stop()

client = anthropic.Anthropic(api_key=API_KEY)

# ... твои внутренние импорты
from modules.intent_detection import predict_intent
from modules.ecg_analysis import analyze_ecg
from modules.image_analysis import analyze_image
from modules.lab_analysis import analyze_lab
from modules.ocr_tools import ocr_and_parse_lab, ocr_and_parse_ecg_img

st.set_page_config(page_title="МедАИ Ассистент", layout="centered")
st.title("🩺 Мультимодальный медицинский ассистент (Claude)")

user_task = st.text_area("Опишите задачу (можно голосом; любые формулировки):", height=90)
uploaded_file = st.file_uploader(
    "Загрузите файл (ЭКГ, снимок, лабораторный анализ, скан или фото бумажного анализа/ЭКГ):",
    type=["csv", "xml", "jpg", "png", "dcm", "pdf", "jpeg", "tiff"])

output, details = None, None

if st.button("Анализировать"):
    if not (user_task or uploaded_file):
        st.warning("Добавьте описание задачи или хотя бы файл.")
        st.stop()

    intent = predict_intent(user_task, uploaded_file)
    st.info(f"Обнаружен кейс: {intent}")

    # ===== Анализ данных =====
    if intent == "ecg":
        if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.pdf')):
            signal, meta = ocr_and_parse_ecg_img(uploaded_file)
            if signal is None:
                st.error("Не удалось извлечь сигнал ЭКГ из изображения/скана.")
                st.stop()
            res, details = analyze_ecg(signal)
        else:
            res, details = analyze_ecg(uploaded_file)
        st.write("Результаты ЭКГ анализа:")
        st.write(res)

    elif intent == "image":
        res, details = analyze_image(uploaded_file)
        st.write("Результаты анализа снимка:")
        st.write(res)

    elif intent == "lab":
        if uploaded_file.name.endswith(('.jpg','.jpeg','.png','.pdf')):
            rows = ocr_and_parse_lab(uploaded_file)
            if not rows:
                st.error("Не удалось прочитать анализ (OCR).")
                st.stop()
            res, details = analyze_lab(rows)
        else:
            res, details = analyze_lab(uploaded_file)
        st.write("Результаты лабораторного анализа:")
        st.write(res)
    else:
        details = user_task

    # ===== Запрос к Claude =====
    prompt = (f"Ты — медицинский AI-ассистент. Клиническая задача:\n"
              f"{user_task}\n"
              f"Данные/результаты прикладного анализа:\n{details}\n"
              "Дай максимально точное заключение, рекомендации, укажи стандарты, объясни reasoning.")

    with st.spinner("Генерируем протокол и объяснения Клодом…"):
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1400,
            temperature=0.1,
            system="Ты — эксперт-медик, всё объясняешь формально и строго по стандартам.",
            messages=[{"role": "user", "content": prompt}]
        )
        # Для актуальной версии anthropic API: response.content[0].text или response.content
        protocol = response.content[0].text if hasattr(response.content[0], "text") else response.content 

        st.subheader("Заключение и протокол от Claude")
        st.text_area("📝 Протокол:", protocol, height=280)
        st.download_button("⬇️ Скачать протокол", protocol, file_name="protocol.txt", mime="text/plain")

