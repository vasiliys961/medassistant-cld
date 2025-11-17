import streamlit as st
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Загружаем переменные окружения
load_dotenv()

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medassistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импортируем модули
from modules.intent_detection import detect_intent
from modules.ecg import process_ecg
from modules.image import process_image
from modules.image_analysis import analyze_image_with_openrouter
from modules.lab import process_lab_analysis
from modules.lab_analysis import analyze_lab_results
from modules.ocr import extract_text_from_image

# ============ КОНФИГ OPENROUTER ============
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.io/api/v1/chat/completions"
MODEL_NAME = "anthropic/claude-3-sonnet-20240229"

# ============ STREAMLIT КОНФИГ ============
st.set_page_config(
    page_title="MedAssistant CLD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 MedAssistant - Clinical Language Diagnostic")
st.write("Мультимодальный медицинский ассистент для анализа ЭКГ, лабораторных анализов и медицинских изображений")

# ============ ФУНКЦИИ ============

def call_openrouter(prompt: str, system_prompt: str = None, max_tokens: int = 1400, temperature: float = 0.1) -> dict:
    """
    Отправляет запрос к OpenRouter API с обработкой ошибок.
    """
    
    if not OPENROUTER_API_KEY:
        return {
            "success": False,
            "content": None,
            "error": "OPENROUTER_API_KEY не установлен в .env или Streamlit secrets"
        }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://medassistant-cld.local",
        "X-Title": "MedAssistant",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0
    }
    
    try:
        logger.info(f"Отправка запроса к OpenRouter. Модель: {MODEL_NAME}")
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(OPENROUTER_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            logger.info("Успешный ответ от OpenRouter")
            return {
                "success": True,
                "content": content,
                "error": None,
                "usage": data.get("usage", {})
            }
        
        elif response.status_code == 401:
            error_msg = "Ошибка аутентификации: неверный API ключ OpenRouter"
            logger.error(error_msg)
            return {"success": False, "content": None, "error": error_msg}
        
        elif response.status_code == 429:
            error_msg = "Превышен лимит запросов (Rate Limit). Попробуйте позже."
            logger.warning(error_msg)
            return {"success": False, "content": None, "error": error_msg}
        
        elif response.status_code == 500:
            error_msg = "Ошибка на сервере OpenRouter (500). Попробуйте позже."
            logger.error(error_msg)
            return {"success": False, "content": None, "error": error_msg}
        
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"success": False, "content": None, "error": error_msg}
    
    except httpx.TimeoutException:
        error_msg = "Timeout: запрос занял слишком много времени"
        logger.error(error_msg)
        return {"success": False, "content": None, "error": error_msg}
    
    except Exception as e:
        error_msg = f"Неожиданная ошибка: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "content": None, "error": error_msg}


def process_uploaded_file(uploaded_file, task_description: str) -> dict:
    """
    Обрабатывает загруженный файл в зависимости от типа.
    """
    
    try:
        logger.info(f"Обработка файла: {uploaded_file.name}")
        
        intent = detect_intent(task_description, uploaded_file.name)
        logger.info(f"Определен intent: {intent}")
        
        result = {
            "intent": intent,
            "analysis": None,
            "raw_data": None,
            "error": None
        }
        
        if intent == "ecg":
            if uploaded_file.name.endswith(('.csv', '.txt')):
                ecg_data = process_ecg(uploaded_file)
                result["raw_data"] = ecg_data
                result["analysis"] = f"ЭКГ данные загружены. Количество отсчетов: {len(ecg_data)}"
                logger.info("ЭКГ успешно обработана")
            else:
                result["error"] = "ECG должна быть в формате CSV или TXT"
        
        elif intent == "image":
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_analysis = process_image(uploaded_file)
                result["raw_data"] = image_analysis
                result["analysis"] = "Изображение загружено и проанализировано"
                logger.info("Изображение успешно обработано")
            else:
                result["error"] = "Поддерживаемые форматы: PNG, JPG, JPEG, BMP"
        
        elif intent == "lab":
            if uploaded_file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
                lab_data = process_lab_analysis(uploaded_file)
                result["raw_data"] = lab_data
                result["analysis"] = f"Лабораторные данные загружены. Параметров: {len(lab_data)}"
                logger.info("Лабораторные анализы успешно обработаны")
            else:
                result["error"] = "Лабораторные анализы должны быть в формате CSV, XLSX или XLS"
        
        elif intent == "document":
            if uploaded_file.name.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                extracted_text = extract_text_from_image(uploaded_file)
                result["raw_data"] = extracted_text
                result["analysis"] = f"Текст извлечен из документа. Длина текста: {len(extracted_text)} символов"
                logger.info("Текст успешно извлечен из документа")
            else:
                result["error"] = "Поддерживаемые форматы документов: PDF, PNG, JPG"
        
        else:
            result["error"] = f"Неизвестный тип файла: {intent}"
        
        return result
    
    except Exception as e:
        error_msg = f"Ошибка при обработке файла: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "intent": None,
            "analysis": None,
            "raw_data": None,
            "error": error_msg
        }


def generate_medical_report(task_description: str, analysis_data: dict) -> dict:
    """
    Генерирует медицинский отчет через OpenRouter API.
    """
    
    context = f"""
    Задача: {task_description}
    Тип анализа: {analysis_data.get('intent', 'неизвестно')}
    Предварительный анализ: {analysis_data.get('analysis', 'нет')}
    """
    
    if analysis_data.get('raw_data'):
        context += f"\nДанные: {json.dumps(analysis_data['raw_data'], ensure_ascii=False, indent=2)[:2000]}"
    
    system_prompt = """Ты — опытный врач-диагност и кардиолог с глубокими знаниями стандартов диагностики.
    Твоя задача — провести качественный анализ медицинских данных, опираясь на современные стандарты медицины.
    В ответе:
    1. Описание находок
    2. Предварительные выводы
    3. Рекомендации по стандартам (ГОСТ, МКБ-10, ESC, ACC/AHA)
    4. Необходимые дополнительные исследования
    5. Рекомендации по лечению и наблюдению
    Формат: структурированный отчет с понятными заголовками."""
    
    prompt = f"""На основе следующих медицинских данных подготовь детальный диагностический отчет:
    
    {context}
    
    Проведи полный анализ с учетом клинических стандартов и рекомендаций."""
    
    logger.info("Формирование запроса для генерации отчета")
    
    result = call_openrouter(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=1400,
        temperature=0.1
    )
    
    return result


# ============ STREAMLIT UI ============

with st.sidebar:
    st.header("Параметры")
    
    st.subheader("Модель")
    st.info(f"Модель: {MODEL_NAME}\n\nAPI: OpenRouter")
    
    st.subheader("Проверка конфигурации")
    if OPENROUTER_API_KEY:
        st.success("API ключ загружен")
    else:
        st.error("API ключ не найден")
        st.write("Добавьте в .env:")
        st.code("OPENROUTER_API_KEY=sk_...")
    
    st.divider()
    
    st.subheader("О приложении")
    st.markdown("""
    MedAssistant - мультимодальный медицинский ассистент.
    
    Типы анализа:
    - ЭКГ (ECG)
    - Медицинские изображения
    - Лабораторные анализы
    - Документы (OCR)
    """)

st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    task_description = st.text_area(
        "Описание задачи",
        placeholder="Пример: боль в груди, одышка...",
        height=100
    )

with col2:
    uploaded_file = st.file_uploader(
        "Загрузите файл",
        type=["csv", "txt", "png", "jpg", "jpeg", "bmp", "xlsx", "xls", "pdf"],
        help="Поддерживаемые форматы: CSV, TXT, PNG, JPG, XLSX, XLS, PDF"
    )

st.write("---")

if st.button("Провести анализ", type="primary", use_container_width=True):
    
    if not task_description:
        st.error("Опишите задачу")
    elif not uploaded_file:
        st.error("Загрузите файл")
    else:
        
        with st.spinner("Обработка файла..."):
            file_result = process_uploaded_file(uploaded_file, task_description)
            
            if file_result["error"]:
                st.error(f"Ошибка: {file_result['error']}")
            else:
                st.success(f"Файл обработан: {file_result['analysis']}")
                
                with st.expander("Предварительный анализ"):
                    st.write(f"Тип: {file_result['intent']}")
                    if file_result['raw_data']:
                        st.write(f"Данные: {json.dumps(file_result['raw_data'], ensure_ascii=False, indent=2)[:500]}")
        
        with st.spinner("Генерация отчета..."):
            report_result = generate_medical_report(task_description, file_result)
            
            if report_result["success"]:
                st.success("Отчет готов!")
                
                st.subheader("Медицинский отчет")
                st.markdown(report_result["content"])
                
                if report_result.get("usage"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Input Tokens", report_result["usage"].get("prompt_tokens", "N/A"))
                    with col2:
                        st.metric("Output Tokens", report_result["usage"].get("completion_tokens", "N/A"))
                    with col3:
                        st.metric("Total Tokens", report_result["usage"].get("total_tokens", "N/A"))
                
                st.download_button(
                    label="Скачать отчет",
                    data=report_result["content"],
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            else:
                st.error(f"Ошибка: {report_result['error']}")
                logger.error(f"Report error: {report_result['error']}")

st.write("---")
st.caption("MedAssistant CLD v1.0 | OpenRouter & Claude 3 Sonnet")
