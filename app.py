cat > app.py << 'EOF'
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
        logger.info(f"Определён intent: {intent}")
        
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
                result["analysis"] = f"ЭКГ данные загружены. Количество отсчётов: {len(ecg_data)}"
                logger.info("ЭКГ успешно об
