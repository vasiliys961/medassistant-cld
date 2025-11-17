import os
import logging
import httpx
import base64
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Конфиг OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.io/api/v1/chat/completions"

def analyze_image_with_openrouter(file):
    """
    Анализирует медицинское изображение через OpenRouter API с Claude Vision.
    """
    try:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY не установлен")
            return {
                "error": "OPENROUTER_API_KEY не установлен",
                "status_code": 400
            }
        
        # Читаем файл и конвертируем в base64
        file.seek(0)
        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Определяем тип медиа
        filename = file.name.lower()
        if filename.endswith('.png'):
            media_type = "image/png"
        elif filename.endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        elif filename.endswith('.bmp'):
            media_type = "image/bmp"
        else:
            media_type = "image/jpeg"  # по умолчанию
        
        logger.info(f"Анализ изображения: {file.name}")
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://medassistant-cld.local",
            "X-Title": "MedAssistant"
        }
        
        payload = {
            "model": "meta-llama/llama-3.2-90b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Проанализируй медицинское изображение (рентген, УЗИ, КТ, МРТ).
                            
Предоставь:
1. Описание видимых структур и патологических изменений
2. Предварительные выводы и диагностические возможности
3. Рекомендации по дополнительным исследованиям
4. Ссылки на медицинские стандарты (если применимо)

Формат: структурированный отчёт."""
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(OPENROUTER_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            analysis = {
                "success": True,
                "analysis": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {}),
                "status_code": 200
            }
            logger.info("Анализ изображения успешно завершён")
            return analysis
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "status_code": response.status_code,
                "success": False
            }
    
    except httpx.TimeoutException:
        error_msg = "Timeout: анализ изображения занял слишком много времени"
        logger.error(error_msg)
        return {"error": error_msg, "status_code": 504, "success": False}
    
    except Exception as e:
        error_msg = f"Ошибка при анализе изображения: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg, "status_code": 500, "success": False}


# Для обратной совместимости
def analyze_image_with_hf_api(file):
    """
    Deprecated: используйте analyze_image_with_openrouter вместо этого
    """
    logger.warning("analyze_image_with_hf_api deprecated, используется OpenRouter вместо этого")
    return analyze_image_with_openrouter(file)
