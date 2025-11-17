import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_intent(task_description: str, filename: str) -> str:
    """
    Определяет тип анализа на основе описания задачи и имени файла.
    
    Returns:
        str: 'ecg', 'image', 'lab', 'document' или 'unknown'
    """
    
    task_lower = task_description.lower()
    filename_lower = filename.lower()
    
    # Ключевые слова для каждого типа
    ecg_keywords = ['экг', 'ecg', 'кардио', 'сердце', 'электрокардиограмма', 'ритм', 'пульс']
    image_keywords = ['рентген', 'узи', 'снимок', 'томография', 'кт', 'мрт', 'image', 'xray', 'ultrasound', 'jpg', 'png', 'jpeg']
    lab_keywords = ['анализ', 'кровь', 'lab', 'тест', 'биохимия', 'гемоглобин', 'лейкоциты', 'glucose', 'csv', 'xlsx']
    document_keywords = ['документ', 'pdf', 'текст', 'выписка', 'диагноз', 'document', 'ocr']
    
    # Проверяем расширение файла
    if filename_lower.endswith(('.csv', '.txt')) and any(kw in task_lower for kw in ecg_keywords):
        logger.info("Intent: ECG")
        return 'ecg'
    
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp')) and any(kw in task_lower for kw in image_keywords):
        logger.info("Intent: Image")
        return 'image'
    
    elif filename_lower.endswith(('.xlsx', '.xls', '.csv')) and any(kw in task_lower for kw in lab_keywords):
        logger.info("Intent: Lab")
        return 'lab'
    
    elif filename_lower.endswith(('.pdf', '.png', '.jpg', '.jpeg')) and any(kw in task_lower for kw in document_keywords):
        logger.info("Intent: Document")
        return 'document'
    
    # Проверяем только по расширению файла
    elif filename_lower.endswith(('.csv', '.txt')):
        logger.info("Intent: ECG (по расширению)")
        return 'ecg'
    
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        logger.info("Intent: Image (по расширению)")
        return 'image'
    
    elif filename_lower.endswith(('.xlsx', '.xls')):
        logger.info("Intent: Lab (по расширению)")
        return 'lab'
    
    elif filename_lower.endswith('.pdf'):
        logger.info("Intent: Document (по расширению)")
        return 'document'
    
    logger.warning(f"Intent не определён для файла: {filename}")
    return 'unknown'
