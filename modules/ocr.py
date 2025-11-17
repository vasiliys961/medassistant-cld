import logging
import PyPDF2
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

def extract_text_from_image(uploaded_file):
    """
    Извлекает текст из изображения или PDF с помощью OCR.
    """
    try:
        logger.info(f"Извлечение текста из: {uploaded_file.name}")
        
        if uploaded_file.name.lower().endswith('.pdf'):
            # Работа с PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            logger.info(f"Текст из PDF извлечён: {len(text)} символов")
            return text
        
        elif uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Работа с изображением
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image, lang='rus+eng')
            logger.info(f"Текст из изображения извлечён: {len(text)} символов")
            return text
        
        else:
            logger.warning(f"Неподдерживаемый формат: {uploaded_file.name}")
            return ""
    
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста: {str(e)}", exc_info=True)
        raise
