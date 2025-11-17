import logging
from PIL import Image
import streamlit as st

logger = logging.getLogger(__name__)

def process_image(uploaded_file):
    """
    Обрабатывает медицинское изображение.
    """
    try:
        logger.info(f"Обработка изображения: {uploaded_file.name}")
        
        image = Image.open(uploaded_file)
        
        image_data = {
            'filename': uploaded_file.name,
            'format': image.format,
            'size': image.size,
            'mode': image.mode
        }
        
        logger.info(f"Изображение обработано: {image_data['size']}")
        return image_data
    
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        raise
