import logging
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

def process_ecg(uploaded_file):
    """
    Обрабатывает ЭКГ данные из CSV/TXT файла.
    """
    try:
        logger.info(f"Обработка ЭКГ: {uploaded_file.name}")
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # .txt
            df = pd.read_csv(uploaded_file, delimiter='\t')
        
        # Преобразуем в словарь для передачи
        ecg_data = {
            'shape': str(df.shape),
            'columns': df.columns.tolist(),
            'preview': df.head(5).to_dict(),
            'statistics': df.describe().to_dict()
        }
        
        logger.info(f"ЭКГ обработана: {ecg_data['shape']}")
        return ecg_data
    
    except Exception as e:
        logger.error(f"Ошибка при обработке ЭКГ: {str(e)}", exc_info=True)
        raise
