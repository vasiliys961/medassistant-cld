import logging
import pandas as pd

logger = logging.getLogger(__name__)

def process_lab_analysis(uploaded_file):
    """
    Обрабатывает лабораторные анализы из CSV/XLSX.
    """
    try:
        logger.info(f"Обработка лабораторных анализов: {uploaded_file.name}")
        
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:  # .csv
            df = pd.read_csv(uploaded_file)
        
        lab_data = {
            'shape': str(df.shape),
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        logger.info(f"Лабораторные анализы обработаны: {lab_data['shape']}")
        return lab_data
    
    except Exception as e:
        logger.error(f"Ошибка при обработке лабораторных анализов: {str(e)}", exc_info=True)
        raise
