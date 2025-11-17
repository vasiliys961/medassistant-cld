import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

def analyze_lab_results(uploaded_file):
    """
    Анализирует результаты лабораторных тестов.
    """
    try:
        logger.info(f"Анализ лабораторных результатов: {uploaded_file.name}")
        
        # Читаем файл в зависимости от расширения
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError(f"Неподдерживаемый формат: {uploaded_file.name}")
        
        # Базовый анализ
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': df.isnull().sum().to_dict(),
            'preview': df.head().to_dict()
        }
        
        logger.info(f"Анализ завершён: {analysis['total_rows']} строк, {analysis['total_columns']} столбцов")
        return analysis
    
    except Exception as e:
        logger.error(f"Ошибка при анализе лабораторных результатов: {str(e)}", exc_info=True)
        raise
