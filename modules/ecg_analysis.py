import pandas as pd
from ecgdetectors import Detectors

def analyze_ecg(ecg_input, fs=500):
    """
    Анализирует ЭКГ-сигнал и находит R-пики.

    Параметры:
    - ecg_input: либо pandas.DataFrame, либо путь к csv-файлу.
      Данные должны быть в формате: ['time', 'signal']
    - fs: частота дискретизации в Гц (по умолчанию 500)

    Возвращает:
    - инфострока (число R-пиков) и словарь с их индексами
    """
    # Получение сигнала из DataFrame или файла
    if isinstance(ecg_input, pd.DataFrame):
        signal = ecg_input.iloc[:, 1].values
    else:  # file-like объект или путь к файлу
        df = pd.read_csv(ecg_input)
        signal = df.iloc[:, 1].values

    detectors = Detectors(fs)
    r_peaks = detectors.pan_tompkins_detector(signal)
    return f"R-пики: {len(r_peaks)}", {"R_peaks": list(r_peaks)}

