import pandas as pd
from pyecgdetectors import Detectors

def analyze_ecg(ecg_input):
    if isinstance(ecg_input, pd.DataFrame):
        signal = ecg_input.iloc[:,1].values
    else:  # file-like
        df = pd.read_csv(ecg_input)
        signal = df.iloc[:,1].values
    fs = 500
    detectors = Detectors(fs)
    r_peaks = detectors.pan_tompkins_detector(signal)
    return f"R-пики: {len(r_peaks)}", {"R_peaks": list(r_peaks)}
