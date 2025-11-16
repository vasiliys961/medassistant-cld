import pandas as pd

def analyze_lab(lab_input):
    if isinstance(lab_input, pd.DataFrame):
        df = lab_input
    else:
        df = pd.read_csv(lab_input)
    issues = []
    for _, row in df.iterrows():
        val = float(row['Значение'])
        norm_low, norm_high = [float(x) for x in row['Норма'].split('-')]
        if val < norm_low or val > norm_high:
            issues.append(f"{row['Параметр']}: {val} вне нормы ({row['Норма']})")
    return "Отклонения: " + "; ".join(issues), df.to_dict()
