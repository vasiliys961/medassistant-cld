def predict_intent(user_task, uploaded_file):
    if uploaded_file:
        name = uploaded_file.name.lower()
        if name.endswith(("csv","xml")):
            return "ecg"
        elif name.endswith(("jpg", "jpeg", "png", "tiff", "pdf")):
            # could be ECG (scan) or lab (paper) or image
            if "экг" in user_task.lower():
                return "ecg"
            elif "анализ" in user_task.lower():
                return "lab"
            return "image"
        elif name.endswith(("dcm",)):
            return "image"
    if "экг" in user_task.lower():  return "ecg"
    if "рентген" in user_task.lower() or "кт" in user_task.lower():  return "image"
    if "анализ" in user_task.lower() or "днк" in user_task.lower():  return "lab"
    return "general"
