import gradio as gr
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# ------------------------
# LOAD TFLITE MODEL
# ------------------------
MODEL_PATH = "model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

# ------------------------
# IMAGE PREPROCESSING
# ------------------------
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0

    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = np.expand_dims(img, axis=0)
    return img

def classify_xray(image):
    if image is None:
        return "No image uploaded.", ""

    processed = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return CLASS_NAMES[idx], f"{confidence:.2f}%"

# ------------------------
# SVM SYMPTOM CHECKER
# ------------------------
symptom_sentences = [
    "fever cough chest pain",
    "cough weight loss night sweats",
    "runny nose sneezing sore throat",
    "high fever severe cough fatigue",
    "fever headache vomiting",
    "no symptoms healthy"
]

symptom_labels = [
    "Pneumonia",
    "Tuberculosis",
    "Common Cold",
    "Corona Virus Disease",
    "Flu",
    "Normal"
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(symptom_sentences)

svm_model = SVC(kernel="linear")
svm_model.fit(X, symptom_labels)

def classify_symptoms(text):
    if text.strip() == "":
        return "Please enter symptoms."

    X_in = tfidf.transform([text])
    prediction = svm_model.predict(X_in)[0]
    return prediction

# ------------------------
# GRADIO UI
# ------------------------
with gr.Blocks(title="AI Health Assistant") as demo:
    
    gr.Markdown("# 🧠 AI Health Assistant")

    with gr.Tab("🩻 Chest X-Ray Classifier"):
        xray_input = gr.Image(type="pil", label="Upload Chest X-Ray")
        xray_button = gr.Button("Predict")
        xray_label = gr.Textbox(label="Prediction")
        xray_conf = gr.Textbox(label="Confidence")

        xray_button.click(fn=classify_xray, 
                          inputs=xray_input, 
                          outputs=[xray_label, xray_conf])

    with gr.Tab("🤒 Symptom Checker"):
        symptom_input = gr.Textbox(label="Enter symptoms (e.g., 'fever cough fatigue')")
        symptom_button = gr.Button("Predict Condition")
        symptom_output = gr.Textbox(label="Possible Condition")

        symptom_button.click(fn=classify_symptoms,
                             inputs=symptom_input,
                             outputs=symptom_output)

demo.launch()
