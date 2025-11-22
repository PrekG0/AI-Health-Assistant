import streamlit as st
import numpy as np
from PIL import Image
from tflite_support import interpreter as tflite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# LOAD TFLITE CNN MODEL
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

# IMAGE PREPROCESSING
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0

    # Ensure RGB
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = np.expand_dims(img, axis=0)
    return img

# MINI SVM SYMPTOM DATASET
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

# Train SVM model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(symptom_sentences)

svm_model = SVC(kernel="linear")
svm_model.fit(X, symptom_labels)

def predict_symptom(text):
    X_input = tfidf.transform([text])
    return svm_model.predict(X_input)[0]

# STREAMLIT UI
st.title("🧠 AI Health Assistant")

option = st.sidebar.radio(
    "Choose a Feature:",
    ["Chest X-Ray Classifier", "Symptom Checker"]
)

# FEATURE 1 — CNN
if option == "Chest X-Ray Classifier":
    st.header("🩻 Chest X-Ray Disease Classifier")

    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        processed = preprocess_image(img)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        st.subheader("Prediction:")
        st.write(f"**{pred_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

# FEATURE 2 — SVM Model
if option == "Symptom Checker":
    st.header("🤒 Symptom-Based Disease Predictor")

    text = st.text_area("Enter symptoms (e.g., 'fever cough fatigue'):")

    if st.button("Predict Condition"):
        if text.strip() == "":
            st.error("Please type some symptoms.")
        else:
            result = predict_symptom(text)
            st.subheader("Possible Condition:")
            st.write(f"**{result}**")
