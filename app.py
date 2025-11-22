import streamlit as st
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# TFLITE SUPPORT TASK API
from tflite_support.task import vision

# Load TFLite model with Task API
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (128, 128)

classifier = vision.ImageClassifier.create_from_file(MODEL_PATH)

CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

# IMAGE PREPROCESSING + PREDICTION
def classify_image(img):
    img = img.resize(IMAGE_SIZE)
    img = np.array(img)

    tensor = vision.TensorImage.create_from_array(img)
    result = classifier.classify(tensor)

    top = result.classifications[0].categories[0]
    return top.index, top.score * 100


# SVM SYMPTOM CHECKER
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

def predict_symptom(text):
    X_in = tfidf.transform([text])
    return svm_model.predict(X_in)[0]


# UI
st.title("🧠 AI Health Assistant")

option = st.sidebar.radio(
    "Choose a Feature:",
    ["Chest X-Ray Classifier", "Symptom Checker"]
)

# X-RAY CNN FEATURE
if option == "Chest X-Ray Classifier":
    st.header("🩻 Chest X-Ray Disease Classifier")

    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        pred_idx, confidence = classify_image(img)
        pred_class = CLASS_NAMES[pred_idx]

        st.subheader("Prediction:")
        st.write(f"**{pred_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")


# SYMPTOM FEATURE
if option == "Symptom Checker":
    st.header("🤒 Symptom-Based Disease Predictor")

    text = st.text_area("Enter symptoms (e.g., 'fever cough fatigue'): ")

    if st.button("Predict Condition"):
        if text.strip() == "":
            st.error("Please type some symptoms.")
        else:
            result = predict_symptom(text)
            st.subheader("Possible Condition:")
            st.write(f"**{result}**")
