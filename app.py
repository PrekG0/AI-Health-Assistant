import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# LOAD CNN MODEL FOR X-RAY CLASSIFICATION
MODEL_PATH = "final_lung_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = {
    0: "Bacterial Pneumonia",
    1: "Corona Virus Disease",
    2: "Normal",
    3: "Tuberculosis",
    4: "Viral Pneumonia"
}

# IMAGE PREPROCESSING
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img)

    if img.ndim == 2:  # grayscale to RGB
        img = np.stack((img,) * 3, axis=-1)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# MINI SYMPTOM DATASET (YOU CAN EXPAND THIS ANYTIME)
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

# TRAIN SVM MODEL
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(symptom_sentences)

svm_model = SVC(kernel="linear")
svm_model.fit(X, symptom_labels)

def predict_symptom(text):
    X_input = tfidf.transform([text])
    return svm_model.predict(X_input)[0]

# STREAMLIT UI
st.title("🧠 AI Health Assistant")
st.write("Choose a feature from the sidebar.")

option = st.sidebar.radio(
    "Features:",
    ["Chest X-Ray Classifier", "Symptom Checker"]
)

# FEATURE 1 — X-RAY CLASSIFIER
if option == "Chest X-Ray Classifier":
    st.header("🩻 Chest X-Ray Disease Classifier")

    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded X-Ray", use_column_width=True)

        img_data = preprocess_image(img)
        prediction = model.predict(img_data)[0]

        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class] * 100

        st.subheader("Prediction:")
        st.write(f"**{CLASS_NAMES[pred_class]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

# FEATURE 2 — SVM SYMPTOM CHECKER
if option == "Symptom Checker":
    st.header("🤒 Symptom-Based Disease Predictor")

    text = st.text_area("Enter your symptoms (e.g., 'fever cough fatigue')")

    if st.button("Predict Condition"):
        if text.strip() == "":
            st.error("Please enter symptoms.")
        else:
            result = predict_symptom(text)
            st.subheader("Possible Condition:")
            st.write(f"**{result}**")
