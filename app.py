import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model_once():
    model = load_model("age_gender_best.h5")  # ✅ Best model
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = load_model_once()

age_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "50+"]

# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # ✅ BGR to RGB
    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_face(face):
    face_processed = preprocess_face(face)
    age_pred, gender_pred = model.predict(face_processed, verbose=0)
    age = age_labels[np.argmax(age_pred)]
    gender_prob = float(gender_pred[0][0])
    gender = "Female" if gender_prob > 0.5 else "Male"  # ✅ UTKFace: 0=Male, 1=Female
    confidence = gender_prob if gender == "Female" else 1 - gender_prob
    return age, gender, round(confidence * 100, 1)

# ===============================
# DRAW RESULTS ON IMAGE
# ===============================
def draw_results(img, faces):
    results = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        if face.size == 0:
            continue

        age, gender, confidence = predict_face(face)
        label = f"{gender}, {age}"

        # Green box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Label background
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img,
                      (x, y - text_h - 15),
                      (x + text_w, y),
                      (0, 255, 0), -1)

        # Label text
        cv2.putText(img, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        results.append((age, gender, confidence))

    return img, results

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Age & Gender Detection System")
st.markdown("Upload an image or use camera to detect **age** and **gender**.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

# ===============================
# TAB 1 — IMAGE UPLOAD
# ===============================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Image load nahi hui. Dobara try karo.")
            st.stop()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            st.warning("⚠️ Koi face detect nahi hua. Clear image upload karo.")
        else:
            img_result, results = draw_results(img, faces)
            st.image(
                cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.markdown("### 🔍 Detection Results")
            for idx, (age, gender, conf) in enumerate(results):
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Face {idx+1} — Gender", gender)
                col2.metric("Age Group", age)
                col3.metric("Confidence", f"{conf}%")

# ===============================
# TAB 2 — CAMERA
# ===============================
with tab2:
    st.info("📸 take picture — age aur gender detect  automatically.")
    camera_img = st.camera_input("Take a picture")

    if camera_img is not None:
        file_bytes = np.asarray(
            bytearray(camera_img.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("❌ Camera image not loading.")
            st.stop()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            st.warning("⚠️ no face deetcted")
        else:
            frame_result, results = draw_results(frame, faces)
            st.image(
                cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.markdown("### 🔍 Detection Results")
            for idx, (age, gender, conf) in enumerate(results):
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Face {idx+1} — Gender", gender)
                col2.metric("Age Group", age)
                col3.metric("Confidence", f"{conf}%")