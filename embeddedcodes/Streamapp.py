# ------------------- IMPORTS -------------------
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras

# ------------------- STREAMLIT CONFIG -------------------
# Must be first Streamlit command in the file
st.set_page_config(
    page_title="VisionAI – Retina & DR Detection",
    page_icon="🩺",
    layout="centered"
)

# ------------------- MODEL LOADING -------------------
@st.cache_resource
def load_models():
    retina_model = keras.models.load_model("ModelRatina.h5")
    dr_model = keras.models.load_model("ModelDR_MobileNet2.h5")
    return retina_model, dr_model

retina_model, dr_model = load_models()

# ------------------- IMAGE PREPROCESSING -------------------
IMG_SIZE = (224, 224)

def preprocess_image(pil_img):
    """Convert PIL image → model input"""
    img = np.array(pil_img)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    return img

# ------------------- PREDICTION FUNCTIONS -------------------
def predict_retina(image):
    labels = ['non-retinal', 'retinal']
    img = preprocess_image(image)
    pred = retina_model.predict(img)
    confidence = float(np.max(pred))
    cls = labels[int(np.argmax(pred))]
    return cls, confidence

def predict_dr(image):
    labels = ['No DR', 'DR']
    img = preprocess_image(image)
    pred = dr_model.predict(img)
    confidence = float(np.max(pred))
    cls = labels[int(np.argmax(pred))]
    return cls, confidence

# ------------------- STREAMLIT UI -------------------
st.markdown("""
# 🩺 VisionAI  
### AI-Powered Retina & Diabetic Retinopathy Detection
Upload an image or capture from camera to analyze eye health in real time.
""")

# ------------------- IMAGE INPUT -------------------
st.subheader("📸 Provide an Image")
input_method = st.radio("Choose input method:", ["Upload Image", "Use Camera"])
image = None

# Upload image
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# Camera capture
elif input_method == "Use Camera":
    camera_img = st.camera_input("Capture an Image")
    if camera_img:
        image = Image.open(camera_img)

# ------------------- PREDICTION PIPELINE -------------------
if image:
    # Create two columns: left for image, right for results
    img_col, result_col = st.columns([1, 1], gap="medium")

    with img_col:
        st.image(image, caption="Input Image", use_container_width=True)

    with result_col:
        # Step 1: Retina detection
        with st.spinner("Step 1: Checking if this is a Retina Image..."):
            retina_result, retina_conf = predict_retina(image)

        st.subheader("🔍 Retina Detection Result")
        st.write(f"**Result:** {retina_result}")
        st.write(f"**Confidence:** `{retina_conf:.2f}`")

        if retina_result == "retinal":
            st.success("✔ Retina detected. Proceeding to DR analysis...")

            # Step 2: DR detection
            with st.spinner("Step 2: Analyzing for Diabetic Retinopathy..."):
                dr_result, dr_conf = predict_dr(image)

            st.subheader("🧪 DR Detection Result")
            st.write(f"**Result:** {dr_result}")
            st.write(f"**Confidence:** `{dr_conf:.2f}`")

            if dr_result == "DR":
                st.error("⚠ DR Detected! Please consult an eye specialist.")
            else:
                st.success("✔ No signs of Diabetic Retinopathy detected.")
        else:
            st.error("❌ This is NOT a retina image. Please provide a valid retinal fundus photo.")


