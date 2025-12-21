import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

from model import load_model, load_class_names
from utils import preprocess_image
from gradcam import GradCAM

# ------------------ Page setup ------------------
st.set_page_config(
    page_title="Gesture Classification",
    layout="centered"
)

st.title("✋ Gesture Classification — InceptionV3")
st.write("Upload an image or use your webcam to classify the hand gesture and view Grad-CAM.")

mode = st.radio(
    "Choose input method:",
    ["Upload Image", "Use Webcam"]
)

# ------------------ Load model & classes ------------------
@st.cache_resource
def load_resources():
    class_names = load_class_names("data/class_names.json")
    model = load_model("models/inception_best.pth", len(class_names))
    cam = GradCAM(model, model.Mixed_7c)
    return model, class_names, cam


model, class_names, gradcam = load_resources()

# =========================================================
#                   UPLOAD IMAGE MODE
# =========================================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # -------- Prediction --------
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🔮 Top Predictions")
        for idx in top3_idx:
            st.write(f"**{class_names[idx]}** — {probs[idx]*100:.2f}%")

        # -------- Grad-CAM --------
        cam = gradcam.generate(input_tensor)

        img_np = np.array(image.resize((299, 299))) / 255.0
        cam_resized = cv2.resize(cam, (299, 299))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )

        overlay = heatmap * 0.4 + img_np * 255

        st.subheader("🔥 Grad-CAM Visualization")
        st.image(overlay.astype(np.uint8), use_column_width=True)

# =========================================================
#                   WEBCAM BONUS MODE
# =========================================================
elif mode == "Use Webcam":
    st.subheader("📷 Webcam Capture")
    st.info("Allow camera access when prompted.")

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        st.image(image, caption="Captured Image", use_column_width=True)

        # -------- Prediction --------
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🔮 Top Predictions")
        for idx in top3_idx:
            st.write(f"**{class_names[idx]}** — {probs[idx]*100:.2f}%")

        # -------- Grad-CAM --------
        cam = gradcam.generate(input_tensor)

        img_np = np.array(image.resize((299, 299))) / 255.0
        cam_resized = cv2.resize(cam, (299, 299))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )

        overlay = heatmap * 0.4 + img_np * 255

        st.subheader("🔥 Grad-CAM Visualization")
        st.image(overlay.astype(np.uint8), use_column_width=True)