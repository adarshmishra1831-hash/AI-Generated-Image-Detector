import os
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from src.model import AIImageDetector
from src.gradcam import GradCAM, overlay_heatmap, preprocess_image
MODEL_PATH  = "outputs/best_model.pth"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Real", "Fake / AI-Generated"]
CLASS_EMOJI = ["✅", "🤖"]
@st.cache_resource
def load_model():
    model = AIImageDetector(num_classes=2, pretrained=False)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
def predict(model, pil_img: Image.Image):
    tensor = preprocess_image(pil_img).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item() * 100
    real_prob   = probs[0].item() * 100
    fake_prob   = probs[1].item() * 100
    return pred_idx, confidence, real_prob, fake_prob, tensor
def get_gradcam(model, pil_img: Image.Image, tensor: torch.Tensor):
    target_layer = model.backbone.blocks[-1]
    gradcam      = GradCAM(model, target_layer)
    tensor_grad = preprocess_image(pil_img).to(DEVICE)
    tensor_grad.requires_grad_()
    cam          = gradcam.generate(tensor_grad)
    img_np       = np.array(pil_img.resize((224, 224)))
    img_bgr      = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    heatmap_img  = overlay_heatmap(img_bgr, cam)
    return heatmap_img
def main():
    st.set_page_config(
        page_title="AI Image Detector",
        page_icon="🕵️",
        layout="wide",
    )
    st.title("🕵️ AI-Generated Image Detector")
    st.markdown(
        "Upload an image to find out whether it's a **real photograph** "
        "or **AI-generated** using deep learning (EfficientNet-B3 + Grad-CAM)."
    )
    st.divider()
    with st.sidebar:
        st.header("⚙️ Settings")
        show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
        st.info(
            "**Grad-CAM** highlights which regions of the image "
            "influenced the model's decision most."
        )
        st.markdown("---")
        st.markdown("**Model:** EfficientNet-B3")
        st.markdown("**Device:** " + DEVICE.upper())
        st.markdown("**Classes:** Real / Fake")
    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return
    pil_img = Image.open(uploaded).convert("RGB")
    model   = load_model()
    with st.spinner("Analyzing image..."):
        pred_idx, confidence, real_prob, fake_prob, tensor = predict(
            model, pil_img)
        heatmap = get_gradcam(model, pil_img, tensor) if show_gradcam else None
    col1, col2, col3 = st.columns([1.2, 1.2, 1.4])
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(pil_img, use_column_width=True)
    with col2:
        st.subheader("🔍 Prediction")
        label  = CLASS_NAMES[pred_idx]
        emoji  = CLASS_EMOJI[pred_idx]
        color  = "green" if pred_idx == 0 else "red"
        st.markdown(
            f"<h2 style='color:{color};'>{emoji} {label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.1f}%")
        st.markdown("**Class Probabilities**")
        st.progress(real_prob / 100,
                    text=f"✅ Real: {real_prob:.1f}%")
        st.progress(fake_prob / 100,
                    text=f"🤖 Fake: {fake_prob:.1f}%")
        if pred_idx == 1:
            st.error("⚠️ This image shows signs of AI generation.")
        else:
            st.success("✅ This image appears to be a real photograph.")

    with col3:
        if show_gradcam and heatmap is not None:
            st.subheader("🌡️ Grad-CAM Heatmap")
            st.image(heatmap, use_column_width=True,
                     caption="Regions influencing the model's decision")
        else:
            st.subheader("ℹ️ About the Model")
            st.markdown
            ("""
            - **Backbone:** EfficientNet-B3 (pretrained on ImageNet)
            - **Fine-tuned** on real vs AI-generated images
            - **Grad-CAM** provides visual explainability
            - Detects images from DALL·E, Midjourney, Stable Diffusion, etc.
            """)
    st.divider()
    with st.expander("📊 Image Details"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Width",  f"{pil_img.width}px")
        col_b.metric("Height", f"{pil_img.height}px")
        col_c.metric("Mode",   pil_img.mode)
if __name__ == "__main__":
    main()
