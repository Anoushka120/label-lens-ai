import os
import streamlit as st
from PIL import Image
from ocr_utils import extract_text_from_image
from analyzers import analyze_rules, analyze_openai, analyze_ml

st.set_page_config(page_title="LabelLens AI", page_icon="🧾")
st.title("🧾 LabelLens AI")
st.caption("Scan food labels and decide whether to buy or avoid")

mode = st.selectbox(
    "Analysis Mode",
    ["Rules (Offline)", "ML Model (Offline)", "OpenAI (Best Reasoning)"]
)

uploaded = st.file_uploader("Upload label image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_container_width=True)

    with st.spinner("Running OCR..."):
        text = extract_text_from_image(img)

    if not text.strip():
        st.warning("Could not read text properly.")
        st.stop()

    with st.expander("Extracted Text"):
        st.text(text)

    if mode == "Rules (Offline)":
        verdict, score, reasons = analyze_rules(text)
    elif mode == "ML Model (Offline)":
        verdict, score, reasons = analyze_ml(text)
    else:
        verdict, score, reasons = analyze_openai(text)

    st.subheader(verdict)
    st.metric("Health Score", f"{score}/100")

    for r in reasons:
        st.markdown(f"- {r}")
