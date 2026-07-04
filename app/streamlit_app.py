from __future__ import annotations

import streamlit as st
from PIL import Image

from cataract_classifier.config_loader import load_config
from cataract_classifier.inference.ensemble import average_probability_ensemble
from cataract_classifier.inference.predictor import preprocess_image
from cataract_classifier.models.registry import get_preprocess_fn, load_registry, load_trained_model


st.set_page_config(
    page_title="Cataract Detection Ensemble",
    page_icon="eye",
    layout="centered",
)

DATA_CONFIG = "configs/data_config.yaml"
MODEL_REGISTRY = "configs/model_registry.yaml"


@st.cache_resource
def load_models():
    registry = load_registry(MODEL_REGISTRY)
    return {model_name: load_trained_model(model_name, registry) for model_name in registry["models"]}


def ensemble_predict(image_pil, models, data_cfg, registry):
    predictions = {}
    for model_name, model in models.items():
        preprocess_fn = get_preprocess_fn(model_name, registry)
        processed = preprocess_image(image_pil, data_cfg["image_size"], preprocess_fn)
        predictions[model_name] = model.predict(processed, verbose=0)[0]
    labels = registry.get("class_labels", data_cfg["class_names"])
    return average_probability_ensemble(predictions, labels)


st.title("Cataract Detection System - Ensemble")
st.markdown("Upload an eye image to detect and classify cataracts using 4 models.")

with st.sidebar:
    st.info(
        """
        This system uses DenseNet121, EfficientNetB3, ResNet50, and VGG19.
        The predicted class is based on average confidence across all four models.
        Configure model paths in configs/model_registry.yaml.
        """
    )

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    if st.button("Analyze Image"):
        with st.spinner("Analyzing with all models..."):
            data_cfg = load_config(DATA_CONFIG)
            registry = load_registry(MODEL_REGISTRY)
            models = load_models()
            result, confidence, avg_probs = ensemble_predict(image_pil, models, data_cfg, registry)
        st.markdown(f"### Prediction: <span style='color:green'>{result}</span>", unsafe_allow_html=True)
        st.markdown(
            f"### Average Confidence: <span style='color:white'>{confidence * 100:.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.write("**Average class probabilities from all models:**")
        for label, prob in zip(registry.get("class_labels", data_cfg["class_names"]), avg_probs):
            st.write(f"- **{label}**: {prob * 100:.2f}%")
else:
    st.info("Please upload an eye image to begin analysis")
