from pathlib import Path
import sys

import pandas as pd
from PIL import Image
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.inference import load_v4_checkpoint, predict  # noqa: E402


DEFAULT_CHECKPOINT = BASE_DIR / "models" / "best_multimodal_v4.pt"

st.set_page_config(page_title="Meme Sentiment", page_icon="🎭", layout="centered")
st.title("🎭 Multimodal Meme Sentiment")
st.caption("DistilBERT text features + pretrained ResNet18 image features")


@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str):
    return load_v4_checkpoint(checkpoint_path)


with st.sidebar:
    st.header("Model")
    checkpoint = st.text_input("V4 checkpoint", str(DEFAULT_CHECKPOINT))
    st.caption("The checkpoint is created by src/multimodal_train_v4.py.")

uploaded_file = st.file_uploader("Upload a meme", type=["png", "jpg", "jpeg", "webp"])
meme_text = st.text_area(
    "Meme text",
    placeholder="Paste the text shown in the meme...",
    help="The Memotion model was trained with both the image and its corrected text.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded meme", use_container_width=True)

    if st.button("Predict sentiment", type="primary", use_container_width=True):
        if not meme_text.strip():
            st.warning("Enter the meme text so both modalities are available.")
        else:
            try:
                model, tokenizer, device, checkpoint_data = load_model(checkpoint)
                result = predict(model, tokenizer, image, meme_text, device)
            except Exception as exc:
                st.error(f"Could not run prediction: {exc}")
            else:
                st.success(
                    f"Prediction: **{result['label'].title()}** — "
                    f"confidence **{result['confidence']:.1%}**"
                )
                chart_data = pd.DataFrame(
                    {"probability": result["probabilities"]}
                )
                st.bar_chart(chart_data, y="probability", horizontal=True)

                val_f1 = checkpoint_data.get("val_macro_f1")
                val_accuracy = checkpoint_data.get("val_accuracy")
                if val_f1 is not None and val_accuracy is not None:
                    st.caption(
                        f"Checkpoint validation macro-F1: {val_f1:.4f} | "
                        f"accuracy: {val_accuracy:.4f}"
                    )
else:
    st.info("Upload a meme and enter its text to begin.")
