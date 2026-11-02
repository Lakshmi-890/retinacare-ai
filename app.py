
import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="RetinaCare AI",
    page_icon="👁️",
    layout="centered"
)

@st.cache_resource
def load_models():
    cnn_model = load_model(
        "retinacare_feature_extractor.keras",
        compile=False
    )
    svm_model = joblib.load("retinacare_svm.pkl")
    return cnn_model, svm_model

cnn_model, svm_model = load_models()

st.title("👁️ RetinaCare AI")
st.subheader("Diabetic Retinopathy Screening")

st.write(
    "Upload a retinal fundus image to obtain an "
    "AI-generated screening prediction."
)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image",
             use_container_width=True)

    if st.button("Analyze Image"):

        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        features = cnn_model.predict(
            image_array, verbose=0
        )

        prediction = svm_model.predict(features)[0]
        probability = svm_model.predict_proba(features)[0][1]

        st.subheader("Screening Result")

        if prediction == 0:
            st.success("Non-Referable DR")
        else:
            st.warning("Referable DR")

        st.write(
            f"Estimated probability of Referable DR: "
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "Academic prototype only. This result is not a "
            "medical diagnosis. Consult an eye-care professional."
        )
