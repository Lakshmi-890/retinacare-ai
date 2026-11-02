import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="RetinaCare AI",
    page_icon="👁️",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F8FB;
}
[data-testid="stSidebar"] {
    background-color: #102A43;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #102A43;
}
.subtitle {
    color: #526777;
    font-size: 17px;
}
.section-title {
    color: #102A43;
    font-size: 25px;
    font-weight: 700;
}
.info-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E1EAF0;
    margin-bottom: 12px;
}
.result-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    border-left: 5px solid #159A9C;
}
div.stButton > button {
    background-color: #159A9C;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 25px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD TRAINED MODELS
# ----------------------------------
@st.cache_resource
def load_models():
    cnn = load_model(
        "retinacare_feature_extractor.keras",
        compile=False
    )
    svm = joblib.load("retinacare_svm.pkl")
    return cnn, svm


# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------
st.sidebar.title("👁️ RetinaCare AI")
st.sidebar.caption("Intelligent Retinal Screening")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Learn About DR",
        "AI Technology",
        "Screening",
        "Understanding Results",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "An academic AI-based diabetic retinopathy "
    "screening prototype."
)


# ----------------------------------
# HOME PAGE
# ----------------------------------
if page == "Home":

    st.markdown(
        '<div class="main-title">RetinaCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Powered Diabetic Retinopathy Screening</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("""
    <div class="info-card">
    <h2 style="color:#102A43;">Protecting Vision Through AI</h2>
    <p style="color:#526777;font-size:17px;">
    RetinaCare AI is an artificial intelligence-based academic
    prototype designed to analyze retinal fundus images and
    classify them into Non-Referable and Referable Diabetic
    Retinopathy categories.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore RetinaCare AI")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📖 Learn")
        st.write(
            "Understand diabetic retinopathy, its causes, "
            "symptoms and stages."
        )

    with c2:
        st.markdown("### 🧠 AI Technology")
        st.write(
            "Discover how EfficientNetB0 and SVM work together."
        )

    with c3:
        st.markdown("### 🔬 Screening")
        st.write(
            "Upload a retinal image and view the model's prediction."
        )

    st.markdown("---")

    st.subheader("Why is Early Screening Important?")

    st.write("""
    Diabetic retinopathy is an eye complication associated
    with diabetes. It can damage the blood vessels in the
    retina and may cause vision problems.

    Early detection and regular eye examinations can help
    identify retinal changes and support timely medical care.
    """)

    st.info(
        "Use the sidebar to explore educational information "
        "and the retinal image screening feature."
    )


# ----------------------------------
# LEARN ABOUT DIABETIC RETINOPATHY
# ----------------------------------
elif page == "Learn About DR":

    st.markdown(
        '<div class="main-title">Learn About Diabetic Retinopathy</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand the disease, its causes, symptoms, "
        "stages and screening methods."
    )

    with st.expander("1. What is Diabetic Retinopathy?", expanded=True):
        st.write("""
        Diabetic Retinopathy (DR) is an eye complication
        caused by diabetes. Prolonged high blood glucose
        levels can damage the tiny blood vessels in the retina.

        The retina is the light-sensitive tissue at the back
        of the eye. Damage to its blood vessels can cause
        leakage, blockage and abnormal blood vessel growth.

        If the condition progresses without appropriate
        medical care, it can lead to serious vision problems.
        """)

    with st.expander("2. Understanding the Human Retina"):
        st.write("""
        The retina is a thin layer of tissue located at the
        back of the eye. It receives light and converts it
        into signals that travel through the optic nerve
        to the brain.

        Important retinal structures:

        - Macula: Responsible for detailed central vision.
        - Optic Disc: Where the optic nerve connects to the retina.
        - Blood Vessels: Supply oxygen and nutrients.
        - Photoreceptors: Detect light and help create visual information.
        """)

    with st.expander("3. Causes and Risk Factors"):
        st.write("""
        Diabetic Retinopathy is associated with damage to
        retinal blood vessels over time.

        Major risk factors include:

        - Prolonged high blood glucose levels.
        - Long duration of diabetes.
        - High blood pressure.
        - High cholesterol.
        - Pregnancy in people with diabetes.
        - Inadequate diabetes management.
        """)

    with st.expander("4. Stages of Diabetic Retinopathy"):
        st.write("""
        Diabetic Retinopathy is generally classified into
        five severity levels:
        """)

        stages = [
            ("Grade 0 - No DR",
             "No visible signs of diabetic retinopathy."),
            ("Grade 1 - Mild NPDR",
             "Small retinal microaneurysms may appear."),
            ("Grade 2 - Moderate NPDR",
             "More retinal blood vessel abnormalities may occur."),
            ("Grade 3 - Severe NPDR",
             "Extensive retinal blood vessel damage is present."),
            ("Grade 4 - Proliferative DR",
             "Abnormal new blood vessels develop on the retina.")
        ]

        for title, description in stages:
            st.markdown(f"**{title}**")
            st.write(description)

    with st.expander("5. Common Symptoms"):
        st.write("""
        Early diabetic retinopathy may not cause noticeable
        symptoms.

        Possible symptoms in more advanced stages include:

        - Blurred or fluctuating vision.
        - Dark spots or floaters.
        - Difficulty seeing at night.
        - Reduced colour vision.
        - Distorted vision.
        - Sudden vision loss.

        Regular eye examinations are important even when
        symptoms are absent.
        """)

    with st.expander("6. Diagnosis and Screening"):
        st.write("""
        Common examination and screening methods include:

        - Fundus Photography: Captures retinal photographs.
        - Dilated Eye Examination: Allows an eye-care professional
          to examine the retina.
        - Optical Coherence Tomography (OCT): Produces
          cross-sectional images of retinal structures.
        - AI-Based Screening: Uses machine learning to identify
          patterns associated with diabetic retinopathy.
        """)

    with st.expander("7. Prevention and Management"):
        st.write("""
        Risk can be reduced through appropriate diabetes
        management and regular eye care.

        - Maintain blood glucose levels within recommended targets.
        - Monitor blood pressure and cholesterol.
        - Attend regular retinal examinations.
        - Follow prescribed treatment.
        - Avoid smoking.
        - Seek prompt care for sudden vision changes.
        """)


# ----------------------------------
# AI TECHNOLOGY PAGE
# ----------------------------------
elif page == "AI Technology":

    st.markdown(
        '<div class="main-title">AI Technology</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Our system combines deep learning feature extraction "
        "with traditional machine learning classification."
    )

    st.subheader("1. Convolutional Neural Network (CNN)")

    st.write("""
    A Convolutional Neural Network is a deep learning
    architecture designed to process image data.

    It automatically learns visual patterns such as edges,
    textures, shapes and structures from images.
    """)

    st.subheader("2. EfficientNetB0")

    st.write("""
    EfficientNetB0 is a pretrained CNN architecture that
    uses compound scaling to balance network depth, width
    and image resolution.

    In RetinaCare AI, EfficientNetB0 is used as a frozen
    feature extractor. It converts each retinal image into
    a representation containing 1280 features.
    """)

    st.subheader("3. Support Vector Machine (SVM)")

    st.write("""
    Support Vector Machine is a supervised machine learning
    algorithm used for classification.

    It identifies a decision boundary that separates
    different classes based on their features.

    Our project uses a linear SVM to classify retinal
    image features into two categories.
    """)

    st.subheader("4. Model Workflow")

    workflow = [
        "Retinal Fundus Image",
        "Image Resizing (224 × 224)",
        "EfficientNetB0 Feature Extraction",
        "StandardScaler",
        "Linear SVM Classification",
        "Screening Prediction"
    ]

    for i, item in enumerate(workflow):
        st.markdown(f"**{i+1}. {item}**")
        if i < len(workflow) - 1:
            st.write("⬇️")

    st.subheader("5. Classification Categories")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Class 0: Non-Referable DR")
        st.write("Original grades 0 and 1.")

    with col2:
        st.warning("Class 1: Referable DR")
        st.write("Original grades 2, 3 and 4.")


# ----------------------------------
# SCREENING PAGE
# ----------------------------------
elif page == "Screening":

    st.markdown(
        '<div class="main-title">Retinal Image Screening</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a retinal fundus photograph to obtain "
        "an AI-generated screening prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Retinal Image",
            width=450
        )

        if st.button("Analyze Retinal Image"):

            try:
                with st.spinner("Analyzing retinal image..."):

                    cnn_model, svm_model = load_models()

                    resized_image = image.resize((224, 224))

                    image_array = np.array(resized_image)
                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    features = cnn_model.predict(
                        image_array, verbose=0
                    )

                    prediction = svm_model.predict(features)[0]

                    probability = svm_model.predict_proba(
                        features
                    )[0][1]

                st.session_state["prediction"] = int(prediction)
                st.session_state["probability"] = float(probability)

                st.success("Image analysis completed.")

            except Exception as e:
                st.error("Unable to analyze the image.")
                st.exception(e)

    if "prediction" in st.session_state:

        st.markdown("---")
        st.subheader("Screening Result")

        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]

        if prediction == 0:
            st.success("Non-Referable DR")
            st.write(
                "The model classified this image as "
                "Non-Referable Diabetic Retinopathy."
            )
        else:
            st.warning("Referable DR")
            st.write(
                "The model classified this image as "
                "Referable Diabetic Retinopathy."
            )

        st.metric(
            "Estimated probability of Referable DR",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        st.caption(
            "This probability is a model score, not a "
            "clinically validated measure of disease certainty."
        )

        st.info(
            "This result is for educational screening only. "
            "Please consult an ophthalmologist for professional "
            "evaluation."
        )


# ----------------------------------
# UNDERSTANDING RESULTS
# ----------------------------------
elif page == "Understanding Results":

    st.markdown(
        '<div class="main-title">Understanding Your Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Learn what the two output categories of "
        "RetinaCare AI mean."
    )

    st.subheader("Non-Referable DR")

    st.success("Class 0")

    st.write("""
    The model predicts that the uploaded image belongs
    to the Non-Referable category.

    This does not guarantee that the retina is healthy
    or that diabetic retinopathy is absent.
    """)

    st.subheader("Referable DR")

    st.warning("Class 1")

    st.write("""
    The model predicts that the uploaded image belongs
    to the Referable category.

    This result indicates that professional eye evaluation
    is important.
    """)

    st.subheader("Understanding Probability")

    st.write("""
    The displayed probability is the SVM model's estimated
    probability for the Referable DR class.

    It is not the percentage of retinal damage and does
    not represent clinical certainty.
    """)

    st.info(
        "A prediction from RetinaCare AI must never replace "
        "a professional eye examination."
    )


# ----------------------------------
# ABOUT PROJECT PAGE
# ----------------------------------
elif page == "About Project":

    st.markdown(
        '<div class="main-title">About RetinaCare AI</div>',
        unsafe_allow_html=True
    )

    st.subheader("Project Objective")

    st.write("""
    The objective of RetinaCare AI is to develop an academic
    prototype that applies deep learning and machine learning
    techniques to retinal fundus image classification.

    The system combines EfficientNetB0 feature extraction
    and SVM classification to identify images belonging to
    Non-Referable and Referable DR categories.
    """)

    st.subheader("Dataset")

    st.write("""
    This project uses the Indian Diabetic Retinopathy Image
    Dataset (IDRiD).

    The original five severity grades are converted into
    two binary classes:

    - Grades 0 and 1: Non-Referable DR.
    - Grades 2, 3 and 4: Referable DR.
    """)

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "TensorFlow / Keras",
        "EfficientNetB0",
        "Support Vector Machine",
        "Scikit-learn",
        "Streamlit",
        "Google Colab"
    ]

    for tech in technologies:
        st.markdown(f"- {tech}")

    st.subheader("Project Limitations")

    st.write("""
    This is an academic prototype trained and evaluated
    using a specific dataset. Its performance may vary
    on images from other sources, cameras or populations.

    It has not been established as a clinically validated
    diagnostic system.
    """)


# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#526777;">
        <b>RetinaCare AI</b><br>
        AI-Powered Diabetic Retinopathy Screening<br><br>
        Academic Prototype | Not a Medical Diagnosis
    </div>
    """,
    unsafe_allow_html=True
)
