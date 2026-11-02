
import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RetinaCare AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f4f7fb;
    color: #243447;
}

/* Main content */
section[data-testid="stMain"] {
    background-color: #f4f7fb;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* General text visibility */
.stApp p,
.stApp li,
.stApp label,
.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #334155;
}

/* Headings */
h1, h2, h3, h4 {
    color: #102a43 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #102b45;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #49647d;
}

/* Sidebar radio buttons */
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 500;
}

/* Cards */
.info-card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
    box-shadow: 0 3px 12px rgba(15, 40, 65, 0.05);
    color: #334155;
}

.info-card h3 {
    color: #102a43 !important;
    margin-top: 0;
}

.info-card p,
.info-card li {
    color: #475569 !important;
    line-height: 1.7;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #123653, #087e8b);
    padding: 38px;
    border-radius: 18px;
    color: white !important;
    margin-bottom: 25px;
}

.hero h1,
.hero h2,
.hero p {
    color: white !important;
}

.hero p {
    font-size: 17px;
    line-height: 1.7;
}

/* Section headings */
.section-heading {
    color: #087e8b !important;
    font-weight: 700;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #dce5ed;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    color: #243447 !important;
}

/* Buttons */
.stButton > button {
    background-color: #087e8b;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #066873;
    color: white !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 1px dashed #9aabbc;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #102a43 !important;
}

/* Alerts */
.stAlert p {
    color: #243447 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b !important;
    font-size: 13px;
    padding: 25px 0;
    border-top: 1px solid #dce5ed;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    cnn = load_model(
        "retinacare_feature_extractor.keras",
        compile=False
    )

    svm = joblib.load("retinacare_svm.pkl")

    return cnn, svm


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='color:white;'>👁️ RetinaCare AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#cbd5e1;'>Intelligent Retinal Screening</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select a page",
        [
            "Home",
            "Learn About DR",
            "AI Technology",
            "Screening",
            "Understanding Results",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.info(
        "An academic AI-based diabetic retinopathy screening prototype."
    )


# --------------------------------------------------
# REUSABLE FOOTER
# --------------------------------------------------

def show_footer():

    st.markdown("""
    <div class="footer">
        RetinaCare AI | Academic Project Prototype<br>
        AI-assisted diabetic retinopathy screening
        <br><br>
        This application is not a substitute for professional
        medical examination or diagnosis.
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>👁️ RetinaCare AI</h1>
        <h2>Intelligent Diabetic Retinopathy Screening</h2>
        <p>
        An AI-powered academic prototype designed to analyze
        retinal fundus images and identify signs associated
        with referable diabetic retinopathy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Welcome to RetinaCare AI")

    st.write("""
    Diabetic retinopathy is an eye condition that can develop
    in people with diabetes. Early identification and regular
    eye examinations are important for protecting vision.

    RetinaCare AI demonstrates how artificial intelligence
    can be used to analyze retinal images and support
    preliminary screening research.
    """)

    st.markdown("## What can you do with this application?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📚 Learn About DR</h3>
            <p>
            Understand diabetic retinopathy, its causes,
            symptoms, stages and screening methods.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🧠 Explore AI Technology</h3>
            <p>
            Learn how EfficientNetB0 and Support Vector
            Machine work together in this project.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 AI Screening</h3>
            <p>
            Upload a retinal fundus image and view the
            model's experimental classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📊 Understand Results</h3>
            <p>
            Learn what the two output classes and
            model-estimated probability mean.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Project Objectives")

    st.markdown("""
    - Demonstrate AI-based retinal image classification.
    - Extract meaningful image features using a CNN.
    - Classify images using a machine learning model.
    - Provide educational information about diabetic retinopathy.
    - Demonstrate the potential of AI-assisted screening.
    """)

    st.warning(
        "This is an academic prototype, not a clinically "
        "validated medical device. Do not use its output "
        "to make medical decisions."
    )


# --------------------------------------------------
# LEARN ABOUT DIABETIC RETINOPATHY
# --------------------------------------------------

elif page == "Learn About DR":

    st.title("Learn About Diabetic Retinopathy")

    st.write(
        "Understand the disease, its causes, symptoms, "
        "stages and screening methods."
    )

    with st.expander("1. What is Diabetic Retinopathy?", expanded=True):

        st.write("""
        Diabetic retinopathy (DR) is an eye condition caused
        by damage to the small blood vessels in the retina
        due to diabetes.

        The retina is the light-sensitive layer at the back
        of the eye. It receives light and sends visual
        information to the brain.

        When blood vessels become damaged, they may leak
        fluid or blood, affecting vision.
        """)

    with st.expander("2. Understanding the Human Retina"):

        st.write("""
        The retina is a thin layer of tissue located at the
        back of the eye.

        Its main function is to receive light and convert
        it into electrical signals that travel to the brain
        through the optic nerve.

        Retinal fundus photographs allow eye specialists
        to examine the retina and identify abnormalities.
        """)

    with st.expander("3. Causes and Risk Factors"):

        st.markdown("""
        Common risk factors include:

        - Long-term diabetes.
        - Poor blood glucose control.
        - High blood pressure.
        - High cholesterol.
        - Pregnancy in people with diabetes.
        - Smoking.
        - Longer duration of diabetes.
        """)

    with st.expander("4. Stages of Diabetic Retinopathy"):

        st.markdown("""
        **Stage 0 – No Apparent DR**

        No visible signs of diabetic retinopathy.

        **Stage 1 – Mild Non-Proliferative DR**

        Small areas of swelling in retinal blood vessels,
        known as microaneurysms, may appear.

        **Stage 2 – Moderate Non-Proliferative DR**

        More blood vessels become affected, and retinal
        abnormalities become more noticeable.

        **Stage 3 – Severe Non-Proliferative DR**

        A larger number of retinal blood vessels are blocked,
        reducing the blood supply to the retina.

        **Stage 4 – Proliferative DR**

        Abnormal new blood vessels grow on the retina.
        These vessels may bleed and cause serious vision
        complications.
        """)

    with st.expander("5. Common Symptoms"):

        st.markdown("""
        Diabetic retinopathy may not cause noticeable
        symptoms during its early stages.

        Possible symptoms include:

        - Blurred vision.
        - Dark spots or floaters.
        - Difficulty seeing at night.
        - Fluctuating vision.
        - Sudden vision loss in advanced cases.

        Regular eye examinations are important even
        when symptoms are absent.
        """)

    with st.expander("6. Diagnosis and Screening"):

        st.markdown("""
        Eye specialists may use:

        - Dilated eye examinations.
        - Retinal fundus photography.
        - Optical Coherence Tomography (OCT).
        - Fluorescein angiography when indicated.

        Retinal fundus images are particularly useful for
        observing blood vessels and retinal abnormalities.
        """)

    with st.expander("7. Prevention and Management"):

        st.markdown("""
        Helpful measures include:

        - Maintaining blood glucose levels as advised
          by a healthcare professional.
        - Controlling blood pressure and cholesterol.
        - Attending regular diabetic eye examinations.
        - Following prescribed treatment.
        - Maintaining a healthy lifestyle.

        Treatment depends on the severity and may include
        injections, laser treatment or surgery.
        """)

    st.info(
        "Consult an eye specialist for medical advice, "
        "screening and treatment."
    )


# --------------------------------------------------
# AI TECHNOLOGY PAGE
# --------------------------------------------------

elif page == "AI Technology":

    st.title("AI Technology Behind RetinaCare AI")

    st.write("""
    RetinaCare AI uses a combination of deep learning
    and traditional machine learning to classify retinal
    fundus images.
    """)

    st.markdown("## 1. Convolutional Neural Network (CNN)")

    st.write("""
    A Convolutional Neural Network is a deep learning
    model designed to process images.

    CNNs learn visual patterns such as edges, shapes,
    textures and more complex structures.

    In this project, the CNN extracts useful features
    from retinal fundus images.
    """)

    st.markdown("## 2. EfficientNetB0")

    st.write("""
    EfficientNetB0 is a pretrained CNN architecture.

    It is used as a feature extractor in this project.
    The model receives a retinal image resized to
    224 × 224 pixels and produces a feature vector
    containing 1280 values.

    These features are passed to the SVM classifier.
    """)

    st.markdown("## 3. Support Vector Machine (SVM)")

    st.write("""
    Support Vector Machine is a supervised machine
    learning algorithm used for classification.

    It learns a decision boundary that separates
    different classes.

    RetinaCare AI uses a linear SVM with feature
    standardization to classify the extracted retinal
    image features.
    """)

    st.markdown("## 4. Project Workflow")

    st.markdown("""
    <div class="info-card">
        <h3>AI Processing Pipeline</h3>
        <p style="text-align:center;">
        <b>Retinal Fundus Image</b>
        <br>⬇<br>
        <b>Image Preprocessing</b>
        <br>Resize to 224 × 224 and convert to RGB
        <br>⬇<br>
        <b>EfficientNetB0</b>
        <br>Extract 1280 image features
        <br>⬇<br>
        <b>StandardScaler</b>
        <br>Standardize extracted features
        <br>⬇<br>
        <b>Linear SVM Classifier</b>
        <br>⬇<br>
        <b>Classification Result</b>
        <br>Non-Referable DR / Referable DR
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 5. Technologies Used")

    technologies = {
        "Technology": [
            "Python",
            "TensorFlow / Keras",
            "EfficientNetB0",
            "Scikit-learn",
            "Streamlit",
            "Pillow",
            "NumPy",
            "Joblib"
        ],
        "Purpose": [
            "Programming language",
            "Deep learning framework",
            "Image feature extraction",
            "Machine learning classification",
            "Web application interface",
            "Image processing",
            "Numerical operations",
            "Model loading"
        ]
    }

    st.table(technologies)


# --------------------------------------------------
# SCREENING PAGE
# --------------------------------------------------

elif page == "Screening":

    st.title("Retinal Image Screening")

    st.write("""
    Upload a retinal fundus image to see the experimental
    classification generated by the RetinaCare AI model.
    """)

    st.warning("""
    Use only appropriate retinal fundus images for this
    academic demonstration. The application has not been
    clinically validated and must not be used for diagnosis.
    """)

    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png"],
        key="retina_upload"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Retinal Image")
            st.image(
                image,
                caption="Uploaded fundus image",
                use_container_width=True
            )

        with col2:

            st.subheader("AI Screening")

            if st.button("Analyze Retinal Image", type="primary"):

                try:

                    with st.spinner("Analyzing retinal image..."):

                        cnn_model, svm_model = load_models()

                        # Preprocessing
                        processed_image = image.resize((224, 224))

                        image_array = np.array(
                            processed_image,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # Feature extraction
                        features = cnn_model.predict(
                            image_array,
                            verbose=0
                        )

                        # SVM classification
                        prediction = svm_model.predict(features)[0]

                        probabilities = svm_model.predict_proba(
                            features
                        )[0]

                        # Find probability belonging to class 1
                        class_indices = list(svm_model.classes_)
                        positive_index = class_indices.index(1)

                        probability = float(
                            probabilities[positive_index]
                        )

                        # Save result
                        st.session_state["prediction"] = int(prediction)
                        st.session_state["probability"] = probability
                        st.session_state["analyzed_image"] = uploaded_file.name

                    st.success("Image analysis completed.")

                except Exception as e:

                    st.error(
                        "Unable to analyze the image. "
                        "Please check the model files and try again."
                    )

                    st.exception(e)

        # Display prediction only for the image analyzed
        if (
            "prediction" in st.session_state
            and st.session_state.get("analyzed_image") == uploaded_file.name
        ):

            st.markdown("---")
            st.subheader("Screening Result")

            prediction = st.session_state["prediction"]
            probability = st.session_state["probability"]

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 0:

                    st.success("Non-Referable DR")

                    st.write("""
                    The model classified this image into
                    the Non-Referable DR category.
                    """)

                else:

                    st.error("Referable DR")

                    st.write("""
                    The model classified this image into
                    the Referable DR category.
                    """)

            with col2:

                st.metric(
                    "Estimated Referable DR Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

            st.info("""
            The displayed probability is an estimate produced
            by the trained model. It is not a clinical risk
            measurement or a confirmed diagnosis.
            """)

            st.warning("""
            A result of Non-Referable DR does not guarantee
            that the eye is healthy. Please follow recommended
            professional eye examination schedules.
            """)


# --------------------------------------------------
# UNDERSTANDING RESULTS PAGE
# --------------------------------------------------

elif page == "Understanding Results":

    st.title("Understanding AI Screening Results")

    st.write("""
    RetinaCare AI produces one of two classification
    outputs based on the model's training labels.
    """)

    st.markdown("## 1. Non-Referable DR")

    st.markdown("""
    <div class="info-card">
        <h3>Non-Referable DR – Class 0</h3>
        <p>
        This means the model classified the uploaded image
        into the non-referable category according to its
        learned patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 2. Referable DR")

    st.markdown("""
    <div class="info-card">
        <h3>Referable DR – Class 1</h3>
        <p>
        This means the model classified the uploaded image
        into the referable category according to its
        learned patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 3. What Does Probability Mean?")

    st.write("""
    The model also provides an estimated probability
    for the Referable DR class.

    For example, an output of 80% means the model's
    probability estimate for its Referable DR class is
    0.80.

    It does not mean that the patient has an 80% medical
    chance of having diabetic retinopathy.
    """)

    st.markdown("## 4. Important Limitations")

    st.markdown("""
    - The model is an academic prototype.
    - It has not been clinically validated.
    - Image quality can affect predictions.
    - The model may produce incorrect classifications.
    - A prediction cannot replace an eye specialist's
      examination.
    """)

    st.error("""
    Never delay medical consultation based on an AI result.
    Seek professional eye care for screening and diagnosis.
    """)


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.title("About RetinaCare AI")

    st.write("""
    RetinaCare AI is an academic project demonstrating
    the use of artificial intelligence for diabetic
    retinopathy image classification.
    """)

    st.markdown("## Project Information")

    st.markdown("""
    <div class="info-card">
        <h3>Project Title</h3>
        <p>RetinaCare AI – Intelligent Retinal Screening</p>

        <h3>Project Domain</h3>
        <p>Artificial Intelligence, Deep Learning and Healthcare</p>

        <h3>Dataset</h3>
        <p>Indian Diabetic Retinopathy Image Dataset (IDRiD)</p>

        <h3>Model Architecture</h3>
        <p>EfficientNetB0 Feature Extractor + Linear SVM Classifier</p>

        <h3>Application Framework</h3>
        <p>Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset and Classification")

    st.write("""
    The project uses retinal fundus images from the
    IDRiD dataset.

    The original diabetic retinopathy grades are grouped
    into two classes for this project:
    """)

    st.markdown("""
    - **Class 0 – Non-Referable DR:** Original grades 0 and 1.
    - **Class 1 – Referable DR:** Original grades 2, 3 and 4.
    """)

    st.markdown("## Project Limitations")

    st.markdown("""
    - The dataset used for this academic project is limited.
    - The model may not generalize to all cameras,
      populations or clinical settings.
    - Its outputs are not validated for clinical use.
    - Additional testing and clinical validation are
      required before any real-world medical application.
    """)

    st.warning("""
    RetinaCare AI is developed for academic and
    educational demonstration only. It is not intended
    to diagnose, treat or prevent any medical condition.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

show_footer()
