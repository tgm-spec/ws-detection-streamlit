import streamlit as st
from PIL import Image
import numpy as np
import cv2

from utils.facecheck import check_face
from utils.preprocess import preprocess_image
from utils.model_loader import load_model
from utils.gradcam import generate_gradcam, overlay_gradcam


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Williams Syndrome Detection",
    page_icon="🧬",
    layout="centered"
)

# -----------------------------
# MODEL LOADING (cached)
# -----------------------------
@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Page",
    ["🧠 Detection Tool", "📖 Patient Guide", "ℹ️ About System"]
)

st.sidebar.subheader("Quick Instructions")

st.sidebar.info(
"""
1. Upload a clear frontal facial image
2. Only one face should be visible
3. Click Analyze Image

Recommended:
• Good lighting
• No sunglasses
• No mask
"""
)

st.sidebar.warning(
"⚠ This application is a screening tool and not a medical diagnosis system."
)

# =====================================================
# DETECTION TOOL PAGE
# =====================================================
if menu == "🧠 Detection Tool":

    st.title("🧬 Facial Image-Based Detection of Williams Syndrome")

    st.markdown(
    """
    This system analyzes facial features using Deep Learning (MobileNetV2)
    to identify patterns associated with Williams Syndrome.
    """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        st.divider()

        result = check_face(image_np)

        face_detected = result["face_detected"]
        face_img = result["face_crop"]
        bbox = result["face_bbox"]
        blur_level = result["blur_level"]
        blur_score = result["blur_score"]
        multiple_faces = result["multiple_faces"]

        st.subheader("Face Detection Preview")

        if face_detected and bbox is not None:

            x = bbox["x"]
            y = bbox["y"]
            w = bbox["w"]
            h = bbox["h"]

            preview_img = image_np.copy()

            cv2.rectangle(
                preview_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            st.success("✅ Face detected successfully")

            st.image(
                preview_img,
                caption="Detected Face Bounding Box",
                use_container_width=True
            )

            st.subheader("Detected Face Region")
            st.image(face_img, width=250)

            st.write(f"Blur Level: **{blur_level}**")
            st.write(f"Blur Score: **{blur_score}**")

            if multiple_faces:
                st.warning("Multiple faces detected. Using the largest face.")

        else:

            st.error("❌ No face detected in the uploaded image.")
            st.write("Please upload a clear image containing a single human face.")
            st.stop()

        st.divider()

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing facial features..."):

                processed_img = preprocess_image(face_img)

                prediction = model.predict(processed_img, verbose=0)[0][0]

                ws_probability = float(prediction)
                normal_probability = float(1 - prediction)

            st.success("Analysis completed")

            st.divider()

            st.subheader("Prediction Result")

            confidence = max(ws_probability, normal_probability)

            col1, col2 = st.columns(2)

            with col1:
                if ws_probability > normal_probability:
                    st.error("⚠ Williams Syndrome Indicators Detected")
                else:
                    st.success("✅ No Significant Indicators Detected")

            with col2:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence*100:.2f}%"
                )

            st.divider()

            st.subheader("Prediction Probabilities")

            col1, col2 = st.columns(2)

            col1.metric(
                "Williams Syndrome Probability",
                f"{ws_probability*100:.2f}%"
            )

            col2.metric(
                "Normal Probability",
                f"{normal_probability*100:.2f}%"
            )

            st.divider()

            st.subheader("Explainable AI (Grad-CAM)")

            heatmap = generate_gradcam(model, processed_img)

            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            overlay = overlay_gradcam(face_bgr, heatmap)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            st.image(
                overlay,
                caption="Highlighted regions show where the AI focused",
                use_container_width=True
            )


# =====================================================
# PATIENT GUIDE PAGE
# =====================================================
elif menu == "📖 Patient Guide":

    st.title("Patient Information Guide")

    st.subheader("What is Williams Syndrome?")

    st.info(
"""
Williams Syndrome is a rare genetic condition caused by a missing
piece of DNA on chromosome 7.

It affects physical development, learning ability, and sometimes
heart and blood vessel health.
"""
    )

    st.subheader("Common Symptoms")

    st.markdown(
"""
• Wide mouth with full lips
• Small chin
• Puffy eyes
• Learning or developmental delays
• Heart or blood vessel problems
• Highly social personality
"""
    )

    st.subheader("Is it Hereditary?")

    st.markdown(
"""
Most cases occur randomly during early development.

However, a person with Williams Syndrome can pass the condition
to their children.
"""
    )

    st.subheader("What Should You Do If Concerned?")

    st.warning(
"""
Do not panic.

This tool only provides early screening information.

If you are concerned, please consult a qualified medical professional.
"""
    )

    st.subheader("How Doctors Confirm the Condition")

    st.markdown(
"""
Doctors usually confirm Williams Syndrome using:

• Physical examination
• Heart evaluation (echocardiogram)
• Genetic testing (DNA analysis)
"""
    )


# =====================================================
# ABOUT SYSTEM PAGE
# =====================================================
else:

    st.title("ℹ️ About System")

    st.markdown(
"""
This application uses Deep Learning and Computer Vision
to analyze facial structures related to Williams Syndrome.

Model architecture:
• MobileNetV2 transfer learning
• Facial feature analysis
• Explainable AI using Grad-CAM
"""
    )

    st.error(
"""
⚠ DISCLAIMER

This application is intended for research and educational purposes only.

It does NOT provide medical diagnosis.

Always consult a licensed healthcare professional
for proper medical evaluation.
"""
    )