import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from utils.facecheck import check_face
from utils.preprocess import preprocess_image
from utils.model_loader import load_model
from utils.gradcam import generate_gradcam, overlay_gradcam


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Williams Syndrome Detection",
    page_icon="🧬",
    layout="centered"
)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

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
3. Click Analyze  

Recommended:
• Good lighting  
• No sunglasses  
• No mask
"""
)

st.sidebar.warning(
"⚠ This system is a research screening tool and not a medical diagnosis system."
)

# =================================================
# DETECTION TOOL
# =================================================

if menu == "🧠 Detection Tool":

    st.title("🧬 Facial Image-Based Detection of Williams Syndrome")

    st.markdown(
"""
This AI system analyzes facial structures using **MobileNetV2 Deep Learning**
to identify facial patterns associated with **Williams Syndrome**.
"""
    )

    st.divider()

    # -------------------------------------------------
    # UPLOAD + ANALYZE SIDE BY SIDE
    # -------------------------------------------------

    col_upload, col_analyze = st.columns([3,1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Facial Image",
            type=["jpg","jpeg","png","webp"]
        )

    with col_analyze:
        st.markdown("<div style='margin-top:42px'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze")

    # -------------------------------------------------
    # IF IMAGE UPLOADED
    # -------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        st.divider()

        # -------------------------------------------------
        # FACE CHECK
        # -------------------------------------------------

        result = check_face(image_np)

        face_detected = result["face_detected"]
        face_img = result["face_crop"]
        bbox = result["face_bbox"]
        blur_level = result["blur_level"]
        blur_score = result["blur_score"]
        multiple_faces = result["multiple_faces"]

        st.subheader("Face Detection")

        if face_detected:

            x = bbox["x"]
            y = bbox["y"]
            w = bbox["w"]
            h = bbox["h"]

            preview = image_np.copy()

            cv2.rectangle(
                preview,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

            st.image(
                preview,
                caption="Detected Face Bounding Box",
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(face_img, caption="Detected Face")

            with col2:

                st.subheader("Image Quality")

                st.write(f"Blur Level: **{blur_level}**")
                st.write(f"Blur Score: **{blur_score:.2f}**")

                if blur_level == "clear":
                    st.success("Image quality suitable for analysis")
                else:
                    st.warning("Blurry image may reduce prediction accuracy")

            if multiple_faces:
                st.warning("Multiple faces detected. Largest face used.")

        else:

            st.error("No face detected. Please upload a clear frontal face.")
            st.stop()

        st.divider()

        # -------------------------------------------------
        # RUN ANALYSIS
        # -------------------------------------------------

        if analyze_clicked:

            status = st.empty()

            status.info("Loading AI model...")
            time.sleep(0.5)

            status.info("Detecting facial structure...")
            time.sleep(0.5)

            status.info("Extracting facial features...")
            time.sleep(0.5)

            status.info("Running neural network prediction...")
            time.sleep(0.5)

            with st.spinner("Finalizing analysis..."):

                processed_img = preprocess_image(face_img)

                prediction = model.predict(
                    processed_img,
                    verbose=0
                )[0][0]

                ws_probability = float(prediction)
                normal_probability = float(1 - prediction)

            status.success("Analysis complete")

            confidence = max(ws_probability, normal_probability)

            st.divider()

            # -------------------------------------------------
            # RESULT SUMMARY PANEL
            # -------------------------------------------------

            st.subheader("AI Screening Summary")

            col1, col2, col3 = st.columns(3)

            with col1:

                if ws_probability > normal_probability:
                    st.error("⚠ Williams Syndrome Indicators")
                else:
                    st.success("✅ No Significant Indicators")

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

            with col3:

                if confidence >= 0.85:
                    st.success("Reliability: High")
                elif confidence >= 0.70:
                    st.warning("Reliability: Moderate")
                else:
                    st.error("Reliability: Low")

            st.progress(confidence)

            st.divider()

            # -------------------------------------------------
            # PROBABILITIES
            # -------------------------------------------------

            st.subheader("Prediction Probabilities")

            col1, col2 = st.columns(2)

            col1.metric(
                "Williams Syndrome",
                f"{ws_probability*100:.2f}%"
            )

            col2.metric(
                "Normal",
                f"{normal_probability*100:.2f}%"
            )

            st.divider()

            # -------------------------------------------------
            # GRADCAM
            # -------------------------------------------------

            st.subheader("Explainable AI (Grad-CAM)")

            heatmap = generate_gradcam(model, processed_img)

            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            overlay = overlay_gradcam(face_bgr, heatmap)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.image(face_img, caption="Original Face")

            with col2:
                st.image(overlay, caption="AI Attention Map")

            st.divider()

            # -------------------------------------------------
            # AI EXPLANATION
            # -------------------------------------------------

            st.subheader("AI Explanation")

            if ws_probability > normal_probability:

                st.info(
"""
The model detected facial patterns associated with **Williams Syndrome**.

Highlighted regions show where the neural network focused,
often including the **eyes, mouth, and mid-face region**.
"""
                )

            else:

                st.info(
"""
The model did **not detect strong facial patterns** associated
with Williams Syndrome.

Highlighted regions show where the AI examined facial features
such as the **eyes, nose, and mouth**.
"""
                )

            st.caption(
"Grad-CAM highlights model attention and should not be interpreted as medical evidence."
            )

            st.divider()

            # -------------------------------------------------
            # PDF REPORT GENERATION
            # -------------------------------------------------

            report_buffer = io.BytesIO()
            styles = getSampleStyleSheet()

            story = []

            story.append(Paragraph("AI Facial Screening Report", styles['Title']))
            story.append(Spacer(1,20))

            result_text = (
                "Williams Syndrome Indicators Detected"
                if ws_probability > normal_probability
                else
                "No Significant Indicators Detected"
            )

            story.append(Paragraph(f"<b>Prediction Result:</b> {result_text}", styles['Normal']))
            story.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.2f}%", styles['Normal']))
            story.append(Paragraph(f"<b>WS Probability:</b> {ws_probability*100:.2f}%", styles['Normal']))
            story.append(Paragraph(f"<b>Normal Probability:</b> {normal_probability*100:.2f}%", styles['Normal']))
            story.append(Paragraph(f"<b>Image Quality:</b> {blur_level}", styles['Normal']))

            story.append(Spacer(1,20))

            story.append(Paragraph(
                "Disclaimer: This AI system is intended for research and educational purposes only "
                "and does not provide medical diagnosis.",
                styles['Normal']
            ))

            pdf = SimpleDocTemplate(report_buffer)
            pdf.build(story)

            st.download_button(
                "📄 Download AI Analysis Report (PDF)",
                data=report_buffer.getvalue(),
                file_name="ws_ai_report.pdf",
                mime="application/pdf"
            )

# =================================================
# PATIENT GUIDE
# =================================================

elif menu == "📖 Patient Guide":

    st.title("Patient Information Guide")

    st.info(
"""
Williams Syndrome is a rare genetic condition caused by a deletion
of genetic material on chromosome 7.

It affects facial development, learning ability,
and cardiovascular health.
"""
    )

    st.subheader("Common Features")

    st.markdown(
"""
• Wide mouth with full lips  
• Small chin  
• Puffy eyes  
• Learning difficulties  
• Heart or blood vessel problems
"""
    )

    st.warning(
"""
This tool provides **AI-based screening only**.

If you suspect Williams Syndrome,
please consult a qualified healthcare professional.
"""
    )


# =================================================
# ABOUT SYSTEM
# =================================================

else:

    st.title("About This System")

    st.markdown(
"""
This application uses **Deep Learning and Computer Vision**
to analyze facial structures related to Williams Syndrome.

Model Architecture:

• MobileNetV2 Transfer Learning  
• Binary Classification (WS / Normal)  
• Grad-CAM Explainable AI
"""
    )

    st.error(
"""
⚠ DISCLAIMER

This system is intended for **research and educational purposes only**.

It does NOT provide medical diagnosis.
Always consult a licensed healthcare professional
for proper medical evaluation.
"""
    )