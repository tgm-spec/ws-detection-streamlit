"""
facecheck.py
-----------------
Face validation module for WS Detection App

Design philosophy:
- Face NOT detected  → reject
- Face detected      → allow prediction
- Blur is used ONLY as a warning signal

Reason:
Williams Syndrome datasets often contain low-resolution images,
so blur should not block prediction.
"""

import cv2
import numpy as np


# -----------------------------
# Load face detector once
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -----------------------------
# Blur estimation
# -----------------------------
def estimate_blur(gray_img):
    """
    Estimate image sharpness using Laplacian variance.
    Lower value → more blur
    """
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


# -----------------------------
# Select primary face
# -----------------------------
def select_primary_face(faces):
    """
    Select largest detected face
    """
    if len(faces) == 1:
        return faces[0]

    return max(faces, key=lambda f: f[2] * f[3])


# -----------------------------
# Main face validation function
# -----------------------------
def check_face(image):
    """
    Parameters
    ----------
    image : numpy array
        Input RGB image

    Returns
    -------
    dict containing:
        face_detected
        face_crop
        blur_level
        blur_score
        face_bbox
        multiple_faces
        reason
    """

    img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # ❌ No face detected
    if len(faces) == 0:
        return {
            "face_detected": False,
            "face_crop": None,
            "blur_level": None,
            "blur_score": None,
            "face_bbox": None,
            "multiple_faces": False,
            "reason": "no_face_detected"
        }

    # Select main face
    x, y, w, h = select_primary_face(faces)

    face_crop = img[y:y+h, x:x+w]

    blur_score = estimate_blur(gray)

    # Blur classification
    if blur_score < 20:
        blur_level = "very_blurry"
    elif blur_score < 50:
        blur_level = "blurry"
    else:
        blur_level = "clear"

    return {
        "face_detected": True,
        "face_crop": face_crop,
        "blur_level": blur_level,
        "blur_score": round(float(blur_score), 2),
        "face_bbox": {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        },
        "multiple_faces": len(faces) > 1,
        "reason": "ok"
    }