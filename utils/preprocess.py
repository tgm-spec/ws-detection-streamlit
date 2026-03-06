import cv2
import numpy as np


def preprocess_image(image):

    # Resize image to model input size
    img = cv2.resize(image, (224, 224))

    # Convert to float32
    img = np.array(img, dtype=np.float32)

    # Normalize pixel values (0–1)
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img