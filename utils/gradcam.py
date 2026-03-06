import numpy as np
import tensorflow as tf
import cv2


def generate_gradcam(model, img_array):
    """
    Generate Grad-CAM heatmap for MobileNetV2 transfer learning model
    """

    # Get MobileNetV2 backbone
    backbone = model.get_layer("mobilenetv2_1.00_224")

    # Last convolution layer
    last_conv_layer = backbone.get_layer("Conv_1")

    # Backbone model (input → last conv output)
    backbone_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output
    )

    # Rebuild classifier head
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input
    x = model.layers[2](x)  # GlobalAveragePooling
    x = model.layers[3](x)  # BatchNorm
    x = model.layers[4](x)  # Dense
    x = model.layers[5](x)  # Dropout
    classifier_output = model.layers[6](x)

    classifier_model = tf.keras.Model(classifier_input, classifier_output)

    # Compute gradients
    with tf.GradientTape() as tape:

        conv_outputs = backbone_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = heatmap.numpy()

    # ReLU
    heatmap = np.maximum(heatmap, 0)

    # Normalize safely
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

    # Convert to colored heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def overlay_gradcam(original_img, heatmap, alpha=0.40):
    """
    Overlay Grad-CAM heatmap on original image
    """

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    overlay = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay