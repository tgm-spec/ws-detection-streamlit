import tensorflow as tf
import streamlit as st


# Cache the model so Streamlit doesn't reload it every time
@st.cache_resource
def load_model():

    model_path = "models/wsV3.keras"

    model = tf.keras.models.load_model(model_path)

    return model