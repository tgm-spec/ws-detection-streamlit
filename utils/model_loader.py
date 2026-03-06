import tensorflow as tf
import streamlit as st


@st.cache_resource
def load_model():

    with st.spinner("Loading AI model..."):

        model_path = "models/wsV3.keras"

        model = tf.keras.models.load_model(model_path)

    return model