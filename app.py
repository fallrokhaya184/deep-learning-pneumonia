import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Détection de pneumonie",
    page_icon="🫁",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modele_pneumonia_final.keras")

st.title("🫁 Détection de pneumonie")
st.markdown(
    "Classification de radiographies pulmonaires "
    "à l'aide d'un modèle de Deep Learning (CNN)."
)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    st.success("Modèle chargé avec succès !")
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Importer une radiographie pulmonaire",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie importée", width="stretch")

    if st.button("Analyser la radiographie", type="primary"):
        with st.spinner("Analyse en cours..."):
            resized = image.resize((150, 150))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = np.expand_dims(array, axis=0)

            prediction = float(model.predict(array, verbose=0)[0][0])

            probability_pneumonia = prediction * 100
            probability_normal = (1 - prediction) * 100

        st.subheader("Résultat de la classification")

        if prediction >= 0.5:
            st.error("Classe prédite : PNEUMONIA")
        else:
            st.success("Classe prédite : NORMAL")

        st.metric("Probabilité NORMAL", f"{probability_normal:.2f} %")
        st.metric("Probabilité PNEUMONIA", f"{probability_pneumonia:.2f} %")

        st.progress(prediction)

st.divider()

st.warning(
    "Prototype pédagogique uniquement. "
    "Ce modèle ne constitue pas un outil de diagnostic médical. "
    "Les résultats doivent être interprétés par un professionnel de santé."
)
