import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.rendering.grid import decode_latent_images
from shap_e.api import generate_3d_model

# Interface utilisateur
st.title("Génération 3D à partir de texte avec Shap-E")
prompt = st.text_input("Entrez un prompt textuel pour générer un modèle 3D")

if st.button("Générer"):
    if prompt:
        with st.spinner("Génération du modèle 3D en cours..."):
            # Générer un modèle 3D à partir du texte
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            latents = sample_latents(prompt, device=device)
            images = decode_latent_images(latents, device=device)
            
            # Affichage des résultats
            st.image(images[0])
            st.success("Modèle 3D généré avec succès !")
    else:
        st.error("Veuillez entrer un prompt textuel.")
