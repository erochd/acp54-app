import streamlit as st
import joblib, requests, os

# --- Page ---
st.set_page_config(layout="wide")
st.title("Test ACP54 App")

# --- Sélection de l’échelon ---
st.sidebar.header("⚙️ Paramètres")
echelon = st.sidebar.selectbox("Sélectionnez l’échelon :", ["J", "K", "L"])

# --- URLs modèles ---
MODEL_URLS = {
    "J": "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_light_vf.pkl",
    "K": "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_K_v2.pkl",
    "L": "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_L.pkl",
}

# --- Fonction de chargement ---
def load_model(url, local_filename):
    if not os.path.exists(local_filename):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(r.content)
    return joblib.load(local_filename)

# --- Charger modèle choisi ---
model_url = MODEL_URLS[echelon]
local_path = os.path.basename(model_url)
model = load_model(model_url, local_path)

st.success(f"✅ Modèle {echelon} chargé avec succès")

# --- Vérifier colonnes attendues ---
if hasattr(model, "feature_names_in_"):
    st.write("Colonnes attendues :", list(model.feature_names_in_))
else:
    st.warning("⚠️ Impossible de récupérer les features attendues")
