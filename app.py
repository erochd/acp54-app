import streamlit as st
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt
import os
import urllib.request  # ✅ ajouté pour le téléchargement depuis Google Drive
import joblib     
import requests
import gdown

# --- Chargement du modèle
@st.cache_resource
def load_model():
    file_id = "12BX-CPQoHH6GFhI6XlaDEpwe9bvjjc-Z"  # ton vrai ID Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "best_model.pkl"

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        model = pickle.load(f)
    return model

best_model = load_model()

# --- Variables attendues
# Toutes les variables saisies + ACP29% (non optimisable)
# --- Valeurs par défaut pour tests
DEFAULT_INPUTS = {'604JFIC214.PV - weak phos acid condenst (M3/H)': 51.849999999999994, '604JTI211.PV - WEAK PHOS ACD CONDST CO0 (DEGC)': 51.81500000000001, '604JTI208.PV (°C)': 57.1, '604AFI063.PV - 5B STM FRM DISTRIBUTION (T/H)': 24.625000000000007, '604APIC066.PV - 5B STM TO DESUPERHEATER (BAR)': 1.975, '604ATI064.PV - 5B STM TO SEPARATOR (DEGC)': 156.07, '604ATI068A.PV - 2BARG STM TO EVAP J (°C)': 130.11, '604JFI221.PV - LP STEAM TO EVA AE02 (T/H)': 19.385000000000005, '604JTIC223.PV - LP STM FRM SEPARTOR A (DEGC)': 74.82499999999999, '604JPI225.PV (BAR)': -0.49, '604JPI226.PV - EVAP HEATR AE02 (BAR)': -0.03, '604JPI228.PV (BAR)': 1.8650000000000002, '604JLI241.PV - FLSH CHAMBR AD01 (%)': 103.12500000000001, '604JPIC242.PV - FLASH CHAMBR AD01 (mm/hg)': 131.39999999999998, '604JPI246.PV - VAPOR TO FSA (mm/hg)': 133.85, '604JFIC250.PV - EVA PMP AP05 PHOS.ACID (M3/H)': 10.749999999999996, 'ACP29% entré Echelons': 1260.0}

ALL_FEATURES = [
'604JFIC214.PV - weak phos acid condenst (M3/H)',
       '604JTI211.PV - WEAK PHOS ACD CONDST CO0 (DEGC)', '604JTI208.PV (°C)',
       '604AFI063.PV - 5B STM FRM DISTRIBUTION (T/H)',
       '604APIC066.PV - 5B STM TO DESUPERHEATER (BAR)',
       '604ATI064.PV - 5B STM TO SEPARATOR (DEGC)',
       '604ATI068A.PV - 2BARG STM TO EVAP J (°C)',
       '604JFI221.PV - LP STEAM TO EVA AE02 (T/H)',
       '604JTIC223.PV - LP STM FRM SEPARTOR A (DEGC)', '604JPI225.PV (BAR)',
       '604JPI226.PV - EVAP HEATR AE02 (BAR)', '604JPI228.PV (BAR)',
       '604JLI241.PV - FLSH CHAMBR AD01 (%)',
       '604JPIC242.PV - FLASH CHAMBR AD01 (mm/hg)',
       '604JPI246.PV - VAPOR TO FSA (mm/hg)',
       '604JFIC250.PV - EVA PMP AP05 PHOS.ACID (M3/H)',
       'ACP29% entré Echelons'
]
# Variables potentiellement optimisables (excluant la variable fixe ACP29%)
OPTIMIZABLE = [f for f in ALL_FEATURES if f != 'ACP29% entré Echelons']

# --- Fonction d'optimisation restreinte
def optimize_selected(input_vals, target, model, selected_vars, var_range=0.3):
    """
    Optimise seulement les variables sélectionnées pour minimiser |pred - target|.
    """
    # Préparer bounds pour chaque variable sélectionnée
    bounds = []
    for var in selected_vars:
        v = input_vals[var]
        low = v * (1 - var_range) if v != 0 else -1
        high = v * (1 + var_range) if v != 0 else 1
        bounds.append((low, high))
    
    # Fonction objectif sur l'espace réduit
    def obj(x):
        tmp = input_vals.copy()
        tmp[selected_vars] = x
        df_tmp = pd.DataFrame([tmp.values], columns=tmp.index)
        pred = model.predict(df_tmp)[0]
        return abs(pred - target)
    
    res = opt.differential_evolution(obj, bounds, maxiter=50, polish=True)
    opt_vals = pd.Series(res.x, index=selected_vars)
    return opt_vals, res.fun

# --- Interface Streamlit
st.title("Prédiction et Optimisation – ACP54% sortie Echelons")
st.markdown("1) Saisissez toutes les variables ci‑dessous.\n"
            "2) Cliquez sur **Prédire**.\n"
            "3) Saisissez une cible.\n"
            "4) Choisissez les variables à optimiser.\n"
            "5) Cliquez sur **Optimiser** pour voir les ajustements proposés.")

# --- Formulaire de prédiction
with st.form("form_pred"):
    st.subheader("1. Saisie des variables")
    user = {}
    for feat in ALL_FEATURES:
        default = DEFAULT_INPUTS.get(feat, 0.0)
        user[feat] = st.number_input(f"{feat}", value=default, key=feat, format="%.3f")
    submit_pred = st.form_submit_button("Prédire")

if submit_pred:
    # Stocker la saisie et prédire
    input_df = pd.DataFrame([user], columns=ALL_FEATURES)
    pred = best_model.predict(input_df)[0]
    st.success(f"Prédiction ACP54% sortie Echelons : **{pred:.2f}**")
    st.session_state.input_df = input_df
    st.session_state.pred = pred

# --- Formulaire d'optimisation
if 'pred' in st.session_state:
    with st.form("form_opt"):
        st.subheader("2. Optimisation locale")
        target = st.number_input("Valeur cible", value=st.session_state.pred)
        to_opt = st.multiselect("Variables à optimiser", OPTIMIZABLE)
        submit_opt = st.form_submit_button("Optimiser les variables")
    
    if submit_opt:
        base = st.session_state.input_df.iloc[0]
        opt_vals, err = optimize_selected(base, target, best_model, to_opt)
        
        # Préparer le DataFrame des suggestions
        df_out = pd.DataFrame({
            'Variable': to_opt,
            'Valeur actuelle': base[to_opt].values,
            'Ajustement brut': opt_vals.values
        })
        
        # Ajouter une colonne avec flèche et valeur formatée
        def with_arrow(row):
            delta = row['Ajustement brut'] - row['Valeur actuelle']
            icon = "🔼" if delta > 0 else "🔽" if delta < 0 else "⏺️"
            return f"{icon} {row['Ajustement brut']:.2f}"
        
        df_out['Ajustement proposé'] = df_out.apply(with_arrow, axis=1)
        df_out.drop(columns=['Ajustement brut'], inplace=True)
        
        # Affichage
        st.subheader("Ajustements proposés")
        st.table(df_out)
        st.info(f"Erreur finale |prédiction–cible| : {err:.2f}")
