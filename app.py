import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as opt
import os
import requests
import datetime
import joblib
import time

# --- Pleine page ---
st.set_page_config(layout="wide")

# --- Choix de l’échelon ---
st.sidebar.header("⚙️ Paramètres")
echelon = st.sidebar.selectbox("Sélectionnez l’échelon :", ["K", "L"])

# --- Définir le modèle et les features en fonction de l’échelon ---
MODEL_URLS = {
    "K": "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_K.pkl",
    "L": "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_L.pkl",
}

FEATURES = {
    "K": {  # ✅ 3xx
        'TIC323': 74.825,
        'PI326': -0.03,
        'PI328': 1.865,
        'TI344': 103.125,
        'PI346': 133.85,
        'ACP29% entré Echelons': 1260.0,
        'Heure_float': 7.00
    },
    "L": {  # ✅ 4xx
        'TIC423': 74.825,
        'PI426': -0.03,
        'PI428': 1.865,
        'TI444': 103.125,
        'PI446': 133.85,
        'ACP29% entré Echelons': 1260.0,
        'Heure_float': 7.00
    }
}

# --- Téléchargement et chargement du modèle ---
def load_model(url, local_filename, echelon):
    if not os.path.exists(local_filename):
        with st.spinner("🔄 Téléchargement du modèle depuis Hugging Face..."):
            response = requests.get(url)
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(response.content)

    obj = joblib.load(local_filename)

    if isinstance(obj, dict):  # cas K (bundle dict)
        best_model = obj["model"]
        expected_cols = obj["feature_names"]
    else:  # cas L (pipeline direct)
        best_model = obj
        try:
            expected_cols = list(best_model.feature_names_in_)
        except AttributeError:
            if echelon == "L":
                expected_cols = ['TIC423', 'PI426', 'PI428', 'TI444', 'PI446', 'ACP29%', 'Heure_float']
            else:
                expected_cols = ['TIC323', 'PI326', 'PI328', 'TI344', 'PI346', 'ACP29%', 'Heure_float']

    return best_model, expected_cols

model_url = MODEL_URLS[echelon]
local_path = os.path.basename(model_url)
best_model, expected_cols = load_model(model_url, local_path, echelon)

# --- Harmonisation ACP29 ---
def harmonize_acp29_column(input_df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    if "ACP29%" in expected_cols and "ACP29%" not in input_df.columns:
        if "ACP29% entré Echelons" in input_df.columns:
            input_df["ACP29%"] = input_df["ACP29% entré Echelons"]
    if "ACP29% entré Echelons" in expected_cols and "ACP29% entré Echelons" not in input_df.columns:
        if "ACP29%" in input_df.columns:
            input_df["ACP29% entré Echelons"] = input_df["ACP29%"]
    return input_df

# --- Features et valeurs par défaut selon l’échelon ---
DEFAULT_INPUTS = FEATURES[echelon]
DISPLAY_FEATURES = list(DEFAULT_INPUTS.keys())
OPTIMIZABLE = [f for f in DISPLAY_FEATURES if f not in ['ACP29% entré Echelons', 'ACP29%', 'Heure_float']]

# --- Optimisation avec bornes fixes ---
def optimize_selected(input_vals, target, model, selected_vars):
    # Bornes spécifiques pour K et L
    custom_bounds = {
        # Pour K
        'TIC323': (70, 90),
        'PI326': (-1, 2),
        'PI328': (0.5, 3),
        'TI344': (60, 120),
        'PI346': (50, 200),
        # Pour L
        'TIC423': (70, 90),
        'PI426': (-1, 2),
        'PI428': (0.5, 3),
        'TI444': (60, 120),
        'PI446': (50, 200),
    }

    bounds = []
    for var in selected_vars:
        if var in custom_bounds:
            bounds.append(custom_bounds[var])
        else:
            v = input_vals[var]
            bounds.append((v * 0.7, v * 1.3) if v != 0 else (-1, 1))

    def obj(x):
        tmp = input_vals.copy()
        for i, var in enumerate(selected_vars):
            tmp[var] = x[i]
        df_tmp = pd.DataFrame([tmp])[expected_cols]
        pred = model.predict(df_tmp)[0]
        return abs(pred - target)

    res = opt.differential_evolution(obj, bounds, maxiter=50, polish=True, workers=1)
    return pd.Series(res.x, index=selected_vars), res.fun

# --- En-tête ---
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 1em;">
    <img src="https://raw.githubusercontent.com/erochd/acp54-app/main/kofert.jpeg" style="width:90px; border-radius: 4px;" />
    <div>
        <h1 style="margin-bottom: 0;">Smart Assistant for Acid Concentration Control</h1>
        <h4 style="margin-top: 0; color: grey;">🎯 Target : <em>ACP54% sortie Echelon {echelon}</em></h4>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Formulaire de prédiction ---
with st.form("form_pred"):
    st.subheader("1. Saisie des variables")
    user_display = {}
    cols = st.columns(3)

    for i, feat in enumerate(DISPLAY_FEATURES):
        if feat != "Heure_float":
            default = DEFAULT_INPUTS[feat]
            with cols[i % 3]:
                user_display[feat] = st.number_input(label=feat, value=default, key=feat)

    # Heure
    with cols[len(DISPLAY_FEATURES) % 3]:
        heure_saisie = st.time_input("Heure de la mesure", value=datetime.time(7, 0))
    user_display['Heure_float'] = heure_saisie.hour + heure_saisie.minute / 60

    submit_pred = st.form_submit_button("Prédire")

if submit_pred:
    input_df = pd.DataFrame([user_display])

    # ✅ Harmonisation des noms ACP29
    input_df = harmonize_acp29_column(input_df, expected_cols)

    missing_cols = set(expected_cols) - set(input_df.columns)
    if missing_cols:
        st.error(f"⛔ Erreur : colonnes manquantes dans l'entrée : {missing_cols}")
    else:
        input_df = input_df[expected_cols]
        pred = best_model.predict(input_df)[0]
        st.success(f"Prédiction ACP54% sortie Echelon {echelon} : **{pred:.2f}**")
        st.session_state.input_df = input_df
        st.session_state.pred = pred

# --- Optimisation ---
if 'pred' in st.session_state:

    with st.form("form_opt"):
        st.subheader("2. Optimisation des paramètres")
        target = st.number_input("Valeur cible", value=st.session_state.pred)

        opt_selected_display = st.multiselect(
            "Variables à optimiser",
            OPTIMIZABLE
        )
        submit_opt = st.form_submit_button("Optimiser les variables")

    if submit_opt:
        if not opt_selected_display:
            st.error("⛔ Merci de sélectionner au moins une variable à optimiser.")
        else:
            with st.spinner("🔄 Optimisation en cours... cela peut prendre quelques secondes"):
                progress_bar = st.progress(0)
                for percent_complete in range(0, 100, 10):
                    time.sleep(0.05)
                    progress_bar.progress(percent_complete + 10)

                base = st.session_state.input_df.iloc[0]
                opt_vals, err = optimize_selected(base, target, best_model, opt_selected_display)
                progress_bar.empty()

            df_out = pd.DataFrame({
                'Variable': opt_selected_display,
                'Valeur actuelle': base[opt_selected_display].values,
                'Ajustement brut': opt_vals.values
            })

            def with_arrow(row):
                delta = row['Ajustement brut'] - row['Valeur actuelle']
                icon = "🔼" if delta > 0 else "🔽" if delta < 0 else "⏺️"
                return f"{icon} {row['Ajustement brut']:.2f}"

            df_out['Ajustement proposé'] = df_out.apply(with_arrow, axis=1)
            df_out.drop(columns=['Ajustement brut'], inplace=True)

            st.subheader("Ajustements proposés")
            st.table(df_out)
            st.info(f"Ecart final |prédiction–cible| : {err:.2f}")
