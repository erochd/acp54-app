import streamlit as st
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt
import os
import requests
import datetime
import joblib

# --- Pleine page ---
st.set_page_config(layout="wide")

# --- Chargement du modèle depuis Hugging Face ---
@st.cache_resource
def load_model():
    url = "https://huggingface.co/erochd/acp54-app/resolve/main/best_modele_acide_light_vf.pkl"
    local_path = "best_modele_acide_light_vf.pkl"

    # Télécharger une seule fois le fichier si pas encore en local
    if not os.path.exists(local_path):
        with st.spinner("🔄 Téléchargement du modèle depuis Hugging Face..."):
            response = requests.get(url)
            response.raise_for_status()  # Lève une erreur HTTP si le lien est invalide
            with open(local_path, "wb") as f:
                f.write(response.content)

    # Charger le modèle avec gestion d’erreur
    try:
        with open(local_path, "rb") as f:
            model = joblib.load(f)
    except Exception as e:
        st.error(f"❌ Échec du chargement du modèle : {e}")
        st.stop()

    return model

# --- Instanciation du modèle (hors fonction) ---
best_model = load_model()


# --- Mapping affichage utilisateur → colonnes du modèle
display_to_model_units = {
    'TIC223': ('TIC223', '°C'),
    'PI226': ('PI226', 'bar'),
    'PI228': ('PI228', 'bar'),
    'TI244': ('TI244', '°C'),
    'PI246': ('PI246', 'mm/hg'),
    'FIC250': ('FIC250', 'm³/h'),
    'ACP29% entrée Echelons': ('ACP29% entré Echelons', ''),
    'Heure_float': ('Heure_float', '')
}

# Valeurs par défaut pour interface
DEFAULT_INPUTS = {
    'TIC223': 74.825,
    'PI226': -0.03,
    'PI228': 1.865,
    'TI244': 103.125,
    'PI246': 133.85,
    'FIC250': 10.75,
    'ACP29% entrée Echelons': 1260.0,
    'Heure_float': 7.00
}

DISPLAY_FEATURES = list(DEFAULT_INPUTS.keys())
OPTIMIZABLE = [f for f in DISPLAY_FEATURES if f not in ['ACP29% entrée Echelons', 'Heure_float']]

# --- Optimisation

def optimize_selected(input_vals, target, model, selected_vars, var_range=0.3):
    bounds = [(v * (1 - var_range), v * (1 + var_range)) if v != 0 else (-1, 1)
              for v in input_vals[selected_vars]]

    def obj(x):
        tmp = input_vals.copy()
        tmp[selected_vars] = x
        pred = model.predict(pd.DataFrame([tmp]))[0]
        return abs(pred - target)

    res = opt.differential_evolution(obj, bounds, maxiter=50, polish=True)
    return pd.Series(res.x, index=selected_vars), res.fun

# --- En-tête
st.markdown("""
<div style="display: flex; align-items: center; gap: 1em;">
    <img src="https://raw.githubusercontent.com/erochd/acp54-app/main/kofert.jpeg" style="width:90px; border-radius: 4px;" />
    <div>
        <h1 style="margin-bottom: 0;">Smart Assistant for Acid Concentration Control</h1>
        <h4 style="margin-top: 0; color: grey;">🎯 Target : <em>ACP54% sortie Echelons</em></h4>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🧪 Étapes d'utilisation

1. **Renseignez les valeurs des paramètres ci-dessous** *(Des exemples sont préremplis – remplacez-les avec vos propres données)*
2. **Cliquez sur _Prédire_** pour estimer la densité ACP54%.
3. **Saisissez une valeur cible** pour la densité souhaitée.
4. **Sélectionnez les variables à optimiser** parmi celles proposées.
5. **Cliquez sur _Optimiser_** pour obtenir des recommandations d'ajustement.
""")

# --- Formulaire de prédiction
with st.form("form_pred"):
    st.subheader("1. Saisie des variables")
    user_display = {}
    cols = st.columns(4)

    # Créer display_to_model à partir de display_to_model_units
    display_to_model = {k: v[0] for k, v in display_to_model_units.items()}

    for i, feat in enumerate(DISPLAY_FEATURES):
        if feat != "Heure_float":
            default = DEFAULT_INPUTS[feat]
            label_model, unit = display_to_model_units[feat]
            label_text = f"{feat} ({unit})" if unit else feat
            with cols[i % 4]:
                user_display[feat] = st.number_input(label=label_text, value=default, key=feat)

    # Ajouter cellule Heure dans les colonnes pour homogénéité visuelle
    with cols[len(DISPLAY_FEATURES) % 4]:
        heure_saisie = st.time_input("Heure de la mesure", value=datetime.time(7, 0))
    user_display['Heure_float'] = heure_saisie.hour + heure_saisie.minute / 60

    submit_pred = st.form_submit_button("Prédire")

if submit_pred:
    # Reconstruire le mapping simple
    display_to_model = {k: v[0] for k, v in display_to_model_units.items()}

    # Créer le dictionnaire d’entrée pour le modèle
    user_input = {display_to_model[k]: v for k, v in user_display.items()}
    input_df = pd.DataFrame([user_input])

    # Vérification des colonnes attendues par le modèle
    if hasattr(best_model, 'feature_names_in_'):
        expected_cols = list(best_model.feature_names_in_)
        missing_cols = set(expected_cols) - set(input_df.columns)
        if missing_cols:
            st.error(f"⛔ Erreur : colonnes manquantes dans l'entrée : {missing_cols}")
        else:
            input_df = input_df[expected_cols]
            pred = best_model.predict(input_df)[0]
            st.success(f"Prédiction ACP54% sortie Echelons : **{pred:.2f}**")
            st.session_state.input_df = input_df
            st.session_state.pred = pred
    else:
        st.warning("⚠️ Le modèle ne contient pas d’attribut 'feature_names_in_'")



# --- Optimisation
if 'pred' in st.session_state:

    # Mapping pour affichage des variables avec unité
    optim_display_labels = {
        f"{feat} ({display_to_model_units[feat][1]})" if display_to_model_units[feat][1] else feat: feat
        for feat in OPTIMIZABLE
    }

    with st.form("form_opt"):
        st.subheader("2. Optimisation des paramètres")
        target = st.number_input("Valeur cible", value=st.session_state.pred)

        opt_selected_display = st.multiselect(
            "Variables à optimiser",
            list(optim_display_labels.keys())
        )
        to_opt = [optim_display_labels[k] for k in opt_selected_display]

        submit_opt = st.form_submit_button("Optimiser les variables")

    if submit_opt:
        base = st.session_state.input_df.iloc[0]
        opt_vals, err = optimize_selected(base, target, best_model, to_opt)
        df_out = pd.DataFrame({
            'Variable': to_opt,
            'Valeur actuelle': base[to_opt].values,
            'Ajustement brut': opt_vals.values
        })

        def with_arrow(row):
            delta = row['Ajustement brut'] - row['Valeur actuelle']
            icon = "🔼" if delta > 0 else "🔽" if delta < 0 else "⏺️"
            return f"{icon} {row['Ajustement brut']:.2f}"

        df_out['Ajustement proposé'] = df_out.apply(with_arrow, axis=1)
        df_out.drop(columns=['Ajustement brut'], inplace=True)

        # ✅ Ajout des unités dans la colonne 'Variable'
        df_out['Variable'] = df_out['Variable'].apply(
            lambda x: f"{x} ({display_to_model_units[x][1]})" if display_to_model_units[x][1] else x
        )

        st.subheader("Ajustements proposés")
        st.table(df_out)
        st.info(f"Ecart final |prédiction–cible| : {err:.2f}")
