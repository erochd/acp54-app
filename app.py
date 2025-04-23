import streamlit as st
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt
import os
import requests

# --- Largeur pleine page ---
st.set_page_config(layout="wide")

# --- Chargement du modèle
@st.cache_resource
def load_model():
    url = "https://huggingface.co/erochd/acp54-app/resolve/main/best_model_y2.pkl"
    local_path = "best_model_y2.pkl"
    if not os.path.exists(local_path):
        with st.spinner("🔄 Téléchargement du modèle depuis Hugging Face..."):
            response = requests.get(url)
            with open(local_path, "wb") as f:
                f.write(response.content)
    with open(local_path, "rb") as f:
        model = pickle.load(f)
    return model

best_model = load_model()

# --- Données d’entrée
DEFAULT_INPUTS = {
    '604JFIC214.PV - weak phos acid condenst (M3/H)': 51.85,
    '604JTI211.PV - WEAK PHOS ACD CONDST CO0 (DEGC)': 51.815,
    '604JTI208.PV (°C)': 57.1,
    '604AFI063.PV - 5B STM FRM DISTRIBUTION (T/H)': 24.625,
    '604APIC066.PV - 5B STM TO DESUPERHEATER (BAR)': 1.975,
    '604ATI064.PV - 5B STM TO SEPARATOR (DEGC)': 156.07,
    '604ATI068A.PV - 2BARG STM TO EVAP J (°C)': 130.11,
    '604JFI221.PV - LP STEAM TO EVA AE02 (T/H)': 19.385,
    '604JTIC223.PV - LP STM FRM SEPARTOR A (DEGC)': 74.825,
    '604JPI225.PV (BAR)': -0.49,
    '604JPI226.PV - EVAP HEATR AE02 (BAR)': -0.03,
    '604JPI228.PV (BAR)': 1.865,
    '604JLI241.PV - FLSH CHAMBR AD01 (%)': 103.125,
    '604JPIC242.PV - FLASH CHAMBR AD01 (mm/hg)': 131.4,
    '604JPI246.PV - VAPOR TO FSA (mm/hg)': 133.85,
    '604JFIC250.PV - EVA PMP AP05 PHOS.ACID (M3/H)': 10.75,
    'ACP29% entré Echelons': 1260.0
}
ALL_FEATURES = list(DEFAULT_INPUTS.keys())
OPTIMIZABLE = [f for f in ALL_FEATURES if f != 'ACP29% entré Echelons']

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
### Étapes d’utilisation

1. **Renseignez les valeurs des paramètres ci-dessous** *(Des exemples sont préremplis – remplacez-les avec vos propres données)*
2. **Cliquez sur _Prédire_** pour estimer la densité ACP54%.
3. **Saisissez une valeur cible** pour la densité souhaitée.
4. **Sélectionnez les variables à optimiser** parmi celles proposées.
5. **Cliquez sur _Optimiser_** pour obtenir des recommandations d’ajustement.
""")

# --- Formulaire de prédiction
with st.form("form_pred"):
    st.subheader("1. Saisie des variables")
    user = {}
    cols = st.columns(4)
    for i, feat in enumerate(ALL_FEATURES):
        default = DEFAULT_INPUTS[feat]
        display_name = feat[:40] + "..." if len(feat) > 43 else feat
        with cols[i % 4]:
            st.markdown(
                f'<div title="{feat}" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 250px;"><strong>{display_name}</strong></div>',
                unsafe_allow_html=True
            )
            user[feat] = st.number_input(
                label=feat,
                value=default,
                key=feat,
                format="%.3f",
                label_visibility="collapsed"
            )
    submit_pred = st.form_submit_button("Prédire")

if submit_pred:
    input_df = pd.DataFrame([user], columns=ALL_FEATURES)
    pred = best_model.predict(input_df)[0]
    st.success(f"Prédiction ACP54% sortie Echelons : **{pred:.2f}**")
    st.session_state.input_df = input_df
    st.session_state.pred = pred

# --- Formulaire d'optimisation
if 'pred' in st.session_state:
    with st.form("form_opt"):
        st.subheader("2. Optimisation des paramètres")
        target = st.number_input("Valeur cible", value=st.session_state.pred)
        to_opt = st.multiselect("Variables à optimiser", OPTIMIZABLE)
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
            return f"{icon} {row['Ajustement brut']:.2f}"
        df_out['Ajustement proposé'] = df_out.apply(with_arrow, axis=1)
        df_out.drop(columns=['Ajustement brut'], inplace=True)
        st.subheader("Ajustements proposés")
        st.table(df_out)
        st.info(f"Erreur finale |prédiction–cible| : {err:.2f}")
