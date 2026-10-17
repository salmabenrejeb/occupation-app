import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction du domaine professionnel", page_icon="💼")


# 1) Charger le modèle et les informations de préparation (une seule fois)
@st.cache_resource
def charger():
    modele = joblib.load("model.joblib")
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modele, meta


modele, meta = charger()
cat = meta["categories"]  # listes des modalités, dans l'ordre du LabelEncoder

NOMS_DOMAINES = {
    "Col_Blanc": "Col blanc (cadres, administratif, vente, support technique)",
    "Col_Bleu": "Col bleu (métiers manuels, industrie, transport, agriculture)",
    "Services": "Services (services à la personne, sécurité, armée)",
}

# Niveau d'études : le modèle a besoin du nombre (education-num) du dataset Adult
ETUDES = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad (lycée)": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors (licence)": 13, "Masters (master)": 14,
    "Prof-school": 15, "Doctorate (doctorat)": 16,
}

# 2) Interface : le formulaire
st.title("💼 Prédire le domaine professionnel")
st.write("Remplis le profil d'une personne : le modèle prédit son **domaine professionnel**.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Âge", 17, 90, 38)
    etudes = st.selectbox("Niveau d'études", list(ETUDES.keys()), index=8)
    sexe = st.radio("Sexe", ["Male", "Female"], horizontal=True)
    heures = st.slider("Heures de travail par semaine", 1, 99, 40)
    revenu = st.checkbox("Revenu annuel supérieur à 50K$")
    gain = st.number_input("Gain en capital", min_value=0, value=0, step=100)
    perte = st.number_input("Perte en capital", min_value=0, value=0, step=100)
with col2:
    workclass = st.selectbox("Type d'employeur (workclass)", cat["workclass"], index=cat["workclass"].index("Private"))
    marital = st.selectbox("Situation maritale", cat["marital-status"], index=cat["marital-status"].index("Never-married"))
    relation = st.selectbox("Relation familiale", cat["relationship"], index=cat["relationship"].index("Not-in-family"))
    race = st.selectbox("Origine (race)", cat["race"], index=cat["race"].index("White"))
    pays = st.selectbox("Pays d'origine", cat["native-country"], index=cat["native-country"].index("United-States"))

# 3) Prédiction
if st.button("Prédire", type="primary"):
    # Mêmes transformations que dans le notebook de préparation
    ligne = {
        "age": age,
        "workclass": cat["workclass"].index(workclass),        # LabelEncoder = position dans la liste
        "education-num": ETUDES[etudes],
        "marital-status": cat["marital-status"].index(marital),
        "relationship": cat["relationship"].index(relation),
        "race": cat["race"].index(race),
        "sex": 1 if sexe == "Male" else 0,
        "capital-gain": gain,
        "capital-loss": perte,
        "hours-per-week": heures,
        "native-country": cat["native-country"].index(pays),
        "income": 1 if revenu else 0,
    }
    # Les colonnes doivent être dans le MÊME ordre qu'à l'entraînement
    X_nouveau = pd.DataFrame([ligne])[meta["features"]]

    proba = modele.predict_proba(X_nouveau)[0]
    classe = int(proba.argmax())
    domaines = meta["domaines"]

    st.success(f"Domaine prédit : **{NOMS_DOMAINES[domaines[classe]]}**")
    st.write(f"Confiance du modèle : **{proba[classe] * 100:.1f} %**")
    st.bar_chart(pd.DataFrame({"Probabilité": proba}, index=domaines))

st.caption(
    f"Modèle XGBoost entraîné sur le dataset Adult (Census). Accuracy sur le jeu de test : "
    f"{meta['test_accuracy'] * 100:.1f} %. Projet pédagogique : les prédictions ne sont pas fiables à 100 %."
)