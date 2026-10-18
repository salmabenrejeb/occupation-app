import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent

# 1) Créer l'application FastAPI, brancher le dossier "static" (CSS) et "templates" (HTML)
app = FastAPI(title="Prédiction du domaine professionnel")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

# 2) Charger le modèle et les informations de préparation (une seule fois, au démarrage)
modele = joblib.load(BASE / "model.joblib")
with open(BASE / "meta.json", encoding="utf-8") as f:
    meta = json.load(f)
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

# Valeurs affichées par défaut dans le formulaire
DEFAUT = {
    "age": 38, "etudes": "HS-grad (lycée)", "sexe": "Male", "heures": 40, "revenu": False,
    "gain": 0, "perte": 0, "workclass": "Private", "marital": "Never-married",
    "relation": "Not-in-family", "race": "White", "pays": "United-States",
}


def page(request: Request, valeurs: dict, resultat=None, erreur=None):
    """Prépare toutes les données envoyées à la page HTML."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "v": valeurs,                     # valeurs du formulaire
            "cat": cat,                       # listes de choix
            "etudes": list(ETUDES.keys()),
            "resultat": resultat,
            "erreur": erreur,
            "accuracy": meta["test_accuracy"] * 100,
        },
    )


# 3) Route GET : afficher la page avec le formulaire
@app.get("/", response_class=HTMLResponse)
def accueil(request: Request):
    return page(request, DEFAUT)


# 4) Route POST : recevoir le formulaire, prédire, ré-afficher la page avec le résultat
@app.post("/", response_class=HTMLResponse)
def predire(
    request: Request,
    age: int = Form(..., ge=17, le=90),
    etudes: str = Form(...),
    sexe: str = Form(...),
    heures: int = Form(..., ge=1, le=99),
    revenu: str | None = Form(None),          # une case non cochée n'est pas envoyée
    gain: int = Form(0, ge=0),
    perte: int = Form(0, ge=0),
    workclass: str = Form(...),
    marital: str = Form(...),
    relation: str = Form(...),
    race: str = Form(...),
    pays: str = Form(...),
):
    valeurs = {
        "age": age, "etudes": etudes, "sexe": sexe, "heures": heures, "revenu": revenu is not None,
        "gain": gain, "perte": perte, "workclass": workclass, "marital": marital,
        "relation": relation, "race": race, "pays": pays,
    }
    try:
        # Mêmes transformations que dans le notebook de préparation
        ligne = {
            "age": age,
            "workclass": cat["workclass"].index(workclass),   # LabelEncoder = position dans la liste
            "education-num": ETUDES[etudes],
            "marital-status": cat["marital-status"].index(marital),
            "relationship": cat["relationship"].index(relation),
            "race": cat["race"].index(race),
            "sex": 1 if sexe == "Male" else 0,
            "capital-gain": gain,
            "capital-loss": perte,
            "hours-per-week": heures,
            "native-country": cat["native-country"].index(pays),
            "income": 1 if revenu is not None else 0,
        }
    except (ValueError, KeyError):
        return page(request, valeurs, erreur="Une des valeurs choisies n'est pas valide.")

    # Les colonnes doivent être dans le MÊME ordre qu'à l'entraînement
    X_nouveau = pd.DataFrame([ligne])[meta["features"]]
    proba = modele.predict_proba(X_nouveau)[0]
    classe = int(proba.argmax())
    domaines = meta["domaines"]

    resultat = {
        "domaine": NOMS_DOMAINES[domaines[classe]],
        "confiance": round(float(proba[classe]) * 100, 1),
        "details": [{"nom": d.replace("_", " "), "pct": round(float(p) * 100, 1)} for d, p in zip(domaines, proba)],
    }
    return page(request, valeurs, resultat=resultat)