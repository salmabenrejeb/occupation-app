import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent

# 1) Créer l'application FastAPI, brancher le dossier "static" (CSS, JS) et "templates" (HTML)
app = FastAPI(title="Prédiction du domaine professionnel")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

# 2) Charger le modèle et les informations de préparation (une seule fois, au démarrage)
modele = joblib.load(BASE / "model.joblib")
with open(BASE / "meta.json", encoding="utf-8") as f:
    meta = json.load(f)
cat = meta["categories"]  # listes des modalités, dans l'ordre du LabelEncoder (NE PAS MODIFIER)

# ---------------------------------------------------------------------------
# Textes en français (affichage uniquement : le modèle reçoit toujours les
# valeurs d'origine du dataset, donc les encodages ne changent pas)
# ---------------------------------------------------------------------------

# Niveau d'études : le modèle a besoin du nombre (education-num) du dataset Adult
ETUDES = {
    "Maternelle": 1, "Primaire (1re-4e année)": 2, "Primaire (5e-6e année)": 3,
    "Collège (7e-8e année)": 4, "9e année": 5, "10e année": 6, "11e année": 7,
    "12e année (sans diplôme)": 8, "Bac / fin de lycée": 9,
    "Études supérieures non terminées": 10, "Diplôme professionnel": 11,
    "Diplôme universitaire court (Bac+2)": 12, "Licence": 13, "Master": 14,
    "École professionnelle (droit, médecine…)": 15, "Doctorat": 16,
}

TRAD = {
    # type d'employeur
    "Federal-gov": "Fonction publique fédérale", "Local-gov": "Administration locale",
    "Private": "Secteur privé", "Self-emp-inc": "Indépendant (société)",
    "Self-emp-not-inc": "Indépendant (hors société)", "State-gov": "Administration d'État",
    "Without-pay": "Non rémunéré",
    # situation maritale
    "Divorced": "Divorcé(e)", "Married-AF-spouse": "Marié(e) (conjoint militaire)",
    "Married-civ-spouse": "Marié(e) (conjoint civil)", "Married-spouse-absent": "Marié(e) (conjoint absent)",
    "Never-married": "Célibataire", "Separated": "Séparé(e)", "Widowed": "Veuf / veuve",
    # relation familiale
    "Husband": "Mari", "Wife": "Épouse", "Not-in-family": "Hors famille",
    "Other-relative": "Autre parent", "Own-child": "Enfant du foyer", "Unmarried": "Non marié(e)",
    # origine
    "Amer-Indian-Eskimo": "Amérindien / Inuit", "Asian-Pac-Islander": "Asiatique / Îles du Pacifique",
    "Black": "Noir(e)", "Other": "Autre", "White": "Blanc(he)",
    # pays
    "Cambodia": "Cambodge", "Canada": "Canada", "China": "Chine", "Columbia": "Colombie", "Cuba": "Cuba",
    "Dominican-Republic": "République dominicaine", "Ecuador": "Équateur", "El-Salvador": "Salvador",
    "England": "Angleterre", "France": "France", "Germany": "Allemagne", "Greece": "Grèce",
    "Guatemala": "Guatemala", "Haiti": "Haïti", "Holand-Netherlands": "Pays-Bas", "Honduras": "Honduras",
    "Hong": "Hong Kong", "Hungary": "Hongrie", "India": "Inde", "Iran": "Iran", "Ireland": "Irlande",
    "Italy": "Italie", "Jamaica": "Jamaïque", "Japan": "Japon", "Laos": "Laos", "Mexico": "Mexique",
    "Nicaragua": "Nicaragua", "Outlying-US(Guam-USVI-etc)": "Territoires américains (Guam…)",
    "Peru": "Pérou", "Philippines": "Philippines", "Poland": "Pologne", "Portugal": "Portugal",
    "Puerto-Rico": "Porto Rico", "Scotland": "Écosse", "South": "Sud (« South »)", "Taiwan": "Taïwan",
    "Thailand": "Thaïlande", "Trinadad&Tobago": "Trinité-et-Tobago", "United-States": "États-Unis",
    "Vietnam": "Viêt Nam", "Yugoslavia": "Yougoslavie",
}

METIERS_FR = {
    "Prof-specialty": "Professions intellectuelles", "Exec-managerial": "Cadres dirigeants",
    "Adm-clerical": "Employés administratifs", "Sales": "Vente", "Tech-support": "Support technique",
    "Craft-repair": "Artisanat et réparation", "Machine-op-inspct": "Opérateurs de machines",
    "Transport-moving": "Transport", "Handlers-cleaners": "Manutention et nettoyage",
    "Farming-fishing": "Agriculture et pêche", "Other-service": "Autres services",
    "Protective-serv": "Sécurité et protection", "Priv-house-serv": "Employés de maison",
    "Armed-Forces": "Forces armées",
}

# Les 3 domaines (même regroupement que dans le notebook). "cle" sert pour les couleurs CSS.
DOMAINES = {
    "Col_Blanc": {
        "cle": "blanc", "code": 0, "nom": "Col blanc", "icone": "user",
        "sous_titre": "cadres, administratif, vente, support technique",
        "resume": "Gestion, encadrement, intellectuel, vente, support",
        "explication": "Le modèle estime que ce profil ressemble davantage aux personnes qui exercent des "
                       "métiers administratifs, intellectuels, commerciaux ou d'encadrement.",
        "metiers": ["Prof-specialty", "Exec-managerial", "Adm-clerical", "Sales", "Tech-support"],
    },
    "Col_Bleu": {
        "cle": "bleu", "code": 1, "nom": "Col bleu", "icone": "tool",
        "sous_titre": "métiers manuels, industrie, transport, agriculture",
        "resume": "Manuel, bâtiment, industrie, logistique, agriculture",
        "explication": "Le modèle estime que ce profil ressemble davantage aux personnes qui exercent des "
                       "métiers manuels : artisanat, industrie, transport, manutention ou agriculture.",
        "metiers": ["Craft-repair", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing"],
    },
    "Services": {
        "cle": "services", "code": 2, "nom": "Services", "icone": "shield",
        "sous_titre": "services à la personne, sécurité, armée",
        "resume": "Services directs, protection, sécurité",
        "explication": "Le modèle estime que ce profil ressemble davantage aux personnes qui exercent des "
                       "métiers de services : services aux particuliers, sécurité, protection ou armée.",
        "metiers": ["Other-service", "Protective-serv", "Priv-house-serv", "Armed-Forces"],
    },
}

# Profils d'exemple : de vraies lignes du dataset Adult, avec leur métier réel.
# Niveau = difficulté pour le modèle (facile : très sûr / moyen : hésite / difficile : se trompe).
EXEMPLES = [
    {"id": "f1", "niveau": "Facile", "titre": "Cadre dirigeant", "detail": "52 ans · Licence · indépendant",
     "domaine_reel": "Col_Blanc", "metier_reel": "Exec-managerial",
     "profil": {"age": 52, "etudes": "Licence", "sexe": "Male", "heures": 50, "revenu": True, "gain": 0, "perte": 0,
                "workclass": "Self-emp-inc", "marital": "Married-civ-spouse", "relation": "Husband",
                "race": "White", "pays": "United-States"}},
    {"id": "f2", "niveau": "Facile", "titre": "Ouvrier qualifié", "detail": "35 ans · 10e année · privé",
     "domaine_reel": "Col_Bleu", "metier_reel": "Craft-repair",
     "profil": {"age": 35, "etudes": "10e année", "sexe": "Male", "heures": 40, "revenu": False, "gain": 0, "perte": 0,
                "workclass": "Private", "marital": "Married-civ-spouse", "relation": "Husband",
                "race": "White", "pays": "United-States"}},
    {"id": "m1", "niveau": "Moyen", "titre": "Artisan, 50 h/semaine", "detail": "38 ans · Bac · privé",
     "domaine_reel": "Col_Bleu", "metier_reel": "Craft-repair",
     "profil": {"age": 38, "etudes": "Bac / fin de lycée", "sexe": "Male", "heures": 50, "revenu": False, "gain": 0, "perte": 0,
                "workclass": "Private", "marital": "Married-civ-spouse", "relation": "Husband",
                "race": "White", "pays": "United-States"}},
    {"id": "m2", "niveau": "Moyen", "titre": "Employée de service", "detail": "58 ans · Collège · veuve",
     "domaine_reel": "Services", "metier_reel": "Other-service",
     "profil": {"age": 58, "etudes": "Collège (7e-8e année)", "sexe": "Female", "heures": 30, "revenu": False, "gain": 0, "perte": 0,
                "workclass": "Private", "marital": "Widowed", "relation": "Not-in-family",
                "race": "Other", "pays": "United-States"}},
    {"id": "d1", "niveau": "Difficile", "titre": "Service, diplômée", "detail": "23 ans · Licence · célibataire",
     "domaine_reel": "Services", "metier_reel": "Other-service",
     "profil": {"age": 23, "etudes": "Licence", "sexe": "Female", "heures": 30, "revenu": False, "gain": 0, "perte": 0,
                "workclass": "Private", "marital": "Never-married", "relation": "Own-child",
                "race": "White", "pays": "United-States"}},
    {"id": "d2", "niveau": "Difficile", "titre": "Artisan diplômé", "detail": "35 ans · Licence · administration locale",
     "domaine_reel": "Col_Bleu", "metier_reel": "Craft-repair",
     "profil": {"age": 35, "etudes": "Licence", "sexe": "Male", "heures": 40, "revenu": False, "gain": 0, "perte": 0,
                "workclass": "Local-gov", "marital": "Married-civ-spouse", "relation": "Husband",
                "race": "White", "pays": "United-States"}},
]

# Valeurs affichées par défaut dans le formulaire
DEFAUT = {
    "age": 34, "etudes": "Bac / fin de lycée", "sexe": "Male", "heures": 40, "revenu": False,
    "gain": 0, "perte": 0, "workclass": "Private", "marital": "Never-married",
    "relation": "Not-in-family", "race": "White", "pays": "United-States",
}


def fr(x: float) -> str:
    """Nombre au format français : 98.4 -> '98,4'."""
    return f"{x:.1f}".replace(".", ",")


def choix(colonne: str):
    """Liste (valeur d'origine, libellé français) pour une liste déroulante, triée par libellé."""
    paires = [(v, TRAD.get(v, v)) for v in cat[colonne]]
    return sorted(paires, key=lambda p: p[1])


def page(request: Request, valeurs: dict, resultat=None, erreur=None, exemple_actif=None):
    """Prépare toutes les données envoyées à la page HTML."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "v": valeurs,
            "choix": {
                "workclass": choix("workclass"), "marital": choix("marital-status"),
                "relation": choix("relationship"), "race": choix("race"), "pays": choix("native-country"),
                "etudes": [(k, k) for k in ETUDES],
            },
            "domaines": list(DOMAINES.values()),
            "metiers_fr": METIERS_FR,
            "exemples": EXEMPLES,
            "exemple_actif": exemple_actif,
            "resultat": resultat,
            "erreur": erreur,
            "accuracy": fr(meta["test_accuracy"] * 100),
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
    exemple: str | None = Form(None),         # identifiant du profil d'exemple (si utilisé)
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
    noms_domaines = [d.replace(" ", "_") for d in meta["domaines"]]
    gagnant = DOMAINES[noms_domaines[classe]]

    resultat = {
        "domaine": gagnant,
        "confiance": fr(float(proba[classe]) * 100),
        "details": [
            {"domaine": DOMAINES[n], "pct": round(float(p) * 100, 1), "pct_txt": fr(float(p) * 100)}
            for n, p in zip(noms_domaines, proba)
        ],
        "exemple": None,
    }

    # Si c'est un profil d'exemple : on compare avec le vrai métier du dataset
    ex = next((e for e in EXEMPLES if e["id"] == exemple), None)
    if ex:
        resultat["exemple"] = {
            "niveau": ex["niveau"],
            "titre": ex["titre"],
            "domaine_reel": DOMAINES[ex["domaine_reel"]]["nom"],
            "metier_reel": METIERS_FR[ex["metier_reel"]],
            "correct": ex["domaine_reel"] == noms_domaines[classe],
        }
    return page(request, valeurs, resultat=resultat, exemple_actif=ex["id"] if ex else None)