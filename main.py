from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import logging
import traceback
import joblib
import numpy as np
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

# Configurez le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

# Charger les modèles au démarrage de l'application pour améliorer les performances
try:
    model_e = joblib.load("Modele_score_E.joblib")
    model_s = joblib.load("Modele_score_S.joblib")
    model_g = joblib.load("Modele_score_G.joblib")
    model_overall = joblib.load("esg_Line_regression_model.joblib")
except Exception as load_err:
    logger.error(f"Erreur lors du chargement des modèles : {load_err}")
    raise


# print(model_e.feature_names_in_)
# print(model_s.feature_names_in_)
# print(model_g.feature_names_in_)


class PredictionInput(BaseModel):
    secteur: str
    Envir_emission_GES: float = Field(alias="Envir - emission GES")
    Envir_intensite_GES: float = Field(alias="Envir - intensité des GES")
    Envir_exposition_combustibles: float = Field(alias="Envir - exposition aux secteurs des combustibles fossiles")
    Envir_utilisation_totale_energie: float = Field(alias="Envir - utilisation totale d'énergie")
    Envir_utilisation_eau: float = Field(alias="Envir - utilisation d'eau")
    Envir_intensite_utilisation_eau: float = Field(alias="Envir - intensité d'utilisation d'eau")
    Envir_risque_deforestation: float = Field(alias="Envir - risque de déforestations")
    Envir_score_prejudice_env: float = Field(alias="Envir - score de préjudice en environnement")
    Envir_politique_eau: float = Field(alias="Envir - politique concernant l'eau")
    Envir_initiative_reduction_emissions: float = Field(alias="Envir - initiative de réduction d'emissions")
    Envir_intensite_energie_ventes: float = Field(alias="Envir - intensité d'energie par ventes")
    Envir_especes_zones_protegees: float = Field(alias="Envir - especes et zones naturelles protégées")
    Envir_dechets_dangereux: float = Field(alias="Envir - dechets dangereux")
    Social_soutien_droits_humains: float = Field(alias="Social - soutien et respect des droits humains")
    Social_pas_complaisant_humains: float = Field(alias="Social - pas complaisant sur droits humains")
    Social_elimination_travail_force: float = Field(alias="Social - elimination du travail forcé")
    Social_abolition_travail_enfants: float = Field(alias="Social - abolution du travail des droits des enfants")
    Social_elimination_discrimination: float = Field(alias="Social - elimination de discrimination à l'emploi")
    Social_precaution_defis_env: float = Field(alias="Social - approche de précaution envers défis environnementaux")
    Social_responsabilite_env: float = Field(alias="Social - promeut la résponsabilité environnement")
    Social_technologie_ecologique: float = Field(alias="Social - technologie écologique")
    Social_anti_corruption: float = Field(alias="Social - anti corruption")
    Social_exposition_armes: float = Field(alias="Social - exposition aux armes controversées")
    Social_prevention_accidents: float = Field(alias="Social - polices de prevention des accidents du travail")
    Social_code_conduite: float = Field(alias="Social - code de conduite du fournisseur")
    Social_transparence_info: float = Field(alias="Social - transparence de l'information")
    Social_incidents_survenus: float = Field(alias="Social - nombres d'incidents survenus")
    Social_diligence_droits_humains: float = Field(alias="Social - processus de diligence (droits humains)")
    Social_ecarts_genres: float = Field(alias="Social - ecarts de remunaration entre genres")
    Gouv_diversite_genres_conseil: float = Field(alias="Gouv - diversité des genres au conseil")
    Gouv_politiques_anti_corruption: float = Field(alias="Gouv - politiques contre corruptiopn et pot de vin")
    Gouv_administrateurs_independants: float = Field(alias="Gouv - % d'administrateurs indépendants")
    Gouv_president_independant: float = Field(alias="Gouv - president independant")
    Gouv_administrateur_principal_independant: float = Field(alias="Gouv - admisstrateur principal  independant")
    Gouv_politique_ethique: float = Field(alias="Gouv - politique en matiere d'éthique")
    Gouv_protection_employes: float = Field(alias="Gouv - politique en matiere de protection des employés")
    Gouv_egalite_chances: float = Field(alias="Gouv - politique en matiere d'égalité des chances")
    Gouv_sante_securite: float = Field(alias="Gouv - politique en matiere de santé et sécurité")
    Gouv_droit_humain: float = Field(alias="Gouv - politique en matiere de droit humain")
    Gouv_droits_enfants: float = Field(alias="Gouv - polique en matiere de droits des enfants")
    Gouv_formation: float = Field(alias="Gouv - politique en matiuere de formation")
    Gouv_ethyque_anti_corruption: float = Field(alias="Gouv - politique en matiere d'ethyque anti corrupsition")
    Gouv_comite_remuneration: float = Field(alias="Gouv - % d'administrateur indépendants au Comité de rémunération")
    Gouv_disposition_remuneration_dirigeants: float = Field(alias="Gouv - disposition de reprise de rémunération des dirigeants")
    Gouv_lignes_directrices_dirigeants: float = Field(alias="Gouv - société a des lignes directrices en matiere d'actions des dirigeants")
    Gouv_comite_audit: float = Field(alias="Gouv - % d'administrateurs indépendants au comité d'audit")
    Gouv_bonus_ceo: float = Field(alias="Gouv - bonus versé au CEO")
    Gouv_salaire_ceo: float = Field(alias="Gouv - salaire versé au CEO")

# Initialiser l'application FastAPI
app = FastAPI()

# Ajoutez ce middleware après l'initialisation de l'application FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines, à limiter si nécessaire
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les en-têtes
)



@app.post("/predict")
async def predict(request: Request):  # Ajoutez le paramètre 'request'
    try:
        logger.info(f"Requête reçue : {await request.json()}")
        data = await request.json()
        
        # Mappage explicite des noms des champs pour chaque modèle
        feature_map_e = {
            "secteur": "secteur",
            "Envir - emission GES": "Envir - emission GES",
            "Envir - intensité des GES": "Envir - intensité des GES",
            "Envir - exposition aux secteurs des combustibles fossiles": "Envir - exposition aux secteurs des combustibles fossiles",
            "Envir - utilisation totale d'énergie": "Envir - utilisation totale d'énergie",
            "Envir - utilisation d'eau": "Envir - utilisation d'eau",
            "Envir - intensité d'utilisation d'eau": "Envir - intensité d'utilisation d'eau",
            "Envir - risque de déforestations": "Envir - risque de déforestations",
            "Envir - score de préjudice en environnement": "Envir - score de préjudice en environnement",
            "Envir - politique concernant l'eau": "Envir - politique concernant l'eau",
            "Envir - initiative de réduction d'emissions": "Envir - initiative de réduction d'emissions",
            "Envir - intensité d'energie par ventes": "Envir - intensité d'energie par ventes",
            "Envir - especes et zones naturelles protégées": "Envir - especes et zones naturelles protégées",
            "Envir - dechets dangereux": "Envir - dechets dangereux",
        }

        feature_map_s = {
            "secteur": "secteur",
            "Social - soutien et respect des droits humains": "Social - soutien et respect des droits humains",
            "Social - pas complaisant sur droits humains": "Social - pas complaisant sur droits humains",
            "Social - elimination du travail forcé": "Social - elimination du travail forcé",
            "Social - abolution du travail des droits des enfants": "Social - abolution du travail des droits des enfants",
            "Social - elimination de discrimination à l'emploi": "Social - elimination de discrimination à l'emploi",
            "Social - approche de précaution envers défis environnementaux": "Social - approche de précaution envers défis environnementaux",
            "Social - promeut la résponsabilité environnement": "Social - promeut la résponsabilité environnement",
            "Social - technologie écologique": "Social - technologie écologique",
            "Social - anti corruption": "Social - anti corruption",
            "Social - exposition aux armes controversées": "Social - exposition aux armes controversées",
            "Social - polices de prevention des accidents du travail": "Social - polices de prevention des accidents du travail",
            "Social - code de conduite du fournisseur": "Social - code de conduite du fournisseur",
            "Social - transparence de l'information": "Social - transparence de l'information",
            "Social - nombres d'incidents survenus": "Social - nombres d'incidents survenus",
            "Social - processus de diligence (droits humains)": "Social - processus de diligence (droits humains)",
            "Social - ecarts de remunaration entre genres": "Social - ecarts de remunaration entre genres",
        }

        feature_map_g = {
            "secteur": "secteur",
            "Gouv - diversité des genres au conseil": "Gouv - diversité des genres au conseil",
            "Gouv - politiques contre corruptiopn et pot de vin": "Gouv - politiques contre corruptiopn et pot de vin",
            "Gouv - % d'administrateurs indépendants": "Gouv - % d'administrateurs indépendants",
            "Gouv - president independant": "Gouv - president independant",
            "Gouv - admisstrateur principal  independant": "Gouv - admisstrateur principal  independant",
            "Gouv - politique en matiere d'éthique": "Gouv - politique en matiere d'éthique",
            "Gouv - politique en matiere de protection des employés": "Gouv - politique en matiere de protection des employés",
            "Gouv - politique en matiere d'égalité des chances": "Gouv - politique en matiere d'égalité des chances",
            "Gouv - politique en matiere de santé et sécurité": "Gouv - politique en matiere de santé et sécurité",
            "Gouv - politique en matiere de droit humain": "Gouv - politique en matiere de droit humain",
            "Gouv - polique en matiere de droits des enfants": "Gouv - polique en matiere de droits des enfants",
            "Gouv - politique en matiuere de formation": "Gouv - politique en matiuere de formation",
            "Gouv - politique en matiere d'ethyque anti corrupsition": "Gouv - politique en matiere d'ethyque anti corrupsition",
            "Gouv - % d'administrateur indépendants au Comité de rémunération": "Gouv - % d'administrateur indépendants au Comité de rémunération",
            "Gouv - disposition de reprise de rémunération des dirigeants": "Gouv - disposition de reprise de rémunération des dirigeants",
            "Gouv - société a des lignes directrices en matiere d'actions des dirigeants": "Gouv - société a des lignes directrices en matiere d'actions des dirigeants",
            "Gouv - % d'administrateurs indépendants au comité d'audit": "Gouv - % d'administrateurs indépendants au comité d'audit",
            "Gouv - bonus versé au CEO": "Gouv - bonus versé au CEO",
            "Gouv - salaire versé au CEO": "Gouv - salaire versé au CEO",
        }

        # Convertir les données JSON en DataFrame pandas
        df_e = pd.DataFrame([{feature_map_e[key]: data[key] for key in feature_map_e}])
        df_s = pd.DataFrame([{feature_map_s[key]: data[key] for key in feature_map_s}])
        df_g = pd.DataFrame([{feature_map_g[key]: data[key] for key in feature_map_g}])

        # Prédictions pour E, S et G
        score_e = model_e.predict(df_e)[0]
        score_s = model_s.predict(df_s)[0]
        score_g = model_g.predict(df_g)[0]

        # Prédiction pour le score overall
        input_overall = pd.DataFrame([{
            "Score Prédit E": score_e,
            "Score Prédit S": score_s,
            "Score Prédit G": score_g,
        }])
        score_overall = model_overall.predict(input_overall)[0]

        # Retourner les résultats
        return {
            "Score_E": score_e,
            "Score_S": score_s,
            "Score_G": score_g,
            "Score_Overall": score_overall,
        }

    except Exception as e:
        logger.error(f"Erreur dans la prédiction : {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")
