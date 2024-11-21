import json
import joblib
import numpy as np

# Charger les modèles au démarrage de la Lambda pour améliorer les performances
model_e = joblib.load("Modele_score_E.joblib")
model_s = joblib.load("Modele_score_S.joblib")
model_g = joblib.load("Modele_score_G.joblib")
model_overall = joblib.load("esg_Line_regression_model.joblib")

def lambda_handler(event, context):
    try:
        # Récupérer les données d'entrée depuis le corps de la requête
        body = json.loads(event["body"])

        # Vérifier les données d'entrée pour le modèle E
        input_e = [body.get(col, None) for col in model_e.feature_names_in_]
        if None in input_e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Données manquantes pour le modèle E."})
            }

        # Vérifier les données d'entrée pour le modèle S
        input_s = [body.get(col, None) for col in model_s.feature_names_in_]
        if None in input_s:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Données manquantes pour le modèle S."})
            }

        # Vérifier les données d'entrée pour le modèle G
        input_g = [body.get(col, None) for col in model_g.feature_names_in_]
        if None in input_g:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Données manquantes pour le modèle G."})
            }

        # Prédictions pour E, S et G
        score_e = model_e.predict([input_e])[0]
        score_s = model_s.predict([input_s])[0]
        score_g = model_g.predict([input_g])[0]

        # Prédiction pour le score global (overall)
        input_overall = [score_e, score_s, score_g]
        score_overall = model_overall.predict([input_overall])[0]

        # Résultats
        result = {
            "Score_E": score_e,
            "Score_S": score_s,
            "Score_G": score_g,
            "Score_Overall": score_overall
        }

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }