import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, roc_auc_score

# Étape 1: Chargement des données
file_path = 'KPIs for telecommunication.csv'
# Le fichier CSV utilise des points-virgules comme délimiteurs
telecom_data = pd.read_csv(file_path, delimiter=';')

# Étape 2: Analyse statistique de base des variables
# Calcule les statistiques descriptives pour chaque colonne (KPI)
statistical_summary = telecom_data.describe()
# Calcule la variance pour chaque KPI
variance = telecom_data.var()
statistical_summary.loc['variance'] = variance
# Calcule l'interquartile range (IQR) pour chaque KPI
iqr = telecom_data.quantile(0.75) - telecom_data.quantile(0.25)
statistical_summary.loc['IQR'] = iqr
# Affichage des statistiques descriptives
print(statistical_summary)

# Étape 3: Évaluation des données manquantes
# Calcule le pourcentage de valeurs manquantes pour chaque KPI
missing_values_percentage = telecom_data.isnull().mean() * 100
print(missing_values_percentage)

# Étape 4: Traitement des valeurs manquantes
# Crée un objet imputer qui remplit les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
# Applique l'imputation sur les données pour gérer les valeurs manquantes
telecom_data_imputed = pd.DataFrame(imputer.fit_transform(telecom_data), columns=telecom_data.columns)
# Vérifie s'il reste des valeurs manquantes
missing_after_imputation = telecom_data_imputed.isnull().sum()
print(missing_after_imputation, telecom_data_imputed.head())

# Étape 5: Division des données en ensembles d'apprentissage et de test
# Divise les données en 70% pour l'apprentissage et 30% pour le test
train_data, test_data = train_test_split(telecom_data_imputed, test_size=0.3, random_state=42)
print(train_data.shape, test_data.shape)


# Étape 6: Construction et entraînement du modèle de forêt d'isolement

# Puisque nous n'avons pas d'étiquettes, nous ne pouvons pas utiliser roc_auc_score pour la recherche de grille.
# Nous allons procéder en entraînant une Forêt d'Isolation avec des paramètres par défaut raisonnables.

# Nous définirons un petit ensemble de paramètres à tester, car une recherche exhaustive n'est pas réalisable sans fonction de score.
params_to_try = [
    {'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.01},
    {'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.05},
    {'n_estimators': 200, 'max_samples': 'auto', 'contamination': 0.01},
    {'n_estimators': 200, 'max_samples': 'auto', 'contamination': 0.05},
]

# Nous allons stocker les modèles et leurs scores d'anomalie respectifs pour comparaison.
models = []
anomaly_scores = []

for params in params_to_try:
    # Initialisation de la Forêt d'Isolation avec l'ensemble actuel de paramètres
    iso_forest = IsolationForest(random_state=42, n_estimators=params['n_estimators'],
                                 max_samples=params['max_samples'], contamination=params['contamination'])
    # Entraînement du modèle
    iso_forest.fit(train_data)
    # Calcul des scores d'anomalie
    scores = -iso_forest.decision_function(test_data)
    # Stockage du modèle et des scores
    models.append(iso_forest)
    anomaly_scores.append(scores)

# Maintenant, visualisons la distribution des scores d'anomalie pour chaque modèle
fig, axs = plt.subplots(len(params_to_try), figsize=(10, 5*len(params_to_try)))
for i, scores in enumerate(anomaly_scores):
    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=50)
    plt.title(f"Modèle {i+1} - n_estimators: {params_to_try[i]['n_estimators']}, contamination: {params_to_try[i]['contamination']}")
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")
    plt.show()

# Étape 7: Calcul des scores d'anomalies sur l'ensemble de test
anomaly_scores = iso_forest_default.decision_function(test_data)
# Transforme les scores d'anomalie pour une meilleure interprétation
anomaly_scores_transformed = -anomaly_scores
print("the first 10 anomaly scores: ", anomaly_scores_transformed[:10])

# Étape 8: Identification des anomalies
# Convertit les scores d'anomalie en DataFrame pour une manipulation plus facile
anomaly_scores_df = pd.DataFrame(anomaly_scores_transformed, columns=['Anomaly_Score'])
# Ajoute les scores au jeu de données de test
test_data_with_scores = test_data.copy()
test_data_with_scores['Anomaly_Score'] = anomaly_scores_df
# Trie les données de test avec les scores pour trouver les scores d'anomalie les plus élevés et les plus bas
highest_anomaly_scores = test_data_with_scores.sort_values('Anomaly_Score', ascending=False).head(5)
lowest_anomaly_scores = test_data_with_scores.sort_values('Anomaly_Score').head(5)

# Étape 9: Analyse des résultats
# Affiche les observations avec les scores d'anomalie les plus élevés et les plus bas
print(highest_anomaly_scores, lowest_anomaly_scores)

# Analyse de Résultats:
# Les observations avec les scores d'anomalie les plus élevés sont considérées comme les plus atypiques par le modèle. 
# Cela peut indiquer des comportements ou des performances qui s'écartent significativement de la norme.
# À l'inverse, les observations avec les scores d'anomalie les plus bas sont considérées comme normales.
