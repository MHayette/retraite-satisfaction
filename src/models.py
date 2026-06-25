"""
models.py
---------
Modélisation ML : Régression linéaire + Random Forest
Projet : Analyse des déterminants de la satisfaction en maison de retraite
Variable cible : Satisfaction (sur 10)
Facteurs clés identifiés : Hygiene, Soins_Qualite, Reconfort, Competence_Personnel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler


# ─────────────────────────────────────────────
# VARIABLES
# ─────────────────────────────────────────────

# Variables retenues après analyse R (significatives)
FEATURES_FINALES = [
    "Soins_Qualite", "Competence_Personnel", "Reconfort", "Hygiene"
]

# Toutes les variables de services (modèle complet)
FEATURES_COMPLETES = [
    "Accueil", "Soins_Qualite", "Competence_Personnel",
    "Disponibilite_Personnel", "Reconfort", "Qualite_Restauration",
    "Hygiene", "Confort_Chambre", "Services_Annexes", "PRIX_PENSION"
]

CIBLE = "Satisfaction"


# ─────────────────────────────────────────────
# 1. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────

def preparer_donnees(df: pd.DataFrame, features: list, test_size: float = 0.2):
    """
    Sépare les données en train/test.
    Retourne X_train, X_test, y_train, y_test.
    """
    X = df[features]
    y = df[CIBLE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"✅ Train : {X_train.shape[0]} observations")
    print(f"   Test  : {X_test.shape[0]} observations")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. ÉVALUATION
# ─────────────────────────────────────────────

def evaluer_modele(y_test, y_pred, nom_modele: str) -> dict:
    """Calcule MAE, RMSE et R² sur le jeu de test."""
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n── {nom_modele} ──")
    print(f"   MAE  : {mae:.3f}")
    print(f"   RMSE : {rmse:.3f}")
    print(f"   R²   : {r2:.3f}")

    return {"modele": nom_modele, "MAE": round(mae, 3),
            "RMSE": round(rmse, 3), "R2": round(r2, 3)}


# ─────────────────────────────────────────────
# 3. RÉGRESSION LINÉAIRE MULTIPLE
# ─────────────────────────────────────────────

def regression_lineaire(df: pd.DataFrame, features: list = FEATURES_FINALES) -> dict:
    """
    Régression linéaire multiple.
    Reproduit en Python le modèle final identifié en R :
    Satisfaction ~ Soins_Qualite + Competence_Personnel + Reconfort + Hygiene
    """
    X_train, X_test, y_train, y_test = preparer_donnees(df, features)

    modele = LinearRegression()
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    # Métriques
    resultats = evaluer_modele(y_test, y_pred, "Régression Linéaire")

    # Coefficients (comme summary() en R)
    coefficients = pd.DataFrame({
        "Variable"    : features,
        "Coefficient" : modele.coef_.round(3)
    }).sort_values("Coefficient", ascending=False)

    print(f"\n   Intercept : {modele.intercept_:.3f}")
    print("\n   Coefficients :")
    print(coefficients.to_string(index=False))
    print("\n   → Hygiène est le levier n°1 (coefficient le plus élevé)")

    # Validation croisée k=5
    scores_cv = cross_val_score(modele, df[features], df[CIBLE],
                                cv=5, scoring="r2")
    print(f"\n   Validation croisée (k=5) R² : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")

    resultats["coefficients"] = coefficients
    resultats["cv_r2_mean"]   = round(scores_cv.mean(), 3)
    return resultats


# ─────────────────────────────────────────────
# 4. RANDOM FOREST
# ─────────────────────────────────────────────

def random_forest(df: pd.DataFrame, features: list = FEATURES_COMPLETES,
                  optimiser: bool = False) -> dict:
    """
    Random Forest Regressor avec validation croisée.
    Option : optimisation des hyperparamètres via GridSearchCV.
    """
    X_train, X_test, y_train, y_test = preparer_donnees(df, features)

    if optimiser:
        print("\n🔍 GridSearchCV en cours...")
        param_grid = {
            "n_estimators"  : [100, 200],
            "max_depth"     : [None, 5, 10],
            "min_samples_split" : [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        modele = grid.best_estimator_
        print(f"   Meilleurs paramètres : {grid.best_params_}")
    else:
        modele = RandomForestRegressor(n_estimators=200, random_state=42)
        modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)
    resultats = evaluer_modele(y_test, y_pred, "Random Forest")

    # Importance des features
    importance = pd.DataFrame({
        "Variable"   : features,
        "Importance" : modele.feature_importances_.round(4)
    }).sort_values("Importance", ascending=False)

    print("\n   Importance des variables :")
    print(importance.to_string(index=False))

    # Validation croisée
    scores_cv = cross_val_score(modele, df[features], df[CIBLE],
                                cv=5, scoring="r2")
    print(f"\n   Validation croisée (k=5) R² : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")

    resultats["importance"]  = importance
    resultats["cv_r2_mean"]  = round(scores_cv.mean(), 3)
    return resultats


# ─────────────────────────────────────────────
# 5. PRÉDICTIONS (scénarios)
# ─────────────────────────────────────────────

def predire_scenarios(modele_lr) -> pd.DataFrame:
    """
    Prédit la satisfaction pour 2 profils fictifs
    (reproduit la Partie 7 de l'analyse R) :
    - Profil Moyen   : 3/5 partout
    - Profil Excellent : 5/5 partout
    """
    scenarios = pd.DataFrame({
        "Soins_Qualite"         : [3, 5],
        "Competence_Personnel"  : [3, 5],
        "Reconfort"             : [3, 5],
        "Hygiene"               : [3, 5]
    }, index=["Profil Moyen", "Profil Excellent"])

    predictions = modele_lr.predict(scenarios)

    resultats = pd.DataFrame({
        "Profil"              : scenarios.index,
        "Satisfaction_prédite": predictions.round(2)
    })

    print("\n── Prédictions par scénario ──")
    print(resultats.to_string(index=False))
    return resultats


# ─────────────────────────────────────────────
# 6. VISUALISATION : FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(importance: pd.DataFrame,
                             save_path: str = "reports/figures/feature_importance.png"):
    """Génère un bar chart horizontal de l'importance des variables."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(importance["Variable"], importance["Importance"],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title("Importance des variables — Prédiction de la Satisfaction")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"💾 Graphique sauvegardé : {save_path}")


# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Chargement
    df = pd.read_csv("data/processed/retraite_clean.csv")

    print("=" * 55)
    print("   MODÉLISATION — PROJET RETRAITE")
    print("=" * 55)

    # 1. Régression linéaire (modèle final R)
    res_lr = regression_lineaire(df, FEATURES_FINALES)

    # 2. Random Forest (toutes les variables)
    res_rf = random_forest(df, FEATURES_COMPLETES, optimiser=False)

    # 3. Tableau comparatif
    print("\n── Comparaison des modèles ──")
    comparaison = pd.DataFrame([
        {"Modèle": res_lr["modele"], "MAE": res_lr["MAE"],
         "RMSE": res_lr["RMSE"], "R²": res_lr["R2"],
         "CV R²": res_lr["cv_r2_mean"]},
        {"Modèle": res_rf["modele"], "MAE": res_rf["MAE"],
         "RMSE": res_rf["RMSE"], "R²": res_rf["R2"],
         "CV R²": res_rf["cv_r2_mean"]},
    ])
    print(comparaison.to_string(index=False))

    # 4. Scénarios de prédiction
    lr_fitted = LinearRegression().fit(df[FEATURES_FINALES], df[CIBLE])
    predire_scenarios(lr_fitted)

    # 5. Graphique importance
    plot_feature_importance(res_rf["importance"])

    print("\n✅ Modélisation terminée !")
