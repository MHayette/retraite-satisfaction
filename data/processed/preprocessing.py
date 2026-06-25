"""
preprocessing.py
----------------
Chargement, nettoyage et encodage du jeu de données Retraite.xlsx
Projet : Analyse des déterminants de la satisfaction en maison de retraite
300 résidents | 16 variables | Variable cible : Satisfaction (sur 10)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# COLONNES DU JEU DE DONNÉES
# ─────────────────────────────────────────────

COLONNES = [
    "Formule", "Public", "Activite", "Accueil", "Soins_Qualite",
    "Competence_Personnel", "Disponibilite_Personnel", "Reconfort",
    "Qualite_Restauration", "Hygiene", "Confort_Chambre", "Services_Annexes",
    "Satisfaction", "PRIX_PENSION", "Sexe", "CSP"
]

# Variables de services (notées sur 5)
COLS_SERVICES = [
    "Accueil", "Soins_Qualite", "Competence_Personnel",
    "Disponibilite_Personnel", "Reconfort", "Qualite_Restauration",
    "Hygiene", "Confort_Chambre", "Services_Annexes"
]

# Variables catégorielles
COLS_QUALI = ["Formule", "Public", "Sexe", "CSP"]

# Variable cible
CIBLE = "Satisfaction"


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(path: str = "data/raw/Retraite.xlsx") -> pd.DataFrame:
    """Charge le fichier Excel et renomme les colonnes."""
    df = pd.read_excel(path)
    df.columns = COLONNES
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
# 2. INSPECTION
# ─────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Affiche un aperçu complet du jeu de données."""
    print("\n── Aperçu (5 premières lignes) ──")
    print(df.head())

    print("\n── Types de colonnes ──")
    print(df.dtypes)

    print("\n── Valeurs manquantes par colonne ──")
    manquants = df.isnull().sum()
    print(manquants[manquants > 0])

    print("\n── Statistiques descriptives (services) ──")
    print(df[COLS_SERVICES].describe().round(2))

    print(f"\n── Satisfaction globale ──")
    print(df[CIBLE].describe().round(2))

    print(f"\n── Prix de la pension ──")
    print(df["PRIX_PENSION"].describe().round(2))


# ─────────────────────────────────────────────
# 3. NETTOYAGE
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le jeu de données :
    - Supprime les doublons
    - Impute les valeurs manquantes par la médiane (variables numériques)
    - Convertit les variables qualitatives en catégories
    """
    df = df.copy()

    # Supprimer les doublons
    nb_doublons = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"🗑️  Doublons supprimés : {nb_doublons}")

    # Imputation par la médiane pour les variables de services
    for col in COLS_SERVICES:
        if df[col].isnull().sum() > 0:
            mediane = df[col].median()
            df[col].fillna(mediane, inplace=True)
            print(f"📊 '{col}' → imputation médiane ({mediane:.2f})")

    # Conversion des variables qualitatives
    df["Sexe"] = df["Sexe"].replace({
        "Un homme": "homme",
        "Une femme": "femme"
    }).astype("category")

    for col in ["Formule", "Public", "CSP"]:
        df[col] = df[col].astype("category")

    print(f"\n✅ Nettoyage terminé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"   Valeurs manquantes restantes : {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 4. DÉTECTION DES OUTLIERS (méthode IQR)
# ─────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les outliers via la méthode IQR sur les variables numériques.
    Retourne un rapport résumant les outliers par colonne.
    """
    cols = COLS_SERVICES + [CIBLE, "PRIX_PENSION"]
    resultats = []

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < borne_inf) | (df[col] > borne_sup)).sum()
        resultats.append({
            "colonne"    : col,
            "Q1"         : round(Q1, 2),
            "Q3"         : round(Q3, 2),
            "IQR"        : round(IQR, 2),
            "borne_inf"  : round(borne_inf, 2),
            "borne_sup"  : round(borne_sup, 2),
            "n_outliers" : n_outliers
        })

    rapport = pd.DataFrame(resultats)
    print("\n── Rapport outliers (IQR) ──")
    print(rapport.to_string(index=False))
    return rapport


# ─────────────────────────────────────────────
# 5. ENCODAGE POUR LE ML
# ─────────────────────────────────────────────

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles pour scikit-learn :
    - Sexe, Public, Formule → LabelEncoder
    - CSP → get_dummies (one-hot)
    """
    df = df.copy()
    le = LabelEncoder()

    for col in ["Sexe", "Public", "Formule"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        print(f"🔢 '{col}' encodée → '{col}_enc'")

    # One-hot pour CSP
    dummies = pd.get_dummies(df["CSP"], prefix="CSP", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    print(f"🔡 'CSP' encodée (one-hot) → {list(dummies.columns)}")

    return df


# ─────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde le DataFrame en CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 Fichier sauvegardé : {path}")


# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Chargement
    df_raw = load_data("data/raw/Retraite.xlsx")

    # 2. Inspection
    inspect_data(df_raw)

    # 3. Nettoyage
    df_clean = clean_data(df_raw)

    # 4. Outliers
    detect_outliers(df_clean)

    # 5. Sauvegarde données nettoyées
    save_data(df_clean, "data/processed/retraite_clean.csv")

    # 6. Encodage pour ML
    df_encoded = encode_data(df_clean)

    # 7. Sauvegarde données encodées
    save_data(df_encoded, "data/processed/retraite_encoded.csv")

    print("\n✅ Pipeline de prétraitement terminé !")
