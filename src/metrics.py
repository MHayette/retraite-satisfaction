"""
metrics.py
----------
Calcul des métriques clés de satisfaction
Projet : Analyse des déterminants de la satisfaction en maison de retraite
300 résidents | Variable cible : Satisfaction (sur 10)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# COLONNES SERVICES
# ─────────────────────────────────────────────

COLS_SERVICES = [
    "Accueil", "Soins_Qualite", "Competence_Personnel",
    "Disponibilite_Personnel", "Reconfort", "Qualite_Restauration",
    "Hygiene", "Confort_Chambre", "Services_Annexes"
]


# ─────────────────────────────────────────────
# 1. SATISFACTION GLOBALE
# ─────────────────────────────────────────────

def satisfaction_globale(df: pd.DataFrame) -> dict:
    """
    Calcule les métriques globales de satisfaction (sur 10).
    """
    s = df["Satisfaction"]
    metriques = {
        "moyenne"  : round(s.mean(), 2),
        "mediane"  : round(s.median(), 2),
        "std"      : round(s.std(), 2),
        "min"      : round(s.min(), 2),
        "max"      : round(s.max(), 2),
        "taux_satisfaits"    : round((s >= 6).mean() * 100, 1),   # note >= 6/10
        "taux_insatisfaits"  : round((s < 5).mean() * 100, 1),    # note < 5/10
    }

    print("── Satisfaction Globale ──")
    for k, v in metriques.items():
        print(f"   {k:<22} : {v}")
    return metriques


# ─────────────────────────────────────────────
# 2. NPS INTERNE
# ─────────────────────────────────────────────

def nps_interne(df: pd.DataFrame) -> dict:
    """
    Calcule le Net Promoter Score interne (adapté sur 10) :
    - Promoteurs  : Satisfaction >= 8
    - Passifs     : Satisfaction 6-7
    - Détracteurs : Satisfaction <= 5
    """
    s = df["Satisfaction"]
    n = len(s)

    promoteurs   = (s >= 8).sum()
    passifs      = ((s >= 6) & (s < 8)).sum()
    detracteurs  = (s <= 5).sum()

    nps = round(((promoteurs - detracteurs) / n) * 100, 1)

    resultats = {
        "promoteurs_%"   : round(promoteurs / n * 100, 1),
        "passifs_%"      : round(passifs / n * 100, 1),
        "detracteurs_%"  : round(detracteurs / n * 100, 1),
        "NPS"            : nps
    }

    print("\n── NPS Interne ──")
    print(f"   Promoteurs  (≥8)  : {resultats['promoteurs_%']}%")
    print(f"   Passifs     (6-7) : {resultats['passifs_%']}%")
    print(f"   Détracteurs (≤5)  : {resultats['detracteurs_%']}%")
    print(f"   NPS               : {nps}")
    return resultats


# ─────────────────────────────────────────────
# 3. SATISFACTION PAR SERVICE (sur 5)
# ─────────────────────────────────────────────

def satisfaction_par_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la moyenne, médiane et écart-type de chaque service.
    Classement du moins satisfaisant au plus satisfaisant.
    """
    resultats = df[COLS_SERVICES].agg(["mean", "median", "std"]).T.round(2)
    resultats.columns = ["Moyenne", "Médiane", "Écart-type"]
    resultats = resultats.sort_values("Moyenne")

    print("\n── Satisfaction par Service (sur 5) ──")
    print(resultats.to_string())
    return resultats


# ─────────────────────────────────────────────
# 4. SATISFACTION PAR PROFIL
# ─────────────────────────────────────────────

def satisfaction_par_profil(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Calcule la satisfaction moyenne par groupe (Public, Sexe, Formule, CSP).
    """
    resultats = (
        df.groupby(variable)["Satisfaction"]
        .agg(["mean", "median", "std", "count"])
        .round(2)
        .rename(columns={
            "mean"   : "Moyenne",
            "median" : "Médiane",
            "std"    : "Écart-type",
            "count"  : "Effectif"
        })
        .sort_values("Moyenne")
    )

    print(f"\n── Satisfaction par {variable} ──")
    print(resultats.to_string())
    return resultats


# ─────────────────────────────────────────────
# 5. TAUX DE PARTICIPATION AUX ACTIVITÉS
# ─────────────────────────────────────────────

def taux_participation(df: pd.DataFrame) -> dict:
    """
    Calcule le taux de participation aux activités global
    et par type de Public (Valide, Semi-valide, Dépendant).
    """
    # Participation globale (Activite = 1 si participant, 0 sinon)
    # Adapter selon le format réel de la colonne Activite
    if df["Activite"].dtype in ["float64", "int64"]:
        taux_global = round(df["Activite"].mean() * 100, 1)
    else:
        # Si catégorielle : compter les "Oui" ou valeur positive
        valeur_positive = df["Activite"].value_counts().index[0]
        taux_global = round((df["Activite"] == valeur_positive).mean() * 100, 1)

    # Par Public
    taux_par_public = (
        df.groupby("Public")["Activite"]
        .mean()
        .mul(100)
        .round(1)
        .rename("Taux participation (%)")
    )

    print("\n── Taux de Participation aux Activités ──")
    print(f"   Global : {taux_global}%")
    print("\n   Par type de Public :")
    print(taux_par_public.to_string())

    return {"global": taux_global, "par_public": taux_par_public}


# ─────────────────────────────────────────────
# 6. CORRÉLATIONS AVEC LA SATISFACTION
# ─────────────────────────────────────────────

def correlations_satisfaction(df: pd.DataFrame) -> pd.Series:
    """
    Calcule les corrélations de Pearson entre chaque service
    et la satisfaction globale. Classement par force décroissante.
    """
    cols = COLS_SERVICES + ["PRIX_PENSION"]
    corr = (
        df[cols + ["Satisfaction"]]
        .corr()["Satisfaction"]
        .drop("Satisfaction")
        .sort_values(ascending=False)
        .round(3)
    )

    print("\n── Corrélations avec la Satisfaction (Pearson) ──")
    print(corr.to_string())
    print(f"\n   → Corrélation Prix / Satisfaction : {corr['PRIX_PENSION']:.3f}")
    return corr


# ─────────────────────────────────────────────
# RAPPORT COMPLET
# ─────────────────────────────────────────────

def rapport_complet(df: pd.DataFrame) -> dict:
    """Lance tous les calculs et retourne un dictionnaire de résultats."""
    print("=" * 50)
    print("   RAPPORT DE MÉTRIQUES — PROJET RETRAITE")
    print("=" * 50)

    resultats = {
        "satisfaction_globale"   : satisfaction_globale(df),
        "nps"                    : nps_interne(df),
        "par_service"            : satisfaction_par_service(df),
        "par_public"             : satisfaction_par_profil(df, "Public"),
        "par_sexe"               : satisfaction_par_profil(df, "Sexe"),
        "par_formule"            : satisfaction_par_profil(df, "Formule"),
        "participation"          : taux_participation(df),
        "correlations"           : correlations_satisfaction(df),
    }

    print("\n" + "=" * 50)
    print("✅ Rapport terminé !")
    return resultats


# ─────────────────────────────────────────────
# EXÉCUTION DIRECTE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("data/processed/retraite_clean.csv")
    rapport_complet(df)
