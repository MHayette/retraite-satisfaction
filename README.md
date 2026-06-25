# retraite-satisfaction
Analysis of resident satisfaction determinants in nursing homes — EDA, KPI metrics, statistical modeling (scikit-learn, R) with visualizations.

# 🏠 Retraite — Analyse des déterminants de la satisfaction globale en maison de retraite

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/statut-en%20cours-yellow)

## 📋 Description du projet

Ce projet analyse les **déterminants de la satisfaction globale** des résidents en maison de retraite à partir du jeu de données `Retraite`.

L'objectif est double :
- **Comprendre** quels facteurs (âge, dépendance, durée de séjour, service, participation aux activités) influencent le plus la satisfaction des résidents.
- **Modéliser** ces relations via des approches statistiques et de machine learning pour formuler des recommandations concrètes aux établissements.

---

## 🎯 Objectifs analytiques

| Objectif | Approche |
|---|---|
| Calculer les métriques de satisfaction | Statistiques descriptives (Python/Pandas) |
| Identifier les services les moins performants | Bar charts, ANOVA (R) |
| Mesurer l'impact de la dépendance sur la satisfaction | Régression, corrélations |
| Prédire la satisfaction globale | Random Forest, régression linéaire (scikit-learn) |
| Valider les résultats statistiquement | Tests de Kruskal-Wallis, Chi² (R) |

---

## 📁 Structure du dépôt

```
retraite-satisfaction/
│
├── README.md                        # Ce fichier
├── requirements.txt                 # Dépendances Python
├── environment.R                    # Dépendances R
│
├── data/
│   ├── raw/
│   │   └── Retraite.csv             # Jeu de données brut (non modifié)
│   └── processed/
│       ├── retraite_clean.csv       # Données nettoyées
│       └── retraite_encoded.csv     # Données encodées pour le ML
│
├── notebooks/
│   ├── 01_exploration.ipynb         # Chargement, EDA, statistiques descriptives
│   ├── 02_metriques.ipynb           # Calcul des KPIs et métriques clés
│   ├── 03_visualisations.ipynb      # Histogrammes, boxplots, courbes d'évolution
│   └── 04_modelisation.ipynb        # Régression, Random Forest, validation croisée
│
├── src/
│   ├── preprocessing.py             # Nettoyage et encodage des données
│   ├── metrics.py                   # Calcul des métriques de satisfaction
│   └── models.py                    # Entraînement et évaluation des modèles ML
│
├── R/
│   ├── analyse_statistique.R        # ANOVA, Chi², Kruskal-Wallis, régression logistique
│   └── visualisations_ggplot.R      # Graphiques publication-ready avec ggplot2
│
└── reports/
    ├── rapport_final.pdf            # Rapport synthétique (généré depuis R/Python)
    └── figures/                     # Exports des visualisations (PNG, SVG)
        ├── satisfaction_globale.png
        ├── satisfaction_par_service.png
        ├── evolution_satisfaction.png
        ├── boxplot_dependance.png
        ├── heatmap_correlations.png
        └── feature_importance.png
```

---

## 📊 Jeu de données — `Retraite`

**Thème** : Satisfaction des résidents en maison de retraite

### Variables principales

| Variable | Type | Description |
|---|---|---|
| `age` | Numérique | Âge du résident (années) |
| `dependance` | Catégorielle/Ordinale | Niveau de dépendance (GIR 1 à 6) |
| `duree_sejour` | Numérique | Durée de séjour dans l'établissement (mois) |
| `service` | Catégorielle | Service d'affectation (soins, animation, restauration…) |
| `participation_activites` | Numérique/Binaire | Taux ou fréquence de participation aux activités |
| `satisfaction_globale` | Numérique (1–5) | **Variable cible** — score de satisfaction |

> ⚠️ Le jeu de données brut (`data/raw/Retraite.csv`) n'est jamais modifié. Toutes les transformations sont appliquées en mémoire ou exportées dans `data/processed/`.

---

## 📐 Méthodologie

### Phase 1 — Exploration & Nettoyage
- Chargement avec `pandas`, inspection des types et valeurs manquantes
- Imputation : médiane pour les variables numériques, mode pour les catégorielles
- Détection des outliers via la méthode IQR et boxplots
- Encodage : `LabelEncoder` pour les variables ordinales, `get_dummies` pour les nominales

### Phase 2 — Métriques clés
- **Taux de satisfaction global** : moyenne pondérée du score sur l'ensemble des résidents
- **Satisfaction par service** : score moyen groupé par service (`groupby`)
- **Taux de participation aux activités** : proportion de résidents actifs par tranche d'âge
- **NPS interne** : score ≥ 4 = promoteur, score ≤ 2 = détracteur

### Phase 3 — Visualisations
| Type | Variables | Outil |
|---|---|---|
| Histogramme | Âge, durée de séjour | Matplotlib |
| Bar chart | Satisfaction par service | Matplotlib / ggplot2 |
| Boxplot | Satisfaction selon le niveau de dépendance | Seaborn / ggplot2 |
| Courbe d'évolution | Satisfaction au fil du temps | Matplotlib |
| Heatmap | Matrice de corrélations | Seaborn |

### Phase 4 — Modélisation (scikit-learn)
```
Variables prédictives (X) : age, dependance, duree_sejour, service, participation_activites
Variable cible       (y) : satisfaction_globale
```

1. **Régression linéaire multiple** — baseline, coefficients interprétables
2. **Random Forest Regressor** — capture des non-linéarités, importance des features
3. **Validation croisée** — k-fold (k=5), métriques : MAE, RMSE, R²
4. **GridSearchCV** — optimisation des hyperparamètres du Random Forest

### Phase 5 — Analyse statistique (R)
- **ANOVA à un facteur** : différence de satisfaction entre services
- **Test de Kruskal-Wallis** : alternative non-paramétrique (si normalité rejetée)
- **Test du Chi²** : lien entre niveau de dépendance et satisfaction catégorisée
- **Régression logistique** : satisfaction binaire (haute ≥ 4 vs basse < 4)

---

## ⚙️ Installation & Reproduction

### Prérequis Python

```bash
# Cloner le dépôt
git clone https://github.com/mhayette/retraite-satisfaction.git
cd retraite-satisfaction

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Prérequis R

```r
# Dans la console R
source("environment.R")
```

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

Exécuter dans l'ordre : `01` → `02` → `03` → `04`

---

## 📦 Dépendances Python (`requirements.txt`)

```
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
jupyter>=1.0
openpyxl>=3.1
```

---

## 📈 Résultats attendus

### Métriques de performance (modélisation)

| Modèle | R² (CV) | MAE | RMSE |
|---|---|---|---|
| Régression linéaire | ~0.xx | ~0.xx | ~0.xx |
| Random Forest | ~0.xx | ~0.xx | ~0.xx |

> Les valeurs seront renseignées après exécution complète du pipeline.

### Facteurs déterminants (hypothèses)
Les analyses préliminaires suggèrent que les facteurs suivants pourraient le plus influencer la satisfaction :
1. La qualité du service (soins, restauration)
2. Le niveau de dépendance (GIR)
3. La participation aux activités

---

## 🗓️ Feuille de route

- [x] Définition du sujet et de la méthodologie
- [ ] Collecte et chargement des données
- [ ] Notebook 01 — Exploration
- [ ] Notebook 02 — Métriques
- [ ] Notebook 03 — Visualisations
- [ ] Notebook 04 — Modélisation
- [ ] Scripts R
- [ ] Rapport final PDF

---

## 👥 Auteurs

| Nom | Rôle |
|---|---|
| *À compléter* | Analyse de données, modélisation Python |
| *À compléter* | Analyse statistique R, rapport |

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

---

## 📬 Contact

Pour toute question relative au projet, ouvrir une [issue GitHub](https://github.com/<votre-username>/retraite-satisfaction/issues).
