# NBA Stats - Analyse des déterminants du salaire en NBA

*Auteurs : LUNATI Elliot et POULLAIN Gianni*

Ce projet analyse les **déterminants du salaire des joueurs NBA sur les 25 dernières années** en combinant collecte de données, analyse statistique et modélisation prédictive par Machine Learning.

## Problématique

**Quels sont les déterminants du salaire en NBA sur les 25 dernières années ?**

**Les salaires en NBA sont régulièrement critiqués**, jugés tantôt trop élevés : https://fanarch.com/blogs/nba/who-are-the-most-overpaid-nba-players-in-2025

tantôt insuffisants selon les joueurs : 
https://www.complex.com/sports/a/jose-martinez/steph-curry-thinks-nba-players-are-underpaid

Ce projet vise à identifier les déterminants réels de ces rémunérations en analysant l'impact des performances sur le terrain sur l'évolution des salaires au cours de la carrière des joueurs.


Pour répondre à cette problématique, nous avons collecté les statistiques de 10 000+ joueurs sur 25 saisons via l'API officielle NBA et du scraping. Afin de trouver les déterminants de l'évolution du salaire en NBA, nous avons conduit une analyse descriptive des données, et ensuite avons appliqué des techniques statistiques (régressions linéaires, LOWESS) et de Machine Learning (XGBoost, LightGBM, RandomForest) .

### Résultats clés

L'analyse démontre que **les salaires sont globalement en adéquation avec les performances mesurables** :
- La relation salaire-performance n'est pas linéaire : l'effet marginal d'un point supplémentaire s'amplifie exponentiellement pour les superstars (>20 pts/match)
- L'expérience des joueurs et le temps de jeu jouent un rôle important dans l'évolution des salaires, parfois plus que les performances en fonction des modèles
- Les performances offensives semblent avoir plus d'impact sur le salaire que les performances défensives
- Notre meilleur modèle atteint un **R² de 0.85** pour prédire le salaire de l'année suivante à partir des performances de l'année n-1.


## Installation

### Prérequis
- Python 3.7 ou supérieur
- Google Chrome (pour le scraping avec Selenium)

### Étapes

1. **Cloner le projet**
```bash
git clone https://github.com/ElliotLunati/NBA_stats.git
cd NBA_stats
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Fonctionnalités principales

### 1. Collecte automatique de données
- **Stats des joueurs** : De 1996-97 à 2024-25 (Regular Season)
- **Stats des équipes** : De 1996-97 à 2024-25 (Regular Season)
- **Salaires** : De 1999-00 à 2024-25 (scraping ESPN avec Selenium)
- **Historique MVP** : Depuis 1956 (scraping Basketball Reference)

### 2. Nettoyage et preprocessing
- Suppression des variables hautement corrélées (>80%)
- Élimination des outliers (salaires anormaux, fins de carrière)
- Encodage des variables catégorielles
- Normalisation des variables numériques
- Gestion des valeurs manquantes

### 3. Analyse statistique
- **Statistiques descriptives** : Évolution des performances et salaires par saison
- **Analyse de corrélations** : Identification des variables fortement corrélées
- **Régressions linéaires simples et multiples** : Salaire ~ PTS, REB, AST, STL, BLK
- **Régressions locales (LOWESS)** : Modélisation non-linéaire des relations

### 4. Modélisation prédictive
- **Modèle Blended** combinant :
  - XGBoost
  - LightGBM
  - RandomForestRegressor
- **Prédiction du salaire année n+1** à partir des performances année n
- **Feature importance** : Identification des métriques les plus déterminantes
- **Performance** : R² ~ 0.85 avec salaire précédent, R² ~ 0.75 sans salaire précédent

## Structure du projet

```
NBA_stats/
├── main.ipynb                    # Notebook principal avec toute l'analyse
├── src/                          # Dossier contenant toutes nos fonctions
│   ├── build_nba_bdd.py               # Construction de la base de données
│   ├── clean_dataset_fun.py           # Nettoyage et preprocessing
│   ├── train_blended_model.py         # Entraînement du modèle blendé
│   └── plot_func.py                   # Fonctions de visualisation
├── requirements.txt               # Dépendances Python
├── data/                          # Données collectées par saison
│   ├── merged_data.csv           # Dataset final fusionné
│   ├── 1996-97/ à 2024-25/       # Données par saison
│   │   ├── Regular_Season/
│   │   ├── Playoffs/
│   │   └── Salaries/
│   └── MVP/
│       └── nba_mvp_history.csv
└── README.md
```