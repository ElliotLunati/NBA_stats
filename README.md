# NBA Stats - Collecte de données NBA

Ce projet permet de collecter automatiquement les statistiques NBA (joueurs, équipes, salaires et MVP) depuis plusieurs sources.

## Prérequis

- Python 3.7 ou supérieur
- Google Chrome (pour le scraping avec Selenium)

## Installation

1. **Cloner le projet**
```bash
git clone https://github.com/ElliotLunati/NBA_stats.git
cd NBA_stats
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Utilisation

### Avec le notebook Jupyter

Ouvrez `main.ipynb` et exécutez les cellules pour :

1. **Collecter les stats des joueurs** (Regular Season + Playoffs)
```python
export_all_seasons()
```

2. **Collecter les stats des équipes**
```python
export_all_teams_seasons()
```

3. **Collecter les salaires** (depuis ESPN)
```python
export_all_salaries()
```

4. **Collecter l'historique des MVP** (depuis Basketball Reference)
```python
scrape_mvp_data()
```

## Structure des données

Les données sont organisées par saison dans le dossier `data/` :

```
data/
├── 1996-97/
│   ├── Regular_Season/
│   │   ├── nba_players_stats_1996-97_Regular_Season.csv
│   │   └── nba_teams_stats_1996-97_Regular_Season.csv
│   ├── Playoffs/
│   │   ├── nba_players_stats_1996-97_Playoffs.csv
│   │   └── nba_teams_stats_1996-97_Playoffs.csv
│   └── Salaries/
│       └── nba_salaries_1996-97.csv
├── ...
└── MVP/
    └── nba_mvp_history.csv
```

## Fonctionnalités

- **Stats des joueurs** : De 1996-97 à 2024-25 (Regular Season + Playoffs)
- **Stats des équipes** : De 1996-97 à 2024-25 (Regular Season + Playoffs)
- **Salaires** : De 1999-00 à 2024-25 (source : ESPN)
- **MVP** : Historique complet depuis 1956 (source : Basketball Reference)

## Dépendances

- `nba-api` : API officielle NBA
- `selenium` : Automatisation du navigateur pour le scraping
- `beautifulsoup4` : Parsing HTML
- `requests` : Requêtes HTTP
- `pandas` : Manipulation des données

## Licence

Ce projet est à usage éducatif et de recherche.