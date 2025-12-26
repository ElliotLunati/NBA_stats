"""
Module de nettoyage et d'analyse du dataset NBA.

Ce module contient des fonctions utilitaires pour pour des 
représentations graphiques
"""



#################################
##        IMPORTATIONS         ##
#################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy import interp
import seaborn as sns

df = pd.read_csv('./data/merged_data.csv')



#################################
##  Statistiques Descriptives  ##
#################################



def plot_salary_quartiles_evolution(data, season_start='1999-00'):
    """
    Trace l'évolution du salaire moyen ajusté par tranche de 25%.
    
    Cette fonction affiche un graphique montrant l'évolution des salaires
    ajustés pour chaque quartile (0-25%, 25-50%, 50-75%, 75-100%)
    ainsi que la moyenne totale, normalisée avec une base 100 en 2005-06.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame contenant les colonnes 'Season' et 'adjusted_salary'.
        Par défaut, utilise le DataFrame global df.
    
    Returns
    -------
    None
        Affiche le graphique avec matplotlib.
    """
    # Filtrer les données à partir de la saison choisie
    df_filtered = data[data['Season'] >= season_start].copy()
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    # Tracer l'évolution de chaque tranche normalisée à sa valeur initiale
    quartile_stats = []
    seasons = df_filtered['Season'].unique()
    
    for season in sorted(seasons):
        season_data = df_filtered[
            df_filtered['Season'] == season
        ]['adjusted_salary'].sort_values()
        n = len(season_data)
        
        # Calculer la moyenne de chaque tranche de 25%
        q1_mean = (season_data.iloc[:n//4].mean()
                   if n >= 4 else season_data.mean())
        q2_mean = (season_data.iloc[n//4:n//2].mean()
                   if n >= 4 else season_data.mean())
        q3_mean = (season_data.iloc[n//2:3*n//4].mean()
                   if n >= 4 else season_data.mean())
        q4_mean = (season_data.iloc[3*n//4:].mean()
                   if n >= 4 else season_data.mean())
        
        quartile_stats.append({
            'Season': season,
            'Q1': q1_mean,
            'Q2': q2_mean,
            'Q3': q3_mean,
            'Q4': q4_mean
        })
    
    quartile_df = pd.DataFrame(quartile_stats)
    
    # Normaliser chaque tranche par rapport à sa première valeur
    q1_base = quartile_df['Q1'].iloc[0]
    q2_base = quartile_df['Q2'].iloc[0]
    q3_base = quartile_df['Q3'].iloc[0]
    q4_base = quartile_df['Q4'].iloc[0]
    
    axes.plot(quartile_df['Season'], (quartile_df['Q1'] / q1_base * 100),
              label='Moyenne tranche (0-25%)', linewidth=2)
    axes.plot(quartile_df['Season'], (quartile_df['Q2'] / q2_base * 100),
              label='Moyenne tranche (25-50%)', linewidth=2)
    axes.plot(quartile_df['Season'], (quartile_df['Q3'] / q3_base * 100),
              label='Moyenne tranche (50-75%)', linewidth=2)
    axes.plot(quartile_df['Season'], (quartile_df['Q4'] / q4_base * 100),
              label='Moyenne tranche (75-100%)', linewidth=2)
    axes.axhline(y=100, linewidth=2, linestyle='--',
                 color='black', alpha=0.5)
    
    # Calculer la moyenne totale pour chaque saison
    mean_by_season = df_filtered.groupby('Season')['adjusted_salary'].mean()
    mean_base = mean_by_season.iloc[0]
    axes.plot(quartile_df['Season'], (mean_by_season / mean_base * 100),
              label='Moyenne totale', linewidth=2, linestyle='--')
    
    axes.set_title(
        f'Evolution du salaire moyen ajusté par tranche de 25% '
        f'(Base 100 en {season_start})'
    )
    axes.set_xlabel('Saison')
    axes.set_ylabel('Indice (Base 100)')
    axes.tick_params(axis='x', rotation=45, labelsize=8)
    axes.grid(True, alpha=0.3)
    axes.legend()
    plt.tight_layout()
    plt.show()


def plot_stat(stat_name, placement, data = df):
    """
    Trace un graphique avec la valeur max, min, moyenne et médiane
    d'une variable 'stat_name'.
    """
    placement.set_title(f'{stat_name}/game evolution')
    placement.plot(
        data.groupby('Season')[stat_name].max(),
        c='red',
        linewidth=1,
        label='Max'
    )
    placement.plot(
        data.groupby('Season')[stat_name].mean(),
        c='blue',
        linewidth=1,
        label='Mean'
    )
    placement.plot(
        data.groupby('Season')[stat_name].median(),
        c='green',
        linewidth=1,
        label='Median'
    )
    placement.plot(
        data.groupby('Season')[stat_name].min(),
        c='yellow',
        linewidth=1,
        label='Min'
    )
    placement.grid(True, alpha=0.3)
    placement.tick_params(axis='x', rotation=45, labelsize='8')


def plot_major_stats_evolution(data=df):
    """
    Trace l'évolution des statistiques majeures NBA (2000-2025).
    
    Cette fonction affiche 6 graphiques montrant l'évolution des valeurs
    maximales, moyennes, médianes et minimales pour les statistiques
    suivantes : PTS, REB, AST, STL, BLK et total_minutes.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame contenant les colonnes de statistiques NBA.
        Par défaut, utilise le DataFrame global df.
    
    Returns
    -------
    None
        Affiche le graphique avec matplotlib.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Evolution des statistiques majeures (2000-2025)',
        fontsize=20
    )
    
    labels = ['Max', 'Mean', 'Median', 'Min']
    
    plot_stat('PTS', axes[0, 0], data)
    plot_stat('REB', axes[0, 1], data)
    plot_stat('AST', axes[0, 2], data)
    plot_stat('STL', axes[1, 0], data)
    plot_stat('BLK', axes[1, 1], data)
    plot_stat('total_minutes', axes[1, 2], data)
    axes[1, 2].set_title('Total Minutes Played evolution')
    
    fig.legend(
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.94),
        ncol=4
    )
    plt.show()


def stat_norm(stat_name, data, season_start='1999-00'):
    """
    Retourne la moyenne normalisée par rapport à la moyenne de l'année de départ
    """
    df_filtered = data[data['Season'] >= season_start].copy()
    mean_stat_00 = df_filtered[df_filtered['Season'] == season_start][stat_name].mean()
    mean_stat_season = df_filtered.groupby('Season')[stat_name].mean()
    return mean_stat_season / mean_stat_00


def plot_major_stats_with_salary(data, season_start='1999-00'):
    """
    Trace l'évolution des statistiques majeures NBA avec le salaire ajusté.
    """

    plt.figure(figsize=(8, 6))
    plt.suptitle(f'Evolution des statistiques majeures et du salaire {season_start}', fontsize=15)

    plt.plot(stat_norm('adjusted_salary', data, season_start),linewidth = 5, label= 'Salary')
    plt.plot(stat_norm('PTS', data, season_start),linewidth = 5, label= 'PTS')

    plt.plot(stat_norm('REB', data, season_start),linewidth = 1, label= 'REB')
    plt.plot(stat_norm('AST', data, season_start),linewidth = 1, label= 'AST')
    plt.plot(stat_norm('STL', data, season_start),linewidth = 1, label= 'STL')
    plt.plot(stat_norm('BLK', data, season_start),linewidth = 1, label= 'BLK')

    plt.plot(stat_norm('NBA_FANTASY_PTS', data, season_start),linewidth = 3, label= 'FANTASY_PTS')
    plt.axhline(y=1,linestyle = '--')

    plt.grid(True, alpha=0.3)
    plt.title(f"Normalisée par rapport à la saison {season_start}")
    plt.legend( loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=4)
    plt.tick_params(axis='x', rotation=45, labelsize='8')
    plt.show()


def plot_desc_salary(x, x_name, y='adjusted_salary', y_name='Salaire Ajusté ($)',
                     data = df):
    """
    Trace deux graphiques pour visualiser y par rapport à x :
    1. Nuage de points avec ligne de tendance.
    2. Moyenne et médiane de y par x.
    """
    x_name = str(x_name)
    plt.figure(figsize=(16, 8))

    if x in data.columns and y in data.columns:
        data_x_salary = data[(data[y].notna()) &
                         (data[y] > 0) &
                         (data[x].notna())]

        # Graphique 1 : Scatter plot avec tendance
        plt.subplot(1, 2, 1)
        plt.scatter(data_x_salary[x], data_x_salary[y],
                    alpha=0.3, s=20, color='steelblue')
        z = np.polyfit(data_x_salary[x], data_x_salary[y], 2)
        p = np.poly1d(z)
        x_range = np.linspace(data_x_salary[x].min(),
                              data_x_salary[x].max(), 100)
        plt.plot(x_range, p(x_range), "r--", linewidth=2, label='Tendance')

        plt.title(f'Salaire ajusté en fonction de l\'{x_name} '
                  f'(tous les joueurs)', fontsize=14, fontweight='bold')
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Graphique 2 : Salaire moyen et médian par x
        plt.subplot(1, 2, 2)
        salary_by_x = data_x_salary.groupby(x)['adjusted_salary'] \
            .agg(['mean', 'median', 'count']).reset_index()

        plt.plot(salary_by_x[x], salary_by_x['mean'],
                 marker='o', linewidth=2, markersize=6,
                 label='Moyenne', color='steelblue')
        plt.plot(salary_by_x[x], salary_by_x['median'],
                 marker='s', linewidth=2, markersize=6,
                 label='Médiane', color='coral')

        plt.title(f'Salaire ajusté moyen et médian par {x_name}',
                  fontsize=14, fontweight='bold')
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        print(f"Les colonnes {x} ou {y} n'existent pas dans le dataset.")
        print(f"Colonnes disponibles : {data.columns.tolist()}")


def plot_desc_salary_cont(x, x_name, nb_inter, y='adjusted_salary',
                          y_name='Salaire Ajusté ($)', data = df):
    """
    Trace deux graphiques pour visualiser y par rapport à x continus :
    1. Nuage de points avec ligne de tendance.
    2. Moyenne et médiane de y par intervalles de x.
    """
    plt.figure(figsize=(16, 8))

    if x in data.columns and y in data.columns:
        data_x_salary = data[(data[y].notna()) &
                         (data[y] > 0) &
                         (data[x].notna())]

        # Graphique 1 : Scatter plot avec tendance
        plt.subplot(1, 2, 1)
        plt.scatter(data_x_salary[x], data_x_salary[y],
                    alpha=0.3, s=20, color='steelblue')
        z = np.polyfit(data_x_salary[x], data_x_salary[y], 2)
        p = np.poly1d(z)
        x_range = np.linspace(data_x_salary[x].min(),
                              data_x_salary[x].max(), 100)
        plt.plot(x_range, p(x_range), "r--", linewidth=2, label='Tendance')

        plt.title(f'Salaire ajusté en fonction des {x_name} '
                  f'(tous les joueurs)', fontsize=14, fontweight='bold')
        plt.xlabel(f"{x}/game", fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Graphique 2 : Moyenne et médiane par intervalle
        plt.subplot(1, 2, 2)
        data_x_salary['x_bin'] = pd.cut(data_x_salary[x], bins=nb_inter)

        salary_by_bin = (
            data_x_salary.groupby('x_bin', observed=True)['adjusted_salary']
            .agg(['mean', 'median', 'count'])
            .reset_index()
        )

        salary_by_bin['x_center'] = salary_by_bin['x_bin'].apply(lambda b: b.mid)

        plt.plot(salary_by_bin['x_center'], salary_by_bin['mean'],
                 marker='o', linewidth=2, markersize=6, label='Moyenne')
        plt.plot(salary_by_bin['x_center'], salary_by_bin['median'],
                 marker='s', linewidth=2, markersize=6, label='Médiane')

        plt.title(f"Salaire ajusté moyen et médian par {x_name}",
                  fontsize=14, fontweight='bold')
        plt.xlabel(f"{x}/game", fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

    else:
        print(f"Les colonnes {x} ou {y} n'existent pas dans le dataset.")
        print(f"Colonnes disponibles : {data.columns.tolist()}")


def plot_salary_by_position(df):
    """
    Trace des graphiques pour visualiser le salaire moyen par position.
    """

    # Graphique du salaire moyen par position
    plt.figure(figsize=(14, 8))

    # Vérifier si la colonne Position existe
    if 'Position' in df.columns:
        # Filtrer les données avec salaire non nul et exclure la position 'GF'
        df_with_salary = df[(df['adjusted_salary'].notna()) & 
                                    (df['adjusted_salary'] > 0) & 
                                    (df['Position'] != 'GF')]
        
        # Calculer le salaire moyen par position
        salary_by_position = df_with_salary.groupby('Position')['adjusted_salary'].agg(['mean', 'median', 'count']).reset_index()
        salary_by_position = salary_by_position.sort_values('mean', ascending=False)
        
        # Créer un boxplot pour montrer la distribution
        plt.subplot(1, 2, 1)
        positions_order = salary_by_position['Position'].tolist()
        sns.boxplot(data=df_with_salary, y='Position', x='adjusted_salary', order=positions_order, palette='Set2')
        plt.title('Distribution des salaires par position', fontsize=14, fontweight='bold')
        plt.xlabel('adjusted_salary ($)', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Créer un graphique en barres pour le salaire moyen
        plt.subplot(1, 2, 2)
        bars = plt.barh(salary_by_position['Position'], salary_by_position['mean'], color='steelblue', alpha=0.7)
        plt.title('Salaire moyen par position', fontsize=14, fontweight='bold')
        plt.xlabel('adjusted_salary moyen ($)', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Ajouter les valeurs sur les barres
        for i, (pos, val, count) in enumerate(zip(salary_by_position['Position'], 
                                                    salary_by_position['mean'], 
                                                    salary_by_position['count'])):
            plt.text(val, i, f' ${val/1e6:.1f}M (n={count})', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher le tableau récapitulatif
        print("\nRésumé des salaires par position:")
        print("="*80)
        salary_by_position['mean_millions'] = salary_by_position['mean'] / 1e6
        salary_by_position['median_millions'] = salary_by_position['median'] / 1e6
        print(salary_by_position[['Position', 'mean_millions', 'median_millions', 'count']].to_string(index=False))
        
    else:
        print("La colonne 'Position' n'existe pas dans le dataset.")
        print(f"Colonnes disponibles: {df.columns.tolist()}")


def plot_salary_by_team_over_time(df, season_start='1999-00'):
    """
    Trace l'évolution du salaire moyen par équipe au fil des saisons.
    """

    df_filtered = df[df['Season'] >= season_start].copy()

    # Calcul du salaire moyen par équipe par an
    salary_by_team_year = df_filtered.groupby(['Season', 'Team'])['Salary'].mean().reset_index()
    salary_by_team_year.columns = ['Season', 'Team', 'Average_Salary']

    # Calcul du salaire moyen global par saison (toutes équipes confondues)
    salary_avg_global = df_filtered.groupby('Season')['Salary'].mean().reset_index()
    salary_avg_global.columns = ['Season', 'Average_Salary']

    # Création du graphique
    plt.figure(figsize=(16, 8))

    # Tracer une ligne pour chaque équipe
    for team in salary_by_team_year['Team'].unique():
        team_data = salary_by_team_year[salary_by_team_year['Team'] == team]
        plt.plot(team_data['Season'], team_data['Average_Salary'], label=team, alpha=0.7, linewidth=1)

    # Tracer la moyenne globale en mise en valeur
    plt.plot(salary_avg_global['Season'], salary_avg_global['Average_Salary'], 
            color='red', linewidth=3, label='Moyenne NBA', linestyle='--', zorder=10)

    plt.title('Salaire moyen par équipe au fil des saisons', fontsize=16, fontweight='bold')
    plt.xlabel('Saison', fontsize=12)
    plt.ylabel('Salaire moyen ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




#################################
##    Regressions linéaires    ##
#################################



def plot_rls(
    x,
    y_pred,
    reg_func,
    placement,
    y='adjusted_salary',
    color='blue',
    data=df
):
    """
    x : Variable explicative
    y : Variable expliquée
    y_pred : Prédiction de la variable expliquée
    placement : Localisation de la fenêtre
    reg_func : Fonction de régression utilisée

    Nuage de points du salaire réel et du salaire prédit
    par régression linéaire simple en fonction de x.
    """
    placement.scatter(
        data[x],
        data[y],
        color=color,
        alpha=0.5
    )
    placement.plot(
        data[x],
        data[y_pred],
        c='red',
        lw=2
    )
    placement.set_xlabel(f'{x}/match')
    placement.set_ylabel('Salaire ($)')
    placement.set_title(
        f'Prédiction Salaire ~ {x} '
        f'(R² = {reg_func.rsquared:.3f})'
    )
    placement.ticklabel_format(style='plain', axis='both')


def plot_error(
    y_pred,
    placement,
    y='adjusted_salary',
    color='blue',
    c_line='red',
    nom_reg='',
    data=df
):
    """
    y : Variable expliquée
    y_pred : Prédiction de la variable expliquée
    placement : Localisation de la fenêtre

    Diagramme de la distribution des erreurs de prédiction du modèle.
    """
    errors_func = data[y] - data[y_pred]

    placement.hist(
        errors_func,
        bins=50,
        color=color,
        edgecolor='black',
        density=True
    )
    placement.set_xlabel('Erreur de prédiction ($)')
    placement.set_ylabel('Fréquence')
    placement.set_title('Distribution des erreurs de prédiction')
    placement.axvline(
        x=0,
        color=c_line,
        linestyle='--',
        linewidth=2
    )
    placement.ticklabel_format(style='plain', axis='x')
    placement.tick_params(axis='x', rotation=45, labelsize=8)

    print(f"\nStatistiques des erreurs ({nom_reg}) :")
    print(f"   Erreur moyenne : {errors_func.mean():,.0f}")
    print(f"   Écart-type des erreurs : {errors_func.std():,.0f}")



def plot_lowess(
    x,
    y_pred,
    reg_mco,
    y='adjusted_salary',
    color='blue',
    data=df
):
    """
    x : variable explicative
    y : variable expliquée
    y_pred : prédiction par le modèle MCO
    reg_mco : objet regression linéaire utilisé pour RLS
    data : DataFrame contenant x, y et y_pred

    Graph 1 : Nuage de points de y par rapport à x, avec tendances MCO et LOWESS.
    Graph 2 : Distribution des erreurs du modèle MCO.
    Graph 3 : Distribution des erreurs du modèle LOWESS.
    """

    # Calcul LOWESS
    lowess = sm.nonparametric.lowess
    y_lowess = lowess(data[y], data[x], frac=0.3)
    x_smooth = y_lowess[:, 0]
    y_smooth = y_lowess[:, 1]

    # Interpolation pour obtenir y_predicted_lowess
    data['y_predicted_lowess'] = interp(data[x], x_smooth, y_smooth)

    # Création des sous-graphes
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Graphique 1: Regression Y ~ X
    plot_rls(x, y_pred, reg_mco, axes[0], y=y, color=color, data=data)
    axes[0].plot(x_smooth, y_smooth, c='lime', lw=2, label='Courbe LOWESS')
    axes[0].axvline(
        x=np.median(data[x]),
        color='black',
        linestyle='--',
        linewidth=2,
        label='Médiane'
    )
    axes[0].axvline(
        x=np.percentile(data[x], 90),
        color='black',
        linestyle='--',
        linewidth=2,
        label='Centile 90'
    )
    axes[0].legend()

    # Graphique 2: Distribution des erreurs (MCO)
    plot_error(
        y_pred,
        axes[1],
        y=y,
        color=color,
        nom_reg=f"Salaire ~ {x} (MCO)",
        data=data
    )
    axes[1].set_title('Distribution des erreurs de prédiction (MCO)')

    # Graphique 3: Distribution des erreurs (LOWESS)
    plot_error(
        'y_predicted_lowess',
        axes[2],
        y=y,
        color=color,
        c_line='lime',
        nom_reg=f"Salaire ~ {x} (LOWESS)",
        data=data
    )
    axes[2].set_title('Distribution des erreurs de prédiction (LOWESS)')

    plt.tight_layout()
    plt.show()


def plot_rlm(
    y_pred,
    placement,
    y='adjusted_salary',
    color='blue',
    data=df
):
    """
    y : Variable expliquée
    y_pred : Prédiction de la variable expliquée par la RLM
    placement : Localisation de la fenêtre

    Nuage de points du salaire prédit par rapport au salaire réel.
    """
    placement.scatter(
        data[y],
        data[y_pred],
        alpha=0.5,
        color=color
    )

    min_salary = data[y].min()
    max_salary = data[y].max()

    placement.plot(
        [min_salary, max_salary],
        [min_salary, max_salary],
        'r--',
        lw=2,
        label='Prédiction parfaite'
    )

    placement.set_xlabel('Salaire réel ($)')
    placement.set_ylabel('Salaire prédit ($)')
    placement.set_title('Prédictions vs valeurs réelles')
    placement.ticklabel_format(style='plain', axis='both')