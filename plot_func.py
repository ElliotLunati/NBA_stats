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

df = pd.read_csv('./data/merged_data.csv')

#################################
##  Statistiques Descriptives  ##
#################################


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


def stat_norm(stat_name, data =df):
    """
    Retourne la moyenne normalisée par rapport à la moyenne en 1999-00
    """
    mean_stat_00 = data[data['Season'] == '1999-00'][stat_name].mean()
    mean_stat_season = data.groupby('Season')[stat_name].mean()
    return mean_stat_season / mean_stat_00


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
