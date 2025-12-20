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