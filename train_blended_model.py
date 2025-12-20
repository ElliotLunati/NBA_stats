"""
Ce module contient des fonctions pour calculer et afficher les 
performances d'un modèle blendé combinant XGBoost, LightGBM et RandomForest
pour prédire le salaire ajusté des joueurs NBA l'année suivante en 
fonction des performances de l'année n-1.
"""



#################################
##        IMPORTATIONS         ##
#################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from itertools import product



#################################
##        ENTRAÎNEMENT         ##
#################################



def train_blended_model(
        data,
        target_column='next_adjusted_salary',
        feature_columns=None,
        test_size=0.3,
        random_state=42,
        xgb_weight=0.4,
        lgbm_weight=0.4,
        rf_weight=0.2,
        xgb_params=None,
        lgbm_params=None,
        rf_params=None):
    """
    Entraîne un modèle blendé combinant XGBoost, LightGBM et RandomForest.

    Args:
        data (pd.DataFrame): Le jeu de données complet
        target_column (str): Nom de la colonne cible (y)
        feature_columns (list): Liste des colonnes features (X).
                                Si None, utilise toutes les colonnes
                                sauf target_column
        test_size (float): Proportion du jeu de test (0 à 1)
        random_state (int): Seed pour la reproductibilité
        xgb_weight (float): Poids pour XGBoost dans le blend
        lgbm_weight (float): Poids pour LightGBM dans le blend
        rf_weight (float): Poids pour RandomForest dans le blend
        xgb_params (dict): Paramètres pour XGBoost
        lgbm_params (dict): Paramètres pour LightGBM
        rf_params (dict): Paramètres pour RandomForest

    Returns:
        dict: Dictionnaire contenant:
            - 'xgb_model': Le modèle XGBoost entraîné
            - 'lgbm_model': Le modèle LightGBM entraîné
            - 'rf_model': Le modèle RandomForest entraîné
            - 'X_train': Features d'entraînement
            - 'X_test': Features de test
            - 'y_train': Cible d'entraînement
            - 'y_test': Cible de test
            - 'train_predictions': Prédictions blendées sur train
            - 'test_predictions': Prédictions blendées sur test
            - 'metrics': Métriques de performance
            - 'individual_metrics': Métriques individuelles des modèles
    """
    # Vérifier que la colonne cible existe
    if target_column not in data.columns:
        raise ValueError(
            f"La colonne cible '{target_column}' n'existe pas"
        )

    # Sélectionner les features
    if feature_columns is None:
        feature_columns = [
            col for col in data.columns if col != target_column
        ]
    else:
        # Vérifier que toutes les features existent
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes: {missing_cols}"
            )

    # Paramètres par défaut
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': random_state
        }
    
    if lgbm_params is None:
        lgbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'random_state': random_state,
            'verbose': -1
        }
    
    if rf_params is None:
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': random_state,
            'n_jobs': -1
        }

    # Extraire X et y
    X = data[feature_columns].copy()
    y = data[target_column].copy()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Taille du jeu d'entraînement: {len(X_train)}")
    print(f"Taille du jeu de test: {len(X_test)}")
    print(f"\nPoids du blend: XGBoost={xgb_weight:.2f}, "
          f"LightGBM={lgbm_weight:.2f}, RandomForest={rf_weight:.2f}")

    # ===== Entraînement XGBoost =====
    print(f"\nEntraînement du modèle XGBoost...")
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)

    # Prédictions XGBoost
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    # Métriques XGBoost
    xgb_train_r2 = r2_score(y_train, xgb_train_pred)
    xgb_train_rmse = mean_squared_error(y_train, xgb_train_pred) ** 0.5
    xgb_test_r2 = r2_score(y_test, xgb_test_pred)
    xgb_test_rmse = mean_squared_error(y_test, xgb_test_pred) ** 0.5

    # ===== Entraînement LightGBM =====
    print(f"Entraînement du modèle LightGBM...")
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train)

    # Prédictions LightGBM
    lgbm_train_pred = lgbm_model.predict(X_train)
    lgbm_test_pred = lgbm_model.predict(X_test)

    # Métriques LightGBM
    lgbm_train_r2 = r2_score(y_train, lgbm_train_pred)
    lgbm_train_rmse = mean_squared_error(y_train, lgbm_train_pred) ** 0.5
    lgbm_test_r2 = r2_score(y_test, lgbm_test_pred)
    lgbm_test_rmse = mean_squared_error(y_test, lgbm_test_pred) ** 0.5

    # ===== Entraînement RandomForest =====
    print(f"Entraînement du modèle RandomForest...")
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)

    # Prédictions RandomForest
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    # Métriques RandomForest
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_train_rmse = mean_squared_error(y_train, rf_train_pred) ** 0.5
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_test_rmse = mean_squared_error(y_test, rf_test_pred) ** 0.5

    # ===== Blending des prédictions =====
    print(f"\nBlending des prédictions...")
    y_train_pred = (xgb_weight * xgb_train_pred +
                    lgbm_weight * lgbm_train_pred +
                    rf_weight * rf_train_pred)
    y_test_pred = (xgb_weight * xgb_test_pred +
                   lgbm_weight * lgbm_test_pred +
                   rf_weight * rf_test_pred)

    # Métriques du modèle blendé
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Affichage formaté des résultats
    print("\n" + "=" * 80)
    print("RÉSULTATS DES MODÈLES INDIVIDUELS")
    print("=" * 80)
    print()
    print("XGBoost:")
    print(f"   Train R²: {xgb_train_r2:.4f}  |  Test R²: {xgb_test_r2:.4f}")
    print(f"   Train RMSE: ${xgb_train_rmse:,.2f}  |  "
          f"Test RMSE: ${xgb_test_rmse:,.2f}")
    print()
    print("LightGBM:")
    print(f"   Train R²: {lgbm_train_r2:.4f}  |  "
          f"Test R²: {lgbm_test_r2:.4f}")
    print(f"   Train RMSE: ${lgbm_train_rmse:,.2f}  |  "
          f"Test RMSE: ${lgbm_test_rmse:,.2f}")
    print()
    print("RandomForest:")
    print(f"   Train R²: {rf_train_r2:.4f}  |  Test R²: {rf_test_r2:.4f}")
    print(f"   Train RMSE: ${rf_train_rmse:,.2f}  |  "
          f"Test RMSE: ${rf_test_rmse:,.2f}")
    print()
    print("=" * 80)
    print("RÉSULTATS DU MODÈLE BLENDÉ")
    print("=" * 80)
    print()
    print("Performance sur l'ensemble d'ENTRAÎNEMENT:")
    print(f"   R² Score:  {train_r2:.4f}")
    print(f"   RMSE:      ${train_rmse:,.2f}")
    print(f"   MAE:       ${train_mae:,.2f}")
    print()
    print("Performance sur l'ensemble de TEST:")
    print(f"   R² Score:  {test_r2:.4f}")
    print(f"   RMSE:      ${test_rmse:,.2f}")
    print(f"   MAE:       ${test_mae:,.2f}")
    print("=" * 80)

    # Préparer les métriques
    metrics = {
        'train': {
            'r2_score': train_r2,
            'rmse': train_rmse,
            'mae': train_mae
        },
        'test': {
            'r2_score': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }
    }

    individual_metrics = {
        'xgb': {
            'train_r2': xgb_train_r2,
            'train_rmse': xgb_train_rmse,
            'test_r2': xgb_test_r2,
            'test_rmse': xgb_test_rmse
        },
        'lgbm': {
            'train_r2': lgbm_train_r2,
            'train_rmse': lgbm_train_rmse,
            'test_r2': lgbm_test_r2,
            'test_rmse': lgbm_test_rmse
        },
        'rf': {
            'train_r2': rf_train_r2,
            'train_rmse': rf_train_rmse,
            'test_r2': rf_test_r2,
            'test_rmse': rf_test_rmse
        }
    }

    # Retourner tous les résultats
    return {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'rf_model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'xgb_train_pred': xgb_train_pred,
        'xgb_test_pred': xgb_test_pred,
        'lgbm_train_pred': lgbm_train_pred,
        'lgbm_test_pred': lgbm_test_pred,
        'rf_train_pred': rf_train_pred,
        'rf_test_pred': rf_test_pred,
        'metrics': metrics,
        'individual_metrics': individual_metrics,
        'feature_columns': feature_columns,
        'weights': {
            'xgb': xgb_weight,
            'lgbm': lgbm_weight,
            'rf': rf_weight
        }
    }



#################################
##           PLOTS             ##
#################################



def get_feature_importance(results, top_n=5):
    """
    Affiche l'importance des features pour les trois modèles.

    Args:
        results (dict): Résultats retournés par train_blended_model
        top_n (int): Nombre de features les plus importantes à afficher

    Returns:
        dict: Dictionnaire avec les DataFrames d'importance pour
              chaque modèle
    """
    xgb_model = results['xgb_model']
    lgbm_model = results['lgbm_model']
    rf_model = results['rf_model']
    feature_columns = results['feature_columns']

    # Importance XGBoost
    xgb_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Importance LightGBM
    lgbm_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': lgbm_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Importance RandomForest
    rf_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Affichage
    print(f"\nTop {top_n} features - XGBoost:")
    print(xgb_importance_df.head(top_n).to_string(index=False))
    
    print(f"\n\nTop {top_n} features - LightGBM:")
    print(lgbm_importance_df.head(top_n).to_string(index=False))

    print(f"\n\nTop {top_n} features - RandomForest:")
    print(rf_importance_df.head(top_n).to_string(index=False))

    return {
        'xgb': xgb_importance_df,
        'lgbm': lgbm_importance_df,
        'rf': rf_importance_df
    }


def plot_predictions(results, figsize=(24, 6)):
    """
    Trace les valeurs prédites vs réelles pour les trois modèles et le blend.

    Args:
        results (dict): Résultats retournés par train_blended_model
        figsize (tuple): Taille de la figure (largeur, hauteur)

    Returns:
        tuple: (fig, axes) - Figure et axes matplotlib
    """
    # Extraire les données
    y_test = results['y_test']
    xgb_test_pred = results['xgb_test_pred']
    lgbm_test_pred = results['lgbm_test_pred']
    rf_test_pred = results['rf_test_pred']
    blend_test_pred = results['test_predictions']
    
    metrics = results['metrics']
    individual_metrics = results['individual_metrics']

    # Créer la figure avec 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Graphique pour XGBoost
    ax1 = axes[0]
    ax1.scatter(
        y_test,
        xgb_test_pred,
        alpha=0.5,
        s=20,
        color='blue',
        edgecolors='k',
        linewidths=0.5
    )
    min_val = min(y_test.min(), xgb_test_pred.min())
    max_val = max(y_test.max(), xgb_test_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Valeurs réelles', fontsize=12)
    ax1.set_ylabel('Valeurs prédites', fontsize=12)
    ax1.set_title('XGBoost - Test', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    xgb_r2 = individual_metrics['xgb']['test_r2']
    xgb_rmse = individual_metrics['xgb']['test_rmse']
    textstr = f'R² = {xgb_r2:.4f}\nRMSE = ${xgb_rmse:,.0f}'
    ax1.text(
        0.05, 0.95, textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )

    # Graphique pour LightGBM
    ax2 = axes[1]
    ax2.scatter(
        y_test,
        lgbm_test_pred,
        alpha=0.5,
        s=20,
        color='green',
        edgecolors='k',
        linewidths=0.5
    )
    min_val = min(y_test.min(), lgbm_test_pred.min())
    max_val = max(y_test.max(), lgbm_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('Valeurs réelles', fontsize=12)
    ax2.set_ylabel('Valeurs prédites', fontsize=12)
    ax2.set_title('LightGBM - Test', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    lgbm_r2 = individual_metrics['lgbm']['test_r2']
    lgbm_rmse = individual_metrics['lgbm']['test_rmse']
    textstr = f'R² = {lgbm_r2:.4f}\nRMSE = ${lgbm_rmse:,.0f}'
    ax2.text(
        0.05, 0.95, textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )

    # Graphique pour RandomForest
    ax3 = axes[2]
    ax3.scatter(
        y_test,
        rf_test_pred,
        alpha=0.5,
        s=20,
        color='orange',
        edgecolors='k',
        linewidths=0.5
    )
    min_val = min(y_test.min(), rf_test_pred.min())
    max_val = max(y_test.max(), rf_test_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax3.set_xlabel('Valeurs réelles', fontsize=12)
    ax3.set_ylabel('Valeurs prédites', fontsize=12)
    ax3.set_title('RandomForest - Test', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    rf_r2 = individual_metrics['rf']['test_r2']
    rf_rmse = individual_metrics['rf']['test_rmse']
    textstr = f'R² = {rf_r2:.4f}\nRMSE = ${rf_rmse:,.0f}'
    ax3.text(
        0.05, 0.95, textstr,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5)
    )

    # Graphique pour le Blend
    ax4 = axes[3]
    ax4.scatter(
        y_test,
        blend_test_pred,
        alpha=0.5,
        s=20,
        color='purple',
        edgecolors='k',
        linewidths=0.5
    )
    min_val = min(y_test.min(), blend_test_pred.min())
    max_val = max(y_test.max(), blend_test_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax4.set_xlabel('Valeurs réelles', fontsize=12)
    ax4.set_ylabel('Valeurs prédites', fontsize=12)
    ax4.set_title('Blended Model - Test', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    blend_r2 = metrics['test']['r2_score']
    blend_rmse = metrics['test']['rmse']
    textstr = f'R² = {blend_r2:.4f}\nRMSE = ${blend_rmse:,.0f}'
    ax4.text(
        0.05, 0.95, textstr,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5)
    )

    plt.tight_layout()
    plt.show()

    return fig, axes



#################################
##      MEILLEURS PARAMS       ##
#################################



def find_best_blend_weights(
        data,
        target_column='next_adjusted_salary',
        feature_columns=None,
        test_size=0.3,
        random_state=42,
        weight_step=0.1,
        xgb_params=None,
        lgbm_params=None,
        rf_params=None,
        metric='r2'):
    """
    Trouve la meilleure combinaison de poids pour le blend.

    Args:
        data (pd.DataFrame): Le jeu de données complet
        target_column (str): Nom de la colonne cible
        feature_columns (list): Liste des colonnes features
        test_size (float): Proportion du jeu de test
        random_state (int): Seed pour la reproductibilité
        weight_step (float): Pas de recherche des poids (ex: 0.1)
        xgb_params (dict): Paramètres pour XGBoost
        lgbm_params (dict): Paramètres pour LightGBM
        rf_params (dict): Paramètres pour RandomForest
        metric (str): Métrique à optimiser ('r2', 'rmse', ou 'mae')

    Returns:
        dict: Dictionnaire contenant:
            - 'best_weights': Meilleurs poids trouvés
            - 'best_score': Meilleur score
            - 'all_results': DataFrame avec tous les résultats
            - 'best_results': Résultats complets du meilleur modèle
    """
    # Génération des combinaisons de poids
    weights_range = np.arange(0, 1 + weight_step, weight_step)
    combinations = []
    
    for w1, w2, w3 in product(weights_range, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) < 0.01:  # Somme doit être ~1
            combinations.append((round(w1, 2), round(w2, 2), round(w3, 2)))
    
    print(f"Nombre de combinaisons à tester: {len(combinations)}")
    print(f"Métrique à optimiser: {metric.upper()}\n")
    
    results_list = []
    best_score = -np.inf if metric == 'r2' else np.inf
    best_weights = None
    best_full_results = None
    
    for i, (w_xgb, w_lgbm, w_rf) in enumerate(combinations, 1):
        if i % 50 == 0:
            print(f"Progression: {i}/{len(combinations)} combinaisons testées")
        
        # Entraîner le modèle avec ces poids
        results = train_blended_model(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
            xgb_weight=w_xgb,
            lgbm_weight=w_lgbm,
            rf_weight=w_rf,
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            rf_params=rf_params
        )
        
        # Extraire le score de test
        if metric == 'r2':
            score = results['metrics']['test']['r2_score']
            is_better = score > best_score
        elif metric == 'rmse':
            score = results['metrics']['test']['rmse']
            is_better = score < best_score
        elif metric == 'mae':
            score = results['metrics']['test']['mae']
            is_better = score < best_score
        else:
            raise ValueError(
                f"Métrique inconnue: {metric}. "
                "Utilisez 'r2', 'rmse' ou 'mae'."
            )
        
        # Sauvegarder les résultats
        results_list.append({
            'xgb_weight': w_xgb,
            'lgbm_weight': w_lgbm,
            'rf_weight': w_rf,
            'test_r2': results['metrics']['test']['r2_score'],
            'test_rmse': results['metrics']['test']['rmse'],
            'test_mae': results['metrics']['test']['mae']
        })
        
        # Mettre à jour le meilleur si nécessaire
        if is_better:
            best_score = score
            best_weights = (w_xgb, w_lgbm, w_rf)
            best_full_results = results
    
    # Créer un DataFrame avec tous les résultats
    results_df = pd.DataFrame(results_list)
    
    # Trier selon la métrique
    if metric == 'r2':
        results_df = results_df.sort_values('test_r2', ascending=False)
    elif metric == 'rmse':
        results_df = results_df.sort_values('test_rmse', ascending=True)
    elif metric == 'mae':
        results_df = results_df.sort_values('test_mae', ascending=True)
    
    results_df = results_df.reset_index(drop=True)
    
    # Affichage des résultats
    print("\n" + "=" * 80)
    print("RECHERCHE DE LA MEILLEURE COMBINAISON DE POIDS")
    print("=" * 80)
    print(f"\nMetrique optimisée: {metric.upper()}")
    print(f"\nMeilleurs poids trouvés:")
    print(f"   XGBoost:      {best_weights[0]:.2f}")
    print(f"   LightGBM:     {best_weights[1]:.2f}")
    print(f"   RandomForest: {best_weights[2]:.2f}")
    print(f"\nMeilleur score ({metric.upper()}):", end=" ")
    if metric == 'r2':
        print(f"{best_score:.4f}")
    else:
        print(f"${best_score:,.2f}")
    print("\n" + "=" * 80)
    print("\nTop 10 des meilleures combinaisons:")
    print(results_df.head(10).to_string(index=False))
    print("=" * 80)
    
    return {
        'best_weights': {
            'xgb': best_weights[0],
            'lgbm': best_weights[1],
            'rf': best_weights[2]
        },
        'best_score': best_score,
        'all_results': results_df,
        'best_results': best_full_results
    }


def optimize_blend_and_params(
        data,
        target_column='next_adjusted_salary',
        feature_columns=None,
        test_size=0.3,
        random_state=42,
        weight_step=0.2,
        metric='r2',
        param_grid=None):
    """
    Optimise à la fois les hyperparamètres des modèles et les poids du blend.

    Args:
        data (pd.DataFrame): Le jeu de données complet
        target_column (str): Nom de la colonne cible
        feature_columns (list): Liste des colonnes features
        test_size (float): Proportion du jeu de test
        random_state (int): Seed pour la reproductibilité
        weight_step (float): Pas de recherche des poids (ex: 0.2)
        metric (str): Métrique à optimiser ('r2', 'rmse', ou 'mae')
        param_grid (dict): Grille de paramètres à tester pour chaque modèle.
                          Format: {
                              'xgb': [
                                  {'n_estimators': 100, 'max_depth': 4},
                                  {'n_estimators': 200, 'max_depth': 6}
                              ],
                              'lgbm': [...],
                              'rf': [...]
                          }

    Returns:
        dict: Dictionnaire contenant:
            - 'best_config': Meilleure configuration trouvée
            - 'best_score': Meilleur score
            - 'all_configs': DataFrame avec toutes les configurations testées
            - 'best_model_results': Résultats du meilleur modèle
    """
    # Grille par défaut si non fournie
    if param_grid is None:
        param_grid = {
            'xgb': [
                {
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'max_depth': 4,
                    'random_state': random_state
                },
                {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': random_state
                },
                {
                    'n_estimators': 100,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'random_state': random_state
                }
            ],
            'lgbm': [
                {
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': random_state,
                    'verbose': -1
                },
                {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 7,
                    'random_state': random_state,
                    'verbose': -1
                },
                {
                    'n_estimators': 100,
                    'learning_rate': 0.05,
                    'max_depth': 9,
                    'random_state': random_state,
                    'verbose': -1
                }
            ],
            'rf': [
                {
                    'n_estimators': 50,
                    'max_depth': 5,
                    'random_state': random_state,
                    'n_jobs': -1
                },
                {
                    'n_estimators': 100,
                    'max_depth': 7,
                    'random_state': random_state,
                    'n_jobs': -1
                },
                {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': random_state,
                    'n_jobs': -1
                }
            ]
        }

    # Calculer le nombre total de configurations
    total_configs = (len(param_grid['xgb']) *
                     len(param_grid['lgbm']) *
                     len(param_grid['rf']))
    
    print("=" * 80)
    print("OPTIMISATION DES PARAMÈTRES ET DU BLEND")
    print("=" * 80)
    print(f"\nNombre de configurations de paramètres: {total_configs}")
    print(f"Métrique à optimiser: {metric.upper()}")
    print(f"Pas de recherche des poids: {weight_step}\n")
    
    all_results = []
    best_overall_score = -np.inf if metric == 'r2' else np.inf
    best_overall_config = None
    best_overall_results = None
    
    config_num = 0
    for xgb_params in param_grid['xgb']:
        for lgbm_params in param_grid['lgbm']:
            for rf_params in param_grid['rf']:
                config_num += 1
                
                print(f"\n--- Configuration {config_num}/{total_configs} ---")
                print(f"XGBoost: n_est={xgb_params.get('n_estimators')}, "
                      f"lr={xgb_params.get('learning_rate', 'N/A')}, "
                      f"depth={xgb_params.get('max_depth')}")
                print(f"LightGBM: n_est={lgbm_params.get('n_estimators')}, "
                      f"lr={lgbm_params.get('learning_rate')}, "
                      f"depth={lgbm_params.get('max_depth')}")
                print(f"RandomForest: n_est={rf_params.get('n_estimators')}, "
                      f"depth={rf_params.get('max_depth')}")
                
                # Trouver les meilleurs poids pour cette configuration
                blend_results = find_best_blend_weights(
                    data=data,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    test_size=test_size,
                    random_state=random_state,
                    weight_step=weight_step,
                    xgb_params=xgb_params,
                    lgbm_params=lgbm_params,
                    rf_params=rf_params,
                    metric=metric
                )
                
                score = blend_results['best_score']
                
                # Enregistrer cette configuration
                config_info = {
                    'config_id': config_num,
                    'xgb_n_estimators': xgb_params.get('n_estimators'),
                    'xgb_learning_rate': xgb_params.get('learning_rate'),
                    'xgb_max_depth': xgb_params.get('max_depth'),
                    'lgbm_n_estimators': lgbm_params.get('n_estimators'),
                    'lgbm_learning_rate': lgbm_params.get('learning_rate'),
                    'lgbm_max_depth': lgbm_params.get('max_depth'),
                    'rf_n_estimators': rf_params.get('n_estimators'),
                    'rf_max_depth': rf_params.get('max_depth'),
                    'best_xgb_weight': blend_results['best_weights']['xgb'],
                    'best_lgbm_weight': blend_results['best_weights']['lgbm'],
                    'best_rf_weight': blend_results['best_weights']['rf'],
                    'test_r2': blend_results['best_results']['metrics']['test']['r2_score'],
                    'test_rmse': blend_results['best_results']['metrics']['test']['rmse'],
                    'test_mae': blend_results['best_results']['metrics']['test']['mae']
                }
                all_results.append(config_info)
                
                # Mettre à jour le meilleur global
                if metric == 'r2':
                    is_better = score > best_overall_score
                else:
                    is_better = score < best_overall_score
                
                if is_better:
                    best_overall_score = score
                    best_overall_config = {
                        'xgb_params': xgb_params,
                        'lgbm_params': lgbm_params,
                        'rf_params': rf_params,
                        'weights': blend_results['best_weights']
                    }
                    best_overall_results = blend_results['best_results']
                    score_str = f"{score:.4f}" if metric == 'r2' else f"${score:,.2f}"
                    print(f">>> NOUVEAU MEILLEUR SCORE: {score_str}")
    
    # Créer un DataFrame avec tous les résultats
    results_df = pd.DataFrame(all_results)
    
    # Trier selon la métrique
    if metric == 'r2':
        results_df = results_df.sort_values('test_r2', ascending=False)
    elif metric == 'rmse':
        results_df = results_df.sort_values('test_rmse', ascending=True)
    elif metric == 'mae':
        results_df = results_df.sort_values('test_mae', ascending=True)
    
    results_df = results_df.reset_index(drop=True)
    
    # Affichage final
    print("\n" + "=" * 80)
    print("MEILLEURE CONFIGURATION TROUVÉE")
    print("=" * 80)
    print("\nParamètres XGBoost:")
    for key, value in best_overall_config['xgb_params'].items():
        if key != 'random_state':
            print(f"   {key}: {value}")
    
    print("\nParamètres LightGBM:")
    for key, value in best_overall_config['lgbm_params'].items():
        if key not in ['random_state', 'verbose']:
            print(f"   {key}: {value}")
    
    print("\nParamètres RandomForest:")
    for key, value in best_overall_config['rf_params'].items():
        if key not in ['random_state', 'n_jobs']:
            print(f"   {key}: {value}")
    
    print("\nMeilleurs poids du blend:")
    print(f"   XGBoost:      {best_overall_config['weights']['xgb']:.2f}")
    print(f"   LightGBM:     {best_overall_config['weights']['lgbm']:.2f}")
    print(f"   RandomForest: {best_overall_config['weights']['rf']:.2f}")
    
    print(f"\nMeilleur score ({metric.upper()}): ", end="")
    if metric == 'r2':
        print(f"{best_overall_score:.4f}")
    else:
        print(f"${best_overall_score:,.2f}")
    
    print("\n" + "=" * 80)
    print("\nTop 5 des meilleures configurations:")
    display_cols = [
        'config_id',
        'xgb_n_estimators', 'xgb_max_depth',
        'lgbm_n_estimators', 'lgbm_max_depth',
        'rf_n_estimators', 'rf_max_depth',
        'best_xgb_weight', 'best_lgbm_weight', 'best_rf_weight',
        'test_r2', 'test_rmse', 'test_mae'
    ]
    print(results_df[display_cols].head(5).to_string(index=False))
    print("=" * 80)
    
    return {
        'best_config': best_overall_config,
        'best_score': best_overall_score,
        'all_configs': results_df,
        'best_model_results': best_overall_results
    }