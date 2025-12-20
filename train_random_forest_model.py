import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_random_forest(
        data,
        target_column='next_adjusted_salary',
        feature_columns=None,
        test_size=0.3,
        random_state=42,
        **rf_params):
    """
    Split les données et entraîne un modèle Random Forest.

    Args:
        data (pd.DataFrame): Le jeu de données complet
        target_column (str): Nom de la colonne cible (y)
        feature_columns (list): Liste des colonnes features (X).
                                Si None, utilise toutes les colonnes
                                sauf target_column
        test_size (float): Proportion du jeu de test (0 à 1)
        random_state (int): Seed pour la reproductibilité
        **rf_params: Paramètres supplémentaires pour RandomForest
                     (n_estimators, max_depth, etc.)

    Returns:
        dict: Dictionnaire contenant:
            - 'model': Le modèle entraîné
            - 'X_train': Features d'entraînement
            - 'X_test': Features de test
            - 'y_train': Cible d'entraînement
            - 'y_test': Cible de test
            - 'predictions': Prédictions sur le jeu de test
            - 'metrics': Métriques de performance
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

    # Créer et entraîner le modèle
    model = RandomForestRegressor(
        random_state=random_state,
        **rf_params
    )

    print(f"\nEntraînement du modèle RandomForestRegressor...")
    model.fit(X_train, y_train)

    # Prédictions sur les jeux d'entraînement et de test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculer les métriques pour l'entraînement
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # Calculer les métriques pour le test
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Affichage formaté des résultats
    print("\n" + "=" * 80)
    print("RÉSULTATS DU MODÈLE")
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

    # Retourner tous les résultats
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'metrics': metrics,
        'feature_columns': feature_columns
    }


def get_feature_importance(results, top_n=5):
    """
    Affiche l'importance des features du modèle Random Forest.

    Args:
        results (dict): Résultats retournés par train_random_forest
        top_n (int): Nombre de features les plus importantes à afficher

    Returns:
        pd.DataFrame: DataFrame avec les features et leur importance
    """
    model = results['model']
    feature_columns = results['feature_columns']

    # Créer un DataFrame avec les importances
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })

    # Trier par importance décroissante
    importance_df = importance_df.sort_values(
        'importance',
        ascending=False
    ).reset_index(drop=True)

    # Afficher les top N features
    print(f"\nTop {top_n} features les plus importantes:")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df
