"""
Ce module contient des fonctions pour analyser les corrélations
entre variables, nettoyer les données et le pre-processing.
"""



#################################
##        IMPORTATIONS         ##
#################################



import numpy as np
import pandas as pd
from typing import List, Tuple, Optional



#################################
##        CORRELATIONS         ##
#################################



def find_high_correlations(
    dataframe: pd.DataFrame, 
    threshold: float = 0.80
) -> List[Tuple[str, str, float]]:
    """
    Identifie les paires de variables ayant une corrélation élevée.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données à analyser
        threshold (float): Seuil de corrélation en valeur absolue (par défaut 0.80)
        
    Returns:
        List[Tuple[str, str, float]]: Liste de tuples (variable1, variable2, corrélation)
        
    Example:
        >>> correlations = find_high_correlations(df, threshold=0.80)
        >>> print(f"Trouvé {len(correlations)} paires corrélées")
    """
    # Extraction des variables numériques et calcul de la matrice de corrélation
    correlation_matrix = dataframe.select_dtypes(include=['float64', 'int64']).corr()
    
    # Convertir la matrice en format long et retirer les doublons
    corr_pairs = correlation_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]  # Enlever les corrélations parfaites (diagonale)
    
    # Filtrer les corrélations >= threshold en valeur absolue
    high_corr = corr_pairs[corr_pairs.abs() >= threshold]
    
    # Trier par valeur absolue de corrélation (du plus fort au plus faible)
    high_corr_sorted = high_corr.reindex(high_corr.abs().sort_values(ascending=False).index)
    
    # Construire la liste des paires uniques
    correlations_list = []
    seen = set()
    
    for idx, corr_value in high_corr_sorted.items():
        var1, var2 = idx
        pair = tuple(sorted([var1, var2]))
        
        if pair not in seen:
            seen.add(pair)
            correlations_list.append((var1, var2, corr_value))
    
    return correlations_list



def print_correlation_report(
    correlations: List[Tuple[str, str, float]], 
    threshold: float = 0.80
) -> None:
    """
    Affiche un rapport formaté des corrélations élevées.
    
    Args:
        correlations (List[Tuple[str, str, float]]): Liste des corrélations à afficher
        threshold (float): Seuil de corrélation utilisé (pour l'affichage)
    """
    print(f"Variables avec une corrélation >= {threshold:.0%} (en valeur absolue):\n")
    print("=" * 70)
    print(f"Nombre total de paires trouvées: {len(correlations)}\n")
    
    for i, (var1, var2, corr) in enumerate(correlations, 1):
        print(f"{i:3d}. {var1:20s} <-> {var2:20s} : {corr:+.4f}")
    
    print("\n" + "=" * 70)



def analyze_correlations(
    dataframe: pd.DataFrame, 
    threshold: float = 0.80,
    verbose: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Fonction principale combinant recherche et affichage des corrélations.
    
    Args:
        dataframe (pd.DataFrame): DataFrame à analyser
        threshold (float): Seuil de corrélation en valeur absolue
        verbose (bool): Si True, affiche le rapport détaillé
        
    Returns:
        List[Tuple[str, str, float]]: Liste des paires de variables corrélées
    """
    correlations = find_high_correlations(dataframe, threshold)
    
    if verbose:
        print_correlation_report(correlations, threshold)
    
    return correlations



def remove_highly_correlated_features(
    dataframe: pd.DataFrame,
    threshold: float = 0.80,
    protected_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> List[str]:
    """
    Identifie les colonnes à supprimer en raison de corrélations élevées.
    
    En cas de corrélation élevée entre deux variables, conserve celle qui a
    la corrélation moyenne la plus faible avec les autres variables.
    Les colonnes protégées ne sont jamais supprimées.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données à analyser
        threshold (float): Seuil de corrélation en valeur absolue (défaut: 0.80)
        protected_cols (Optional[List[str]]): Liste des colonnes à protéger
        verbose (bool): Si True, affiche les décisions de suppression
        
    Returns:
        List[str]: Liste des noms de colonnes à supprimer
        
    Example:
        >>> columns_to_drop = remove_highly_correlated_features(
        ...     df, 
        ...     threshold=0.80, 
        ...     protected_cols=['Salary', 'adjusted_salary']
        ... )
        >>> df_cleaned = df.drop(columns=columns_to_drop)
    """
    if protected_cols is None:
        protected_cols = []
    
    # Sélectionner uniquement les colonnes numériques
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])
    
    # Calculer la matrice de corrélation
    corr_matrix = numeric_df.corr().abs()
    
    # Créer la matrice triangulaire supérieure
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Ensemble des colonnes à supprimer
    to_drop = set()
    
    for column in upper_triangle.columns:
        # Trouver les variables corrélées >= threshold avec cette colonne
        high_corr_vars = upper_triangle.index[
            upper_triangle[column] >= threshold
        ].tolist()
        
        if high_corr_vars:
            for var in high_corr_vars:
                if column not in to_drop and var not in to_drop:
                    _process_correlated_pair(
                        column, 
                        var, 
                        corr_matrix,
                        upper_triangle,
                        protected_cols,
                        to_drop,
                        verbose
                    )
    
    return list(to_drop)



def _process_correlated_pair(
    column: str,
    var: str,
    corr_matrix: pd.DataFrame,
    upper_triangle: pd.DataFrame,
    protected_cols: List[str],
    to_drop: set,
    verbose: bool
) -> None:
    """
    Traite une paire de variables corrélées et décide laquelle supprimer.
    
    Fonction auxiliaire privée pour remove_highly_correlated_features.
    
    Args:
        column (str): Première variable de la paire
        var (str): Deuxième variable de la paire
        corr_matrix (pd.DataFrame): Matrice de corrélation complète
        upper_triangle (pd.DataFrame): Matrice triangulaire supérieure
        protected_cols (List[str]): Liste des colonnes protégées
        to_drop (set): Ensemble des colonnes à supprimer (modifié en place)
        verbose (bool): Si True, affiche les décisions
    """
    column_protected = column in protected_cols
    var_protected = var in protected_cols
    corr_value = upper_triangle.loc[var, column]
    
    # Cas 1: Les deux sont protégées
    if column_protected and var_protected:
        if verbose:
            message = (
                f"'{column}' et '{var}' sont toutes deux protégées "
                f"(corr: {corr_value:.3f})"
            )
            print(message)
        return
    
    # Cas 2: Seulement column est protégée
    if column_protected:
        to_drop.add(var)
        if verbose:
            message = (
                f"Suppression de '{var}' - corrélée à '{column}' "
                f"[PROTÉGÉE] (corr: {corr_value:.3f})"
            )
            print(message)
        return
    
    # Cas 3: Seulement var est protégée
    if var_protected:
        to_drop.add(column)
        if verbose:
            message = (
                f"Suppression de '{column}' - corrélée à '{var}' "
                f"[PROTÉGÉE] (corr: {corr_value:.3f})"
            )
            print(message)
        return
    
    # Cas 4: Aucune n'est protégée - utiliser la corrélation moyenne
    avg_corr_column = corr_matrix[column].drop(column).mean()
    avg_corr_var = corr_matrix[var].drop(var).mean()
    
    if avg_corr_column > avg_corr_var:
        to_drop.add(column)
        if verbose:
            message = (
                f"Suppression de '{column}' (corr. moy: {avg_corr_column:.3f}) "
                f"- corrélée à '{var}' (corr: {corr_value:.3f})"
            )
            print(message)
    else:
        to_drop.add(var)
        if verbose:
            message = (
                f"Suppression de '{var}' (corr. moy: {avg_corr_var:.3f}) "
                f"- corrélée à '{column}' (corr: {corr_value:.3f})"
            )
            print(message)



#################################
##          OUTLIERS           ##
#################################



def remove_low_salary_outliers(
    dataframe: pd.DataFrame,
    min_salary: float = 500000,
    salary_column: str = 'adjusted_salary',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Supprime les lignes avec un salaire ajusté inférieur au seuil minimal.
    
    Cette fonction retire les observations avec des salaires anormalement bas
    tout en conservant les lignes où le salaire est manquant (NaN).
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données à nettoyer
        min_salary (float): Seuil minimal de salaire (défaut: 500000)
        salary_column (str): Nom de la colonne contenant le salaire
        verbose (bool): Si True, affiche les statistiques de suppression
        
    Returns:
        pd.DataFrame: DataFrame nettoyé sans les salaires trop bas
        
    Example:
        >>> df_clean = remove_low_salary_outliers(df, min_salary=500000)
        >>> print(f"Lignes restantes: {len(df_clean)}")
    """
    initial_count = len(dataframe)
    
    # Conserver les lignes avec salaire >= min_salary OU salaire manquant
    df_cleaned = dataframe[
        (dataframe[salary_column].isna()) | 
        (dataframe[salary_column] >= min_salary)
    ].copy()
    
    removed_count = initial_count - len(df_cleaned)
    
    if verbose:
        print(f"Suppression des salaires ajustés < {min_salary:,.0f}$")
        print(f"   Lignes supprimées: {removed_count}")
        print(f"   Lignes restantes: {len(df_cleaned)}")
    
    return df_cleaned



def detect_career_end_drops(
    dataframe: pd.DataFrame,
    player_id_col: str = 'PLAYER_ID',
    year_col: str = 'Year',
    salary_col: str = 'adjusted_salary',
    drop_threshold: float = -0.70,
    verbose: bool = True
) -> List[int]:
    """
    Détecte les indices des observations correspondant à des fins de carrière.
    
    Identifie les joueurs ayant subi une baisse de salaire supérieure au seuil
    et retourne les indices de toutes leurs observations à partir de cette baisse.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données
        player_id_col (str): Nom de la colonne identifiant le joueur
        year_col (str): Nom de la colonne contenant l'année
        salary_col (str): Nom de la colonne contenant le salaire
        drop_threshold (float): Seuil de baisse (défaut: -0.70 pour -70%)
        verbose (bool): Si True, affiche les statistiques de détection
        
    Returns:
        List[int]: Liste des indices à supprimer du DataFrame
        
    Example:
        >>> indices = detect_career_end_drops(df, drop_threshold=-0.70)
        >>> df_clean = df.drop(indices)
    """
    # Trier par joueur et année
    df_sorted = dataframe.sort_values([player_id_col, year_col]).copy()
    
    # Calculer la variation de salaire d'une année sur l'autre pour chaque joueur
    df_sorted['salary_change_pct'] = df_sorted.groupby(
        player_id_col
    )[salary_col].pct_change()
    
    # Identifier les joueurs avec une baisse >= drop_threshold
    career_end_mask = df_sorted['salary_change_pct'] <= drop_threshold
    players_with_drops = df_sorted[career_end_mask][player_id_col].unique()
    
    if verbose:
        threshold_display = abs(drop_threshold) * 100
        print(f"Détection des fins de carrière (baisse > {threshold_display:.0f}%)")
        print(f"   Joueurs avec baisse > {threshold_display:.0f}%: {len(players_with_drops)}")
    
    # Liste des indices à supprimer
    indices_to_drop = []
    
    for player_id in players_with_drops:
        player_data = df_sorted[df_sorted[player_id_col] == player_id].copy()
        
        # Trouver la première année où la baisse >= drop_threshold se produit
        career_end_years = player_data[
            player_data['salary_change_pct'] <= drop_threshold
        ][year_col].values
        
        if len(career_end_years) > 0:
            first_career_end_year = career_end_years[0]
            
            # Ajouter tous les indices à partir de cette année
            indices_to_drop.extend(
                player_data[player_data[year_col] >= first_career_end_year].index.tolist()
            )
    
    if verbose:
        print(f"   Lignes identifiées (fin de carrière): {len(indices_to_drop)}")
    
    return indices_to_drop



def clean_outliers(
    dataframe: pd.DataFrame,
    min_salary: float = 500000,
    drop_threshold: float = -0.70,
    player_id_col: str = 'PLAYER_ID',
    year_col: str = 'Year',
    salary_col: str = 'adjusted_salary',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Nettoie le dataset en supprimant les outliers et les fins de carrière.
    
    Cette fonction combine le nettoyage des salaires trop bas et la détection
    des fins de carrière pour produire un dataset nettoyé.
    
    Args:
        dataframe (pd.DataFrame): DataFrame à nettoyer
        min_salary (float): Seuil minimal de salaire (défaut: 500000)
        drop_threshold (float): Seuil de baisse baisse du salaire pour considérer une fin de carrière (défaut: -0.70)
        player_id_col (str): Nom de la colonne identifiant le joueur
        year_col (str): Nom de la colonne contenant l'année
        salary_col (str): Nom de la colonne contenant le salaire
        verbose (bool): Si True, affiche un rapport détaillé
        
    Returns:
        pd.DataFrame: DataFrame nettoyé
        
    Example:
        >>> df_clean = clean_outliers(df, min_salary=500000, drop_threshold=-0.70)
        >>> print(f"Dataset final: {len(df_clean)} lignes")
    """
    initial_count = len(dataframe)
    
    if verbose:
        print("=" * 80)
        print("NETTOYAGE DES OUTLIERS")
        print("=" * 80)
        print(f"\nDataset initial: {initial_count} lignes")
        print()
    
    # Créer une copie pour le nettoyage
    df_cleaned = dataframe.copy()
    
    # Étape 1: Supprimer les salaires trop bas
    if verbose:
        print("1. ", end="")
    
    df_cleaned = remove_low_salary_outliers(
        df_cleaned,
        min_salary=min_salary,
        salary_column=salary_col,
        verbose=verbose
    )
    removed_low_salary = initial_count - len(df_cleaned)
    
    if verbose:
        print()
    
    # Étape 2: Détecter et supprimer les fins de carrière
    if verbose:
        print("2. ", end="")
    
    indices_to_drop = detect_career_end_drops(
        df_cleaned,
        player_id_col=player_id_col,
        year_col=year_col,
        salary_col=salary_col,
        drop_threshold=drop_threshold,
        verbose=verbose
    )
    
    count_before_career_drop = len(df_cleaned)
    df_cleaned = df_cleaned.drop(indices_to_drop)
    removed_career_end = count_before_career_drop - len(df_cleaned)
    
    if verbose:
        print(f"   Lignes supprimées (fin de carrière): {removed_career_end}")
        print(f"   Lignes restantes: {len(df_cleaned)}")
    
    # Résumé final
    total_removed = initial_count - len(df_cleaned)
    removal_percentage = (total_removed / initial_count * 100) if initial_count > 0 else 0
    
    if verbose:
        print(f"\n{'=' * 80}")
        print("RÉSUMÉ DU NETTOYAGE")
        print("=" * 80)
        print(f"Dataset initial:                {initial_count} lignes")
        print(f"Supprimés (salaire < {min_salary:,.0f}$):    {removed_low_salary} lignes")
        print(f"Supprimés (fin de carrière):    {removed_career_end} lignes")
        print(f"Total supprimé:                 {total_removed} lignes ({removal_percentage:.2f}%)")
        print(f"Dataset final:                  {len(df_cleaned)} lignes")
        print(f"\n Nettoyage des outliers terminé")
        print("=" * 80)
    
    return df_cleaned



#################################
##     PREPROCESS PIPELINE     ##
#################################



def remove_identifier_columns(
    dataframe: pd.DataFrame,
    columns_to_remove: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Supprime les colonnes identifiantes non pertinentes pour l'entraînement.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données
        columns_to_remove (Optional[List[str]]): Liste des colonnes à supprimer.
            Si None, utilise une liste par défaut.
        verbose (bool): Si True, affiche les colonnes supprimées
        
    Returns:
        pd.DataFrame: DataFrame sans les colonnes identifiantes
        
    Example:
        >>> df_clean = remove_identifier_columns(df)
        >>> print(f"Colonnes restantes: {df_clean.shape[1]}")
    """
    if columns_to_remove is None:
        columns_to_remove = [
            'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 
            'Team', 'Salary', 'Season', 'Year', 'adjusted_salary', 'Rank'
        ]
    
    # Identifier les colonnes existantes à supprimer
    existing_cols_to_remove = [
        col for col in columns_to_remove if col in dataframe.columns
    ]
    
    if existing_cols_to_remove:
        df_processed = dataframe.drop(columns=existing_cols_to_remove)
        
        if verbose:
            print(f"Colonnes supprimées: {existing_cols_to_remove}")
            print(f"Nouvelles dimensions: {df_processed.shape}")
        
        return df_processed
    else:
        if verbose:
            print("Aucune colonne identifiante à supprimer")
        
        return dataframe.copy()



def encode_categorical_columns(
    dataframe: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode les variables catégorielles avec LabelEncoder.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données
        categorical_columns (Optional[List[str]]): Liste des colonnes à encoder.
            Si None, encode 'Position' et 'TEAM_ABBREVIATION' par défaut.
        verbose (bool): Si True, affiche les informations d'encodage
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame avec colonnes encodées et 
            dictionnaire des encoders utilisés
        
    Example:
        >>> df_encoded, encoders = encode_categorical_columns(df)
        >>> print(f"Encoders disponibles: {list(encoders.keys())}")
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_processed = dataframe.copy()
    encoders = {}
    encoded_columns = []
    
    if categorical_columns is None:
        categorical_columns = ['Position', 'TEAM_ABBREVIATION']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            encoder = LabelEncoder()
            encoded_col_name = f"{col}_encoded"
            
            df_processed[encoded_col_name] = encoder.fit_transform(
                df_processed[col].astype(str)
            )
            encoders[col] = encoder
            encoded_columns.append(col)
            
            if verbose:
                print(f"\n{col} encodée:")
                print(f"   Nombre de classes: {len(encoder.classes_)}")
                
                if col == 'Position' or len(encoder.classes_) <= 10:
                    print(f"   Classes: {list(encoder.classes_)}")
                    mapping = dict(
                        zip(encoder.classes_, encoder.transform(encoder.classes_))
                    )
                    print(f"   Mapping: {mapping}")
    
    # Supprimer les colonnes catégorielles originales
    if encoded_columns:
        df_processed = df_processed.drop(columns=encoded_columns)
        
        if verbose:
            print(f"\nColonnes catégorielles originales supprimées: {encoded_columns}")
    
    return df_processed, encoders



def normalize_numeric_columns(
    dataframe: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, 'StandardScaler']:
    """
    Normalise les colonnes numériques avec StandardScaler.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données
        exclude_columns (Optional[List[str]]): Colonnes à exclure de la normalisation.
            Si None, exclut les colonnes de salaire par défaut.
        verbose (bool): Si True, affiche les informations de normalisation
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: DataFrame normalisé et 
            scaler utilisé pour la transformation
        
    Example:
        >>> df_normalized, scaler = normalize_numeric_columns(df)
        >>> print(f"Colonnes normalisées: {len(scaler.feature_names_in_)}")
    """
    from sklearn.preprocessing import StandardScaler
    
    df_processed = dataframe.copy()
    
    if exclude_columns is None:
        exclude_columns = ['next_adjusted_salary', 'adjusted_salary', 'Salary']
    
    # Identifier les colonnes numériques
    numeric_columns = df_processed.select_dtypes(
        include=['float64', 'int64']
    ).columns.tolist()
    
    # Exclure les colonnes spécifiées
    columns_to_normalize = [
        col for col in numeric_columns if col not in exclude_columns
    ]
    
    if verbose:
        print(f"Colonnes à normaliser ({len(columns_to_normalize)}):")
        if len(columns_to_normalize) > 10:
            print(f"   {columns_to_normalize[:10]}...")
        else:
            print(f"   {columns_to_normalize}")
    
    # Normalisation
    scaler = StandardScaler()
    df_processed[columns_to_normalize] = scaler.fit_transform(
        df_processed[columns_to_normalize]
    )
    
    if verbose:
        print(f"\nNormalisation effectuée avec StandardScaler")
        print(f"   Moyenne ≈ 0, Écart-type ≈ 1")
    
    return df_processed, scaler



def check_missing_values(
    dataframe: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vérifie et gère les valeurs manquantes dans le DataFrame.
    Supprime les lignes où next_adjusted_salary est manquant.
    Remplit les autres valeurs manquantes avec la moyenne de la colonne.
    
    Args:
        dataframe (pd.DataFrame): DataFrame à vérifier
        verbose (bool): Si True, affiche un rapport détaillé
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame nettoyé et série contenant 
            le nombre initial de valeurs manquantes par colonne
        
    Example:
        >>> df_clean, missing = check_missing_values(df)
        >>> print(f"Colonnes avec NaN: {len(missing[missing > 0])}")
    """
    df_processed = dataframe.copy()
    initial_count = len(df_processed)
    missing_values = df_processed.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    if verbose:
        if len(missing_cols) > 0:
            print("Valeurs manquantes détectées:")
            for col, count in missing_cols.items():
                percentage = count / len(df_processed) * 100
                print(f"   {col}: {count} ({percentage:.2f}%)")
        else:
            print("Aucune valeur manquante détectée")
    
    # Supprimer les lignes où next_adjusted_salary est manquant
    if 'next_adjusted_salary' in df_processed.columns:
        rows_before = len(df_processed)
        df_processed = df_processed.dropna(subset=['next_adjusted_salary'])
        rows_dropped = rows_before - len(df_processed)
        
        if verbose and rows_dropped > 0:
            print(f"\nLignes supprimées (next_adjusted_salary manquant): {rows_dropped}")
    
    # Remplir les autres valeurs manquantes avec la moyenne
    remaining_missing = df_processed.isnull().sum()
    cols_to_fill = remaining_missing[remaining_missing > 0].index.tolist()
    
    if cols_to_fill:
        for col in cols_to_fill:
            if df_processed[col].dtype in ['float64', 'int64']:
                mean_value = df_processed[col].mean()
                df_processed[col].fillna(mean_value, inplace=True)
                
                if verbose:
                    print(f"   {col}: rempli avec la moyenne ({mean_value:.2f})")
    
    if verbose:
        final_missing = df_processed.isnull().sum().sum()
        if final_missing == 0:
            print(f"\nToutes les valeurs manquantes ont été traitées")
            print(f"Lignes restantes: {len(df_processed)}")
    
    return df_processed, missing_values



def preprocess_pipeline(
    dataframe: pd.DataFrame,
    columns_to_remove: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    exclude_from_normalization: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Pipeline complet de prétraitement des données NBA.
    
    Cette fonction enchaîne toutes les étapes de prétraitement :
    1. Suppression des colonnes identifiantes
    2. Encodage des variables catégorielles
    3. Normalisation des variables numériques
    4. Vérification des valeurs manquantes
    
    Args:
        dataframe (pd.DataFrame): DataFrame à prétraiter
        columns_to_remove (Optional[List[str]]): Colonnes identifiantes à supprimer
        categorical_columns (Optional[List[str]]): Colonnes catégorielles à encoder
        exclude_from_normalization (Optional[List[str]]): Colonnes à exclure de la normalisation
        verbose (bool): Si True, affiche un rapport détaillé
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame prétraité et dictionnaire contenant
            les encoders et le scaler utilisés
        
    Example:
        >>> df_preprocessed, artifacts = preprocess_pipeline(df, verbose=True)
        >>> print(f"Shape finale: {df_preprocessed.shape}")
        >>> print(f"Artifacts: {list(artifacts.keys())}")
    """
    if verbose:
        print("=" * 80)
        print("PIPELINE DE PRÉTRAITEMENT DES DONNÉES")
        print("=" * 80)
        print(f"\n1. Dataset initial: {dataframe.shape}")
    
    # Créer une copie pour le prétraitement
    df_preprocessed = dataframe.copy()
    artifacts = {}
    
    # Étape 2: Supprimer les colonnes identifiantes
    if verbose:
        print(f"\n2. ", end="")
    
    df_preprocessed = remove_identifier_columns(
        df_preprocessed,
        columns_to_remove=columns_to_remove,
        verbose=verbose
    )
    
    # Étape 3: Encoder les variables catégorielles
    if verbose:
        print(f"\n3. Encodage des variables catégorielles:")
    
    df_preprocessed, encoders = encode_categorical_columns(
        df_preprocessed,
        categorical_columns=categorical_columns,
        verbose=verbose
    )
    artifacts['encoders'] = encoders
    
    # Étape 4: Normaliser les variables numériques
    if verbose:
        print(f"\n4. ", end="")
    
    df_preprocessed, scaler = normalize_numeric_columns(
        df_preprocessed,
        exclude_columns=exclude_from_normalization,
        verbose=verbose
    )
    artifacts['scaler'] = scaler
    
    # Étape 5: Vérifier et gérer les valeurs manquantes
    if verbose:
        print(f"\n5. ", end="")
    
    df_preprocessed, missing_values = check_missing_values(df_preprocessed, verbose=verbose)
    artifacts['missing_values'] = missing_values
    
    # Résumé final
    if verbose:
        print(f"\n{'=' * 80}")
        print("RÉSUMÉ DU PRÉTRAITEMENT")
        print("=" * 80)
        print(f"Shape finale: {df_preprocessed.shape}")
        print(f"Colonnes catégorielles encodées: {len(encoders)}")
        
        if hasattr(scaler, 'n_features_in_'):
            print(f"Colonnes numériques normalisées: {scaler.n_features_in_}")
        
        print(f"\nPremières lignes du dataset prétraité:")
        print(df_preprocessed.head())
        
        # Statistiques des colonnes normalisées (échantillon)
        normalized_cols = [
            col for col in df_preprocessed.select_dtypes(
                include=['float64', 'int64']
            ).columns
            if col not in (exclude_from_normalization or [])
        ]
        
        if normalized_cols:
            sample_cols = normalized_cols[:5]
            print(f"\n{'=' * 80}")
            print("Statistiques des colonnes normalisées (échantillon):")
            print(df_preprocessed[sample_cols].describe())
        
        print(f"\n Prétraitement terminé")
        print("=" * 80)
        print('\n')
    
    return df_preprocessed, artifacts


