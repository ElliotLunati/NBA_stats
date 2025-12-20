"""
Module de construction de la base de données
"""



#################################
##        IMPORTATIONS         ##
#################################



from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguedashplayerstats
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
import requests
from bs4 import BeautifulSoup



#################################
##        INFOS JOUEURS        ##
#################################



def export_players_stats_to_csv(
        season='2023-24',
        season_type='Regular Season',
        filename='nba_players_stats.csv'):
    """
    Exporte les statistiques de tous les joueurs d'une saison NBA.

    Args:
        season (str): La saison au format 'YYYY-YY'
                      (ex: '2023-24', '2022-23')
        season_type (str): Type de saison
                           ('Regular Season', 'Playoffs')
        filename (str): Le nom du fichier CSV à créer

    Returns:
        pd.DataFrame: DataFrame contenant toutes les statistiques
    """
    print(f"Récupération des statistiques de la saison {season}...")

    try:
        # Récupérer les statistiques de tous les joueurs
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame',
            season_type_all_star=season_type
        )
        
        # Convertir en DataFrame
        df = player_stats.get_data_frames()[0]

        # Créer les répertoires si nécessaire
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Exporter en CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nFichier '{filename}' créé avec succès!")

        return df

    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        return None



def export_all_seasons(
        start_year=1996,
        end_year=2024,
        season_types=None):
    """
    Récupère les statistiques des joueurs pour plusieurs saisons.

    Args:
        start_year (int): Année de début
                          (ex: 1996 pour la saison 1996-97)
        end_year (int): Année de fin
                        (ex: 2024 pour la saison 2024-25)
        season_types (list): Types de saison
                             ('Regular Season', 'Playoffs')

    Returns:
        dict: Dictionnaire avec les résultats de chaque saison
    """
    if season_types is None:
        season_types = ['Regular Season', 'Playoffs']

    results = {}
    total_seasons = (end_year - start_year + 1) * len(season_types)
    current = 0

    print(f"{'='*60}")
    start_season = f"{start_year}-{start_year+1-2000}"
    end_season = f"{end_year}-{end_year+1-2000}"
    print(f"Récupération des statistiques de {start_season} "
          f"à {end_season}")
    print(f"Types de saison: {', '.join(season_types)}")
    print(f"Total: {total_seasons} fichiers à créer")
    print(f"{'='*60}\n")
    
    for year in range(start_year, end_year + 1):
        # Formatage de la saison (ex: "1996-97" ou "2023-24")
        season = f"{year}-{str(year + 1)[-2:]}"

        for season_type in season_types:
            current += 1

            # Créer le nom du fichier et le chemin
            season_type_short = season_type.replace(' ', '_')
            folder_path = f"./data/{season}/{season_type_short}"
            filename = (
                f"nba_players_stats_{season}_{season_type_short}.csv"
            )

            # Créer les dossiers si nécessaire
            os.makedirs(folder_path, exist_ok=True)

            print(
                f"[{current}/{total_seasons}] {season} - {season_type}"
            )

            # Récupérer et sauvegarder les données
            full_path = os.path.join(folder_path, filename)
            df = export_players_stats_to_csv(
                season=season,
                season_type=season_type,
                filename=full_path
            )

            if df is not None:
                results[f"{season}_{season_type}"] = {
                    'path': full_path,
                    'players_count': len(df)
                }

            # Pause pour éviter de surcharger l'API
            time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"Récupération terminée!")
    print(
        f"{len(results)} fichiers créés sur {total_seasons} attendus"
    )
    print(f"{'='*60}")

    return results



def scrape_player_salaries(
        season_year=2025,
        filename='nba_salaries.csv'):
    """
    Scrape les salaires des joueurs NBA depuis ESPN.

    Args:
        season_year (int): Année de fin de saison
                           (ex: 2025 pour 2024-25)
        filename (str): Nom du fichier CSV à créer

    Returns:
        pd.DataFrame: DataFrame contenant les salaires des joueurs
    """
    print(f"  Récupération des salaires depuis ESPN pour "
          f"{season_year-1}-{season_year}...")

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': (
                'text/html,application/xhtml+xml,application/xml;'
                'q=0.9,image/webp,*/*;q=0.8'
            ),
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        all_data = []
        page = 1
        max_pages = 20  # Limite de sécurité

        while page <= max_pages:
            # URL ESPN avec pagination
            url = (
                f"https://www.espn.com/nba/salaries/_/year/"
                f"{season_year}/page/{page}/seasontype/1"
            )

            print(f"    Page {page}...", end=" ")

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Trouver le tableau des salaires sur ESPN
            table = soup.find('table', class_='tablehead')

            if not table:
                # Essayer de trouver n'importe quel tableau
                table = soup.find('table')

            if not table:
                print(" Tableau non trouvé")
                break

            # Chercher toutes les lignes avec class="oddrow" ou "evenrow"
            rows = table.find_all('tr', class_=['oddrow', 'evenrow'])
            
            if not rows:
                print(" Fin (aucune ligne)")
                break

            # Extraire les données de cette page
            page_data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    # Colonne 0: Rang
                    rank = cols[0].get_text(strip=True)

                    # Colonne 1: Nom du joueur
                    player_full = cols[1].get_text(strip=True)

                    # Séparer le nom et le poste
                    if ',' in player_full:
                        player_name = player_full.split(',')[0].strip()
                        position = player_full.split(',')[1].strip()
                    else:
                        player_name = player_full
                        position = ''

                    # Colonne 2: Équipe
                    team = cols[2].get_text(strip=True)

                    # Colonne 3: Salaire
                    salary_text = cols[3].get_text(strip=True)
                    salary_clean = (
                        salary_text.replace('$', '')
                        .replace(',', '')
                        .replace(' ', '')
                    )

                    try:
                        salary = (
                            int(salary_clean)
                            if salary_clean.isdigit()
                            else 0
                        )
                    except Exception:
                        salary = 0

                    page_data.append({
                        'Rank': rank,
                        'Player': player_name,
                        'Position': position,
                        'Team': team,
                        'Salary': salary,
                        'Season': f"{season_year-1}-{str(season_year)[-2:]}"
                    })
            
            if not page_data:
                print(" Fin (aucune donnée)")
                break

            all_data.extend(page_data)
            print(f" {len(page_data)} joueurs")

            page += 1
            time.sleep(0.1)

        if not all_data:
            print("   Aucune donnée extraite")
            return None

        df = pd.DataFrame(all_data)

        # Créer les répertoires si nécessaire
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Exporter en CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"   {len(df)} salaires récupérés au total - "
              f"Fichier créé!")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion: {e}")
        return None
    except Exception as e:
        print(f"Erreur lors du scraping: {e}")
        return None
    


def export_all_salaries(start_year=2000, end_year=2025):
    """
    Récupère les salaires des joueurs pour plusieurs saisons.

    Args:
        start_year (int): Année de début
                          (ex: 1997 pour la saison 1996-97)
        end_year (int): Année de fin
                        (ex: 2025 pour la saison 2024-25)

    Returns:
        dict: Dictionnaire avec les résultats de chaque saison
    """
    results = {}
    total_seasons = end_year - start_year + 1

    print(f"{'='*60}")
    print(f" Récupération des salaires NBA")
    start_season = f"{start_year-1}-{str(start_year)[-2:]}"
    end_season = f"{end_year-1}-{str(end_year)[-2:]}"
    print(f"De la saison {start_season} à {end_season}")
    print(f"Total: {total_seasons} saisons")
    print(f"{'='*60}\n")
    
    for year in range(start_year, end_year + 1):
        season_str = f"{year-1}-{str(year)[-2:]}"
        folder_path = f"./data/{season_str}/Salaries"
        filename = f"nba_salaries_{season_str}.csv"

        # Créer les dossiers si nécessaire
        os.makedirs(folder_path, exist_ok=True)

        print(f"[{year-start_year+1}/{total_seasons}] "
              f"Saison {season_str}")

        # Récupérer et sauvegarder les données
        full_path = os.path.join(folder_path, filename)
        df = scrape_player_salaries(season_year=year, filename=full_path)

        if df is not None:
            results[season_str] = {
                'path': full_path,
                'players_count': len(df)
            }

        # Pause pour éviter d'être bloqué
        time.sleep(0.1)

    print(f"{'='*60}")
    print(f" Récupération terminée!")
    print(
        f" {len(results)} fichiers créés sur {total_seasons} attendus"
    )
    print(f"{'='*60}")

    return results



def scrape_mvp_data(filename='./data/MVP/nba_mvp_history.csv'):
    """
    Scrape l'historique des MVP NBA depuis Basketball Reference.

    Args:
        filename (str): Le nom du fichier CSV à créer

    Returns:
        pd.DataFrame: DataFrame contenant l'historique des MVP
    """
    url = "https://www.basketball-reference.com/awards/mvp.html"

    print(" Récupération de l'historique des MVP NBA avec Selenium...")
    print(f"  URL: {url}")

    driver = None

    try:
        # Configuration Chrome en mode headless (invisible)
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(
            '--disable-blink-features=AutomationControlled'
        )
        user_agent = (
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        chrome_options.add_argument(user_agent)
        chrome_options.add_experimental_option(
            "excludeSwitches", ["enable-automation"]
        )
        chrome_options.add_experimental_option(
            'useAutomationExtension', False
        )
        
        print(" Initialisation du navigateur")

        # Créer le driver
        driver = webdriver.Chrome(options=chrome_options)

        # Masquer l'automation
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', "
            "{get: () => undefined})"
        )

        print("Chargement de la page...")
        driver.get(url)

        # Attendre que le tableau soit chargé
        print("Attente du chargement du tableau...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "mvp_NBA"))
        )

        time.sleep(0.1)

        # Récupérer le HTML
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Trouver le tableau des MVP
        table = soup.find('table', id='mvp_NBA')

        if not table:
            print("   Tableau des MVP non trouvé")
            return None

        print("Tableau trouvé")

        # Extraire les données
        data = []
        tbody = table.find('tbody')

        if not tbody:
            print("Corps du tableau non trouvé")
            return None

        rows = tbody.find_all('tr')
        print(f"{len(rows)} lignes trouvées")
        
        for row in rows:
            # Ignorer les lignes d'en-tête répétées
            if 'class' in row.attrs and (
                'over_header' in row['class'] or
                'thead' in row['class']
            ):
                continue

            # Récupérer la saison
            season_th = row.find('th', {'data-stat': 'season'})
            if not season_th:
                continue

            season = season_th.get_text(strip=True)

            # Récupérer les colonnes
            cols = row.find_all('td')

            if len(cols) >= 16:
                lg = cols[0].get_text(strip=True)
                player = cols[1].get_text(strip=True)
                age = cols[3].get_text(strip=True)
                team = cols[4].get_text(strip=True)
                games = cols[5].get_text(strip=True)
                mp = cols[6].get_text(strip=True)
                pts = cols[7].get_text(strip=True)
                trb = cols[8].get_text(strip=True)
                ast = cols[9].get_text(strip=True)
                stl = cols[10].get_text(strip=True)
                blk = cols[11].get_text(strip=True)
                fg_pct = cols[12].get_text(strip=True)
                fg3_pct = cols[13].get_text(strip=True)
                ft_pct = cols[14].get_text(strip=True)
                ws = cols[15].get_text(strip=True)
                ws_per_48 = (
                    cols[16].get_text(strip=True)
                    if len(cols) > 16
                    else ''
                )
                
                data.append({
                    'Season': season,
                    'League': lg,
                    'Player': player,
                    'Age': age,
                    'Team': team,
                    'Games': games,
                    'MP': mp,
                    'PTS': pts,
                    'TRB': trb,
                    'AST': ast,
                    'STL': stl,
                    'BLK': blk,
                    'FG_PCT': fg_pct,
                    'FG3_PCT': fg3_pct,
                    'FT_PCT': ft_pct,
                    'WS': ws,
                    'WS_PER_48': ws_per_48
                })

        if not data:
            print("Aucune donnée extraite")
            return None

        df = pd.DataFrame(data)

        # Créer les répertoires si nécessaire
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Exporter en CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"{len(df)} MVP récupérés - Fichier '{filename}' créé!")
        print(f"Période couverte: {df['Season'].iloc[-1]} à "
              f"{df['Season'].iloc[0]}")

        return df

    except Exception as e:
        print(f"   Erreur lors du scraping: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Fermer le navigateur
        if driver:
            driver.quit()
            print("Navigateur fermé")



#################################
##        INFOS EQUIPES        ##
#################################



def export_teams_stats_to_csv(
        season='2023-24',
        season_type='Regular Season',
        filename='nba_teams_stats.csv'):
    """
    Exporte les statistiques de toutes les équipes d'une saison NBA.

    Args:
        season (str): La saison au format 'YYYY-YY'
                      (ex: '2023-24', '2022-23')
        season_type (str): Type de saison
                           ('Regular Season', 'Playoffs')
        filename (str): Le nom du fichier CSV à créer

    Returns:
        pd.DataFrame: DataFrame contenant toutes les statistiques
    """
    print(f"Récupération des statistiques des équipes "
          f"pour la saison {season}...")

    try:
        # Récupérer les statistiques de toutes les équipes
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame',
            season_type_all_star=season_type
        )
        
        # Convertir en DataFrame
        df = team_stats.get_data_frames()[0]

        initial_total = len(df)

        # Filtrer pour ne garder que les équipes NBA
        if 'TEAM_ID' in df.columns:
            df = df[
                df['TEAM_ID'].astype(str).str.startswith('16106127')
            ]
            nba_teams = len(df)
            filtered_out = initial_total - nba_teams
            if filtered_out > 0:
                print(f"  {filtered_out} équipe(s) non-NBA "
                      f"filtrée(s) (WNBA/G-League)")

        # Supprimer les lignes avec TEAM_NAME vide ou NaN
        if 'TEAM_NAME' in df.columns:
            initial_count = len(df)
            df = df[
                df['TEAM_NAME'].notna() & (df['TEAM_NAME'] != '')
            ]
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"  {removed_count} ligne(s) avec "
                      f"TEAM_NAME vide supprimée(s)")

        # Supprimer les doublons basés sur TEAM_ID
        if 'TEAM_ID' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['TEAM_ID'], keep='first')
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"  {removed_count} doublon(s) "
                      f"TEAM_ID supprimé(s)")
        
        # Créer les répertoires si nécessaire
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Exporter en CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nFichier '{filename}' créé avec succès! "
              f"({len(df)} équipes NBA)")

        return df

    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        return None



def export_all_teams_seasons(
        start_year=1996,
        end_year=2024,
        season_types=None):
    """
    Récupère les statistiques des équipes pour plusieurs saisons.

    Args:
        start_year (int): Année de début
                          (ex: 1996 pour la saison 1996-97)
        end_year (int): Année de fin
                        (ex: 2024 pour la saison 2024-25)
        season_types (list): Types de saison
                             ('Regular Season', 'Playoffs')

    Returns:
        dict: Dictionnaire avec les résultats de chaque saison
    """
    if season_types is None:
        season_types = ['Regular Season', 'Playoffs']

    results = {}
    total_seasons = (end_year - start_year + 1) * len(season_types)
    current = 0

    print(f"{'='*60}")
    print(f"Récupération des statistiques des ÉQUIPES NBA")
    start_season = f"{start_year}-{str(start_year+1)[-2:]}"
    end_season = f"{end_year}-{str(end_year+1)[-2:]}"
    print(f"De {start_season} à {end_season}")
    print(f"Types de saison: {', '.join(season_types)}")
    print(f"Total: {total_seasons} fichiers à créer")
    print(f"{'='*60}\n")
    
    for year in range(start_year, end_year + 1):
        # Formatage de la saison (ex: "1996-97" ou "2023-24")
        season = f"{year}-{str(year + 1)[-2:]}"

        for season_type in season_types:
            current += 1

            # Créer le nom du fichier et le chemin
            season_type_short = season_type.replace(' ', '_')
            folder_path = f"./data/{season}/{season_type_short}"
            filename = (
                f"nba_teams_stats_{season}_{season_type_short}.csv"
            )

            # Créer les dossiers si nécessaire
            os.makedirs(folder_path, exist_ok=True)

            print(
                f"[{current}/{total_seasons}] {season} - {season_type}"
            )

            # Récupérer et sauvegarder les données
            full_path = os.path.join(folder_path, filename)
            df = export_teams_stats_to_csv(
                season=season,
                season_type=season_type,
                filename=full_path
            )

            if df is not None:
                results[f"{season}_{season_type}"] = {
                    'path': full_path,
                    'teams_count': len(df)
                }

            # Pause pour éviter de surcharger l'API
            time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"Récupération terminée!")
    print(
        f"{len(results)} fichiers créés sur {total_seasons} attendus"
    )
    print(f"{'='*60}")

    return results



#################################
##      CONSTRUCTION BDD       ##
#################################



def merge_data(
        start_year=2000,
        end_year=2025,
        output_file='./data/merged_data.csv'):
    """
    Fusionne les statistiques des joueurs avec leurs salaires.

    Combine les données de Regular Season avec les salaires pour chaque
    année, puis fusionne toutes les années en un seul fichier CSV.
    Ajoute 'adjusted_salary' (ajusté à l'inflation) et 'TEAM_W_PCT'.

    Args:
        start_year (int): Année de début
                          (ex: 2000 pour la saison 1999-00)
        end_year (int): Année de fin
                        (ex: 2025 pour la saison 2024-25)
        output_file (str): Chemin du fichier de sortie final

    Returns:
        pd.DataFrame: DataFrame combiné de toutes les années
    """
    # Taux d'inflation cumulatifs (Bureau of Labor Statistics)
    inflation_factors = {
        1999: 1.94,
        2000: 1.88,
        2001: 1.83,
        2002: 1.80,
        2003: 1.76,
        2004: 1.72,
        2005: 1.66,
        2006: 1.61,
        2007: 1.56,
        2008: 1.50,
        2009: 1.51,
        2010: 1.49,
        2011: 1.44,
        2012: 1.41,
        2013: 1.39,
        2014: 1.37,
        2015: 1.37,
        2016: 1.35,
        2017: 1.32,
        2018: 1.29,
        2019: 1.27,
        2020: 1.25,
        2021: 1.20,
        2022: 1.11,
        2023: 1.06,
        2024: 1.03,
        2025: 1.00
    }
    
    print(f"{'='*60}")
    print(f" Fusion des données joueurs + salaires")
    start_season = f"{start_year-1}-{str(start_year)[-2:]}"
    end_season = f"{end_year-1}-{str(end_year)[-2:]}"
    print(f" De {start_season} à {end_season}")
    print(f" Ajustement des salaires à l'inflation de {end_year-1}")
    print(f"{'='*60}\n")

    all_merged_data = []
    success_count = 0

    for year in range(start_year, end_year + 1):
        season_str = f"{year-1}-{str(year)[-2:]}"

        # Chemins des fichiers
        players_file = (
            f"./data/{season_str}/Regular_Season/"
            f"nba_players_stats_{season_str}_Regular_Season.csv"
        )
        salaries_file = (
            f"./data/{season_str}/Salaries/"
            f"nba_salaries_{season_str}.csv"
        )

        print(f"[{year-start_year+1}/{end_year-start_year+1}] "
              f"Saison {season_str}...", end=" ")

        # Vérifier que les deux fichiers existent
        if not os.path.exists(players_file):
            print(f" Fichier joueurs non trouvé")
            continue

        if not os.path.exists(salaries_file):
            print(f" Fichier salaires non trouvé")
            continue
        
        try:
            # Charger les données
            df_players = pd.read_csv(players_file)
            df_salaries = pd.read_csv(salaries_file)

            # Charger les statistiques des équipes pour W_PCT
            teams_file = (
                f"./data/{season_str}/Regular_Season/"
                f"nba_teams_stats_{season_str}_Regular_Season.csv"
            )
            df_teams = None
            if os.path.exists(teams_file):
                df_teams = pd.read_csv(teams_file)

            # Fusionner sur PLAYER_NAME = Player
            df_merged = pd.merge(
                df_players,
                df_salaries,
                left_on='PLAYER_NAME',
                right_on='Player',
                how='inner'
            )

            # Ajouter le W_PCT des équipes si disponible
            if (df_teams is not None and
                'W_PCT' in df_teams.columns and
                    'TEAM_ID' in df_teams.columns):
                # Créer un dictionnaire TEAM_ID -> W_PCT
                team_wpct = (
                    df_teams.set_index('TEAM_ID')['W_PCT'].to_dict()
                )
                # Ajouter la colonne TEAM_W_PCT
                df_merged['TEAM_W_PCT'] = (
                    df_merged['TEAM_ID'].map(team_wpct)
                )
            else:
                df_merged['TEAM_W_PCT'] = None

            # Ajouter la colonne year
            df_merged['Year'] = season_str

            # Calculer le salaire ajusté à l'inflation
            season_year = year - 1
            inflation_factor = inflation_factors.get(season_year, 1.0)

            # Calculer le salaire ajusté
            df_merged['adjusted_salary'] = (
                (df_merged['Salary'] * inflation_factor)
                .round(0)
                .astype(int)
            )

            # Supprimer la colonne 'Player' dupliquée
            if 'Player' in df_merged.columns:
                df_merged = df_merged.drop(columns=['Player'])

            all_merged_data.append(df_merged)
            success_count += 1

            print(f" {len(df_merged)} joueurs fusionnés")

        except Exception as e:
            print(f" Erreur: {e}")
            continue
    
    if not all_merged_data:
        print("\n Aucune donnée à fusionner")
        return None

    # Combiner tous les DataFrames
    print(f"\n{'='*60}")
    print(f" Combinaison de toutes les saisons...")
    df_final = pd.concat(all_merged_data, ignore_index=True)

    # Trier par joueur et année
    df_final = df_final.sort_values(['PLAYER_ID', 'Year'])

    # Créer colonne avec salaire ajusté de l'année suivante
    df_final['next_adjusted_salary'] = (
        df_final.groupby('PLAYER_ID')['adjusted_salary'].shift(-1)
    )

    # Créer colonne pour l'équipe de l'année suivante
    df_final['next_team'] = (
        df_final.groupby('PLAYER_ID')['TEAM_ID'].shift(-1)
    )

    # Créer la colonne Changed_team
    df_final['Changed_team'] = (
        (df_final['TEAM_ID'] != df_final['next_team']).astype(int)
    )
    # Convertir les NaN (dernière saison du joueur)
    df_final.loc[df_final['next_team'].isna(), 'Changed_team'] = None

    # Supprimer la colonne temporaire next_team
    df_final = df_final.drop(columns=['next_team'])

    # Compter le nombre de saisons jouées (YOE: Years Of Experience)
    df_final['YOE'] = df_final.groupby('PLAYER_ID').cumcount() + 1

    # Remettre dans l'ordre original
    df_final = df_final.reset_index(drop=True)

    # Créer les répertoires si nécessaire
    file_dir = os.path.dirname(output_file)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    # Sauvegarder le fichier final
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"  Fichier final créé: {output_file}")
    print(f" Total de lignes: {len(df_final)}")
    print(f" Saisons fusionnées: {success_count}/"
          f"{end_year-start_year+1}")
    print(f" Colonnes: {len(df_final.columns)}")
    print(f"{'='*60}")

    return df_final



def scrape_all_data(
        season_types=['Regular Season', 'Playoffs']):
    """
    Récupère toutes les données NBA et crée la base de données.

    Fonction principale qui orchestre la récupération de toutes les
    statistiques NBA (joueurs, équipes, salaires, MVP) et fusionne
    les données.

    Args:
        season_types (list): Types de saison
                             ('Regular Season', 'Playoffs')
    """

    export_all_seasons(
        start_year=1996,
        end_year=2024,
        season_types=season_types
    )
    export_all_teams_seasons(
        start_year=1996,
        end_year=2024,
        season_types=season_types
    )
    export_all_salaries(start_year=2000, end_year=2025)
    scrape_mvp_data(filename='./data/MVP/nba_mvp_history.csv')
    merge_data(
        start_year=2000,
        end_year=2025,
        output_file='./data/merged_data.csv'
    )