import pandas as pd
import logging
import sys
import os
from recent_form_generator import PlayerPerformanceAnalyzer
from data_standardizer import DataStandardizer
from fantasy_point_calculator import calculate_total_points
from model_predictor import ModelPredictor
from strategy_engine import StrategyEngine

# Import the captaincy priority data
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
from captaincy_priority import get_captaincy_priority_data, get_captain_vice_captain_for_match

class TeamSelector:
    def __init__(self, match_data_path, squad_data_path, match_metadata_path, credit_data_path):
        try:
            logging.info("Initializing TeamSelector...")
            
            # Initialize file paths
            self.match_data_path = match_data_path
            self.squad_data_path = squad_data_path
            self.match_metadata_path = match_metadata_path
            self.credit_data_path = credit_data_path
            
            # Load data files
            logging.info("Loading data files...")
            self.match_data = pd.read_excel(self.match_data_path)
            self.squad_data = pd.read_excel(self.squad_data_path, sheet_name=None)  # Load all sheets as a dictionary
            self.match_metadata = pd.read_excel(self.match_metadata_path)
            self.credit_data = pd.read_excel(self.credit_data_path)
            
            # Standardize column names
            self.match_data.rename(columns={
                'Match no': 'Match Number',
                'Player': 'Player',
                'Runs': 'Runs',
                'Balls': 'Balls Faced',
                'Fours': 'Fours',
                'Sixes': 'Sixes',
                'Strike Rate': 'Strike Rate',
                'Overs': 'Overs',
                'Runs Conceded': 'Runs Conceded',
                'Wickets': 'Wickets',
                'Economy': 'Economy',
                'Catches': 'Catches',
                'Stumping': 'Stumpings',
                'Run Out (Direct)': 'Run Outs (Direct)',
                'Run Out (Indirect)': 'Run Outs (Assist)',
                'Player Role': 'Player Role',
                'Team': 'Team',
                'Credits': 'Credits'
            }, inplace=True)

            self.credit_data.rename(columns={
                'Player Name': 'Player',
                'Player Type': 'Player Role',
                'Team': 'Team',
                'Credits': 'Credits'
            }, inplace=True)

            for sheet_name, df in self.squad_data.items():
                self.squad_data[sheet_name].rename(columns={
                    'Player Name': 'Player',
                    'Player Type': 'Player Role',
                    'Team': 'Team',
                    'IsPlaying': 'IsPlaying',
                    'lineupOrder': 'LineupOrder'
                }, inplace=True)

            self.match_metadata.rename(columns={
                'Match No.': 'Match Number',
                'Date': 'Date',
                'Venue': 'Venue',
                'Home Team': 'Home Team',
                'Away Team': 'Away Team',
                'Toss Winner': 'Toss Winner',
                'Toss Decision': 'Toss Decision',
                'Pitch Condition': 'Pitch Condition',
                'Match Result': 'Match Result',
                'Toss Winner is Match Winner': 'Toss Winner is Match Winner',
                'Winning Team': 'Winning Team'
            }, inplace=True)

            logging.info("Data files loaded and standardized successfully.")
            
            # Initialize other components
            logging.info("Initializing components...")
            self.performance_analyzer = PlayerPerformanceAnalyzer()
            self.model_predictor = ModelPredictor()
            self.strategy_engine = StrategyEngine()
            logging.info("Components initialized successfully.")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to initialize TeamSelector: {e}")

    def load_squad_for_match(self, match_id):
        try:
            logging.info(f"Loading squad data for match ID {match_id}...")
            sheet_name = f"Match_{match_id}"
            if sheet_name not in self.squad_data:
                raise ValueError(f"Sheet '{sheet_name}' not found in squad data.")
            return self.squad_data[sheet_name]
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load squad data for match ID {match_id}: {e}")

    def predict(self, match_id, home_team, away_team, venue, toss_winner, toss_decision):
        try:
            logging.info("Starting prediction process...")
            # Try to load squad data for the match
            try:
                squad_df = self.load_squad_for_match(match_id)
                squad_loaded = True
            except Exception as squad_exc:
                logging.warning(f"Squad data for match ID {match_id} not found: {squad_exc}")
                squad_loaded = False
                squad_df = None

            # If squad data is loaded, proceed as before
            if squad_loaded and squad_df is not None and not squad_df.empty:
                # Filter playing XI
                squad_df = squad_df[squad_df['IsPlaying'].str.upper() == "PLAYING"].copy()
            else:
                # Fallback: Use best 11 + 4 backups from home and away teams
                logging.info("Falling back to best 11 + 4 backups from specified home and away teams.")
                combined_df = self.credit_data[(self.credit_data['Team'].str.upper() == str(home_team).upper()) | (self.credit_data['Team'].str.upper() == str(away_team).upper())].copy()
                required_columns = [
                    'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                    'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                    'Run Outs (Direct)', 'Run Outs (Assist)', 'Maidens', 'Runs Given', 'IsPlaying', 'LineupOrder'
                ]
                for col in required_columns:
                    if col not in combined_df.columns:
                        combined_df[col] = 0
                combined_df['IsPlaying'] = 'PLAYING'
                squad_df = combined_df

            # --- Custom logic for X_FACTOR_SUBSTITUTE inclusion ---
            x_factor_players = {
                'MI': 'Rohit Sharma',
                'RCB': 'Devdutt Padikkal',
                'DC': 'Abishek Porel',
                'CSK': 'Shivam Dube',
                'GT': 'Sherfane Rutherford',
                'KKR': 'Angkrish Raghuvanshi',
                'RR': 'Vaibhav Suryavanshi'
            }
            # For each team, if the player is present and IsPlaying is X_FACTOR_SUBSTITUTE, treat as PLAYING
            for team, player in x_factor_players.items():
                mask = (squad_df['Team'].str.upper() == team) & (squad_df['Player'].str.strip().str.upper() == player.upper())
                if mask.any():
                    idx = squad_df[mask & (squad_df['IsPlaying'].str.upper() == 'X_FACTOR_SUBSTITUTE')].index
                    squad_df.loc[idx, 'IsPlaying'] = 'PLAYING'

            # Continue with the rest of the logic
            # Debug information is logged to file but not printed to console
            logging.debug("Squad DataFrame columns: %s", squad_df.columns.tolist())
            logging.debug("Unique Players in squad_df: %s", squad_df['Player'].unique())
            logging.debug("Unique Teams in squad_df: %s", squad_df['Team'].unique())

            match_row = self.match_metadata[self.match_metadata['Match Number'] == match_id]
            if not match_row.empty:
                actual_home_team = match_row.iloc[0]['Home Team'] if 'Home Team' in match_row.columns else home_team
                actual_away_team = match_row.iloc[0]['Away Team'] if 'Away Team' in match_row.columns else away_team
                if (str(home_team).upper() != str(actual_home_team).upper()) or (str(away_team).upper() != str(actual_away_team).upper()):
                    logging.warning(f"User-provided home/away teams do not match metadata. Using actual teams from metadata: {actual_home_team} vs {actual_away_team}")
                    home_team = actual_home_team
                    away_team = actual_away_team

            df = squad_df.merge(
                self.credit_data[['Player', 'Credits', 'Team', 'Priority']],
                on=['Player', 'Team'],
                how='left'
            )
            if 'Credits_y' in df.columns:
                df['Credits'] = df['Credits_y']
                df = df.drop(columns=['Credits_x', 'Credits_y'], errors='ignore')
            if 'Credits' not in df.columns:
                logging.debug("Merged DataFrame preview: %s", df.head())
                raise ValueError("[ERROR] 'Credits' column is missing after merging squad and credit data.")
            df = df.dropna(subset=['Credits'])
            df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
            required_columns = [
                'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                'Run Outs (Direct)', 'Run Outs (Assist)', 'Maidens', 'Runs Given'
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            match_row = self.match_metadata[self.match_metadata['Match Number'] == match_id]
            if not match_row.empty:
                venue_value = match_row.iloc[0]['Venue'] if 'Venue' in match_row.columns else 'Unknown'
                df['Venue'] = venue_value
                if 'Home Team' not in df.columns and 'Home Team' in match_row.columns:
                    df['Home Team'] = match_row.iloc[0]['Home Team']
                if 'Away Team' not in df.columns and 'Away Team' in match_row.columns:
                    df['Away Team'] = match_row.iloc[0]['Away Team']
            else:
                venue_value = 'Unknown'
                df['Venue'] = 'Unknown'
                if 'Home Team' not in df.columns:
                    df['Home Team'] = 'Unknown'
                if 'Away Team' not in df.columns:
                    df['Away Team'] = 'Unknown'

            # --- Team selection constraints logic ---
            # Only consider players marked as PLAYING
            playing_df = df[df['IsPlaying'].str.upper() == 'PLAYING'].copy()
            # Assign player roles
            role_map = {
                'WK': 'wk', 'WICKETKEEPER': 'wk',
                'BAT': 'bat', 'BATSMAN': 'bat',
                'ALL': 'all', 'ALLROUNDER': 'all',
                'BOWL': 'bowl', 'BOWLER': 'bowl'
            }
            playing_df['Role'] = playing_df['Player Role'].str.upper().map(role_map).fillna('bat')
            
            # Merge Priority from credit_data if not already present
            if 'Priority' not in playing_df.columns:
                playing_df = playing_df.merge(
                    self.credit_data[['Player', 'Team', 'Priority']],
                    on=['Player', 'Team'],
                    how='left'
                )
                playing_df['Priority'] = pd.to_numeric(playing_df['Priority'], errors='coerce').fillna(3)
            
            # Use ML model to enhance player selection
            # Create features for ML prediction
            feature_cols = [
                'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                'Run Outs (Direct)', 'Run Outs (Assist)'
            ]
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in playing_df.columns:
                    playing_df[col] = 0
            
            # Train model predictor with historical data if available
            if not self.match_data.empty:
                # Create target variable based on fantasy points
                target_df = self.match_data.copy()
                target_df['Total_Points'] = calculate_total_points(target_df)
                
                # Train the model
                self.model_predictor.train(target_df, feature_cols, 'Total_Points')
                
                # Predict selection scores for current players
                self.model_predictor.predict(playing_df)
                
                # Combine ML score with Priority for better selection
                # Scale selection_score to 0-1 range
                if 'selection_score' in playing_df.columns:
                    max_score = playing_df['selection_score'].max()
                    min_score = playing_df['selection_score'].min()
                    if max_score > min_score:  # Avoid division by zero
                        playing_df['selection_score_scaled'] = (playing_df['selection_score'] - min_score) / (max_score - min_score)
                    else:
                        playing_df['selection_score_scaled'] = 0.5  # Default if all scores are the same
                    
                    # Combine Priority and ML score (Priority has higher weight - increased from 70% to 85%)
                    playing_df['combined_score'] = (4 - playing_df['Priority']) * 0.85 + playing_df['selection_score_scaled'] * 0.15
                else:
                    # If ML prediction failed, use Priority only
                    playing_df['combined_score'] = 4 - playing_df['Priority']
            else:
                # If no historical data, use Priority only
                playing_df['combined_score'] = 4 - playing_df['Priority']
            
            # Group by team
            team_counts = playing_df['Team'].value_counts().to_dict()
            
            # Use ML model to enhance player selection with Priority values
            # Create features for ML prediction
            feature_cols = [
                'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                'Run Outs (Direct)', 'Run Outs (Assist)'
            ]
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in playing_df.columns:
                    playing_df[col] = 0
            
            # Train model predictor with historical data if available
            if not self.match_data.empty:
                # Create target variable based on fantasy points
                target_df = self.match_data.copy()
                target_df['Total_Points'] = calculate_total_points(target_df)
                
                # Train the model
                self.model_predictor.train(target_df, feature_cols, 'Total_Points')
                
                # Predict selection scores for current players
                self.model_predictor.predict(playing_df)
                
                # Combine ML score with Priority for better selection
                # Scale selection_score to 0-1 range
                if 'selection_score' in playing_df.columns:
                    max_score = playing_df['selection_score'].max()
                    min_score = playing_df['selection_score'].min()
                    if max_score > min_score:  # Avoid division by zero
                        playing_df['selection_score_scaled'] = (playing_df['selection_score'] - min_score) / (max_score - min_score)
                    else:
                        playing_df['selection_score_scaled'] = 0.5  # Default if all scores are the same
                    
                    # Combine Priority and ML score (Priority has higher weight - increased from 70% to 85%)
                    playing_df['combined_score'] = (4 - playing_df['Priority']) * 0.85 + playing_df['selection_score_scaled'] * 0.15
                else:
                    # If ML prediction failed, use Priority only
                    playing_df['combined_score'] = 4 - playing_df['Priority']
            else:
                # If no historical data, use Priority only
                playing_df['combined_score'] = 4 - playing_df['Priority']
            
            # Sort players by combined score (descending)
            playing_df = playing_df.sort_values('combined_score', ascending=False)
            
            # First, ensure we have at least one player from each role
            # This addresses the requirement to ensure minimum role requirements are met
            
            # Initialize selection lists and counters
            selected = []
            team_player_counts = {team: 0 for team in playing_df['Team'].unique()}
            role_counts = {'wk': 0, 'bat': 0, 'all': 0, 'bowl': 0}
            total_credits = 0
            
            # First, select one player from each required role
            for role in ['wk', 'bat', 'all', 'bowl']:
                # Get players of this role, sorted by combined_score
                role_players = playing_df[playing_df['Role'] == role].sort_values('combined_score', ascending=False)
                
                if not role_players.empty:
                    # Take the highest scoring player of this role
                    best_player = role_players.iloc[0]
                    team = best_player['Team']
                    credits = best_player['Credits']
                    
                    # Add this player to the selection
                    selected.append(best_player)
                    team_player_counts[team] += 1
                    role_counts[role] += 1
                    total_credits += credits
                    
                    # Remove this player from consideration for the next steps
                    playing_df = playing_df[playing_df.index != best_player.name]
            
            # Now separate remaining players by priority for selection
            priority_1_players = playing_df[playing_df['Priority'] == 1].copy()
            priority_2_players = playing_df[playing_df['Priority'] == 2].copy()
            priority_3_players = playing_df[playing_df['Priority'] == 3].copy()
            
            # Add Priority 1 players (sorted by ML score within priority)
            for _, row in priority_1_players.sort_values('combined_score', ascending=False).iterrows():
                if len(selected) >= 11:
                    break
                    
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Then add Priority 2 players until we reach 11 players (sorted by ML score within priority)
            for _, row in priority_2_players.sort_values('combined_score', ascending=False).iterrows():
                if len(selected) >= 11:
                    break
                    
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Finally add Priority 3 players if needed to reach 11 players (sorted by ML score within priority)
            for _, row in priority_3_players.sort_values('combined_score', ascending=False).iterrows():
                if len(selected) >= 11:
                    break
                    
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                        
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Ensure min 4 per team
            for team in team_player_counts:
                if team_player_counts[team] < 4:
                    # Try to add more from this team if possible
                    candidates = playing_df[(playing_df['Team'] == team) & (~playing_df.index.isin([s.name for s in selected]))]
                    for _, row in candidates.iterrows():
                        if len(selected) >= 11:
                            break
                        selected.append(row)
                        team_player_counts[team] += 1
                        role_counts[row['Role']] += 1
                        total_credits += row['Credits']
            # Final team (main 11)
            final_team = pd.DataFrame(selected).head(11)
            
            # --- Backup selection logic ---
            # Select backups based on Priority values
            backup_candidates = playing_df[~playing_df.index.isin(final_team.index)]
            
            # First select Priority 1 players not already in main team
            priority_1_backups = backup_candidates[backup_candidates['Priority'] == 1]
            # Then Priority 2 players
            priority_2_backups = backup_candidates[backup_candidates['Priority'] == 2]
            # Then Priority 3 players
            priority_3_backups = backup_candidates[backup_candidates['Priority'] == 3]
            
            # Combine in priority order
            backups_by_priority = pd.concat([priority_1_backups, priority_2_backups, priority_3_backups])
            backups = backups_by_priority.head(4).copy()
            backups['RoleFlag'] = 'Backup'  # Explicitly set RoleFlag for backups
            
            # Add RoleFlag to main team
            final_team['RoleFlag'] = 'Main'
            # Combine for output
            output_team = pd.concat([final_team, backups], ignore_index=True)
            # --- Captain and Vice-Captain selection using user observations and ML ---
            # Only allow C/VC from WK, BAT, ALL roles (not BOWL)
            eligible_roles = ['wk', 'bat', 'all']
            
            # Filter eligible C/VC from main team only, not backups, and not BOWL
            main_eligible = final_team[final_team['Role'].isin(eligible_roles)].copy()
            
            logging.info("Using user observations and ML-based approach for Captain and Vice-Captain selection")
            
            # Define features for C/VC selection
            c_vc_features = [
                'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 
                'Wickets', 'Economy', 'Catches', 'Stumpings', 
                'Run Outs (Direct)', 'Run Outs (Assist)', 'Priority'
            ]
            
            # Ensure all features exist
            for col in c_vc_features:
                if col not in main_eligible.columns:
                    main_eligible[col] = 0
            
            # Calculate a captain score based on performance metrics and priority
            main_eligible['captain_score'] = (
                main_eligible['Runs'] * 0.5 + 
                main_eligible['Wickets'] * 2.0 + 
                main_eligible['Fours'] * 0.2 + 
                main_eligible['Sixes'] * 0.3 + 
                main_eligible['Catches'] * 0.5 + 
                main_eligible['Stumpings'] * 0.5 + 
                main_eligible['Run Outs (Direct)'] * 0.5 + 
                (4 - main_eligible['Priority']) * 5.0  # Priority has significant weight
            )
            
            # If we have historical data and ML predictions, use them
            if 'selection_score' in main_eligible.columns:
                main_eligible['captain_score'] += main_eligible['selection_score'] * 2.0
            
            # Get home and away team players
            home_team_players = main_eligible[main_eligible['Team'] == home_team]
            away_team_players = main_eligible[main_eligible['Team'] == away_team]
            
            # Get user-defined captaincy priority data
            captaincy_data = get_captaincy_priority_data()
            captain_candidates, vice_captain_candidates = get_captain_vice_captain_for_match(home_team, away_team)
            
            # Create a new column for user-defined captaincy priority
            main_eligible['user_captaincy_priority'] = 0
            
            # Assign user captaincy priority values
            for idx, player in main_eligible.iterrows():
                player_name = player['Player']
                if player_name in captaincy_data:
                    if captaincy_data[player_name]['priority'] == 1:
                        main_eligible.loc[idx, 'user_captaincy_priority'] = 100  # High value for captain candidates
                    elif captaincy_data[player_name]['priority'] == 2:
                        main_eligible.loc[idx, 'user_captaincy_priority'] = 50   # Medium value for vice-captain candidates
            
            # Add user captaincy priority to captain score
            main_eligible['captain_score'] += main_eligible['user_captaincy_priority']
            
            # Sort players by captain_score
            main_eligible = main_eligible.sort_values('captain_score', ascending=False)
            
            # First, try to select captain and vice-captain from user-defined candidates
            captain_from_user_data = False
            vice_captain_from_user_data = False
            
            # Find players in the selected team that match user's captain candidates
            user_captain_in_team = main_eligible[main_eligible['Player'].isin(captain_candidates)]
            user_vice_captain_in_team = main_eligible[main_eligible['Player'].isin(vice_captain_candidates)]
            
            # If we have user-defined captain candidates in our team, use the highest scoring one
            if not user_captain_in_team.empty:
                captain = user_captain_in_team.iloc[0]  # Highest scoring captain candidate
                captain_from_user_data = True
                logging.info(f"Captain selected from user observations: {captain['Player']} ({captain['Team']})")
            else:
                # Fallback to ML-based selection for captain
                captain = main_eligible.iloc[0]
                logging.info(f"Captain selected from ML predictions: {captain['Player']} ({captain['Team']})")
            
            # For vice-captain, first try user-defined vice-captain candidates
            if not user_vice_captain_in_team.empty:
                # Filter out the captain if they're also in the vice-captain list
                user_vice_captain_in_team = user_vice_captain_in_team[user_vice_captain_in_team['Player'] != captain['Player']]
                if not user_vice_captain_in_team.empty:
                    vice_captain = user_vice_captain_in_team.iloc[0]
                    vice_captain_from_user_data = True
                    logging.info(f"Vice-Captain selected from user observations: {vice_captain['Player']} ({vice_captain['Team']})")
                else:
                    # If no vice-captain candidates left, try captain candidates (excluding the selected captain)
                    user_captain_in_team = user_captain_in_team[user_captain_in_team['Player'] != captain['Player']]
                    if not user_captain_in_team.empty:
                        vice_captain = user_captain_in_team.iloc[0]
                        vice_captain_from_user_data = True
                        logging.info(f"Vice-Captain selected from user captain observations: {vice_captain['Player']} ({vice_captain['Team']})")
                    else:
                        # Fallback to ML-based selection for vice-captain
                        vice_captain_candidates = main_eligible[main_eligible['Player'] != captain['Player']]
                        vice_captain = vice_captain_candidates.iloc[0] if not vice_captain_candidates.empty else captain
                        logging.info(f"Vice-Captain selected from ML predictions: {vice_captain['Player']} ({vice_captain['Team']})")
            else:
                # If no user-defined vice-captain candidates, try captain candidates (excluding the selected captain)
                user_captain_in_team = user_captain_in_team[user_captain_in_team['Player'] != captain['Player']]
                if not user_captain_in_team.empty:
                    vice_captain = user_captain_in_team.iloc[0]
                    vice_captain_from_user_data = True
                    logging.info(f"Vice-Captain selected from user captain observations: {vice_captain['Player']} ({vice_captain['Team']})")
                else:
                    # Fallback to ML-based selection for vice-captain
                    vice_captain_candidates = main_eligible[main_eligible['Player'] != captain['Player']]
                    vice_captain = vice_captain_candidates.iloc[0] if not vice_captain_candidates.empty else captain
                    logging.info(f"Vice-Captain selected from ML predictions: {vice_captain['Player']} ({vice_captain['Team']})")
            
            # Assign C and VC
            if captain is not None and vice_captain is not None:
                output_team['C'] = output_team['Player'] == captain['Player']
                output_team['VC'] = output_team['Player'] == vice_captain['Player']
                
                # Log the selection source
                if captain_from_user_data:
                    logging.info(f"Captain {captain['Player']} selected based on user observations")
                else:
                    logging.info(f"Captain {captain['Player']} selected based on ML predictions")
                    
                if vice_captain_from_user_data:
                    logging.info(f"Vice-Captain {vice_captain['Player']} selected based on user observations")
                else:
                    logging.info(f"Vice-Captain {vice_captain['Player']} selected based on ML predictions")
            else:
                # Ultimate fallback if something went wrong
                batallwk = final_team[final_team['Role'].isin(['wk','bat','all'])]
                if not batallwk.empty:
                    output_team['C'] = output_team['Player'] == batallwk.iloc[0]['Player']
                    output_team['VC'] = output_team['Player'] == batallwk.iloc[1]['Player'] if len(batallwk) > 1 else False
                    logging.error("Using emergency fallback for Captain and Vice-Captain selection")
                else:
                    output_team['C'] = False
                    output_team['VC'] = False
                    logging.error("No eligible players found for Captain and Vice-Captain roles")
            # Log the summary (removed console print to avoid duplication)
            # The summary will be printed from main.py instead
            logging.info("\n[TEAM SELECTION SUMMARY]")
            logging.info(f"Team composition: {{'wk': {role_counts['wk']}, 'bat': {role_counts['bat']}, 'all': {role_counts['all']}, 'bowl': {role_counts['bowl']}}}")
            logging.info(f"Players from each team: {team_player_counts}")
            logging.info(f"Total credits used: {total_credits}")
            logging.info(f"Final 11: {final_team['Player'].tolist()}")
            logging.info(f"Backups: {backups['Player'].tolist()}")
            logging.info(f"Captain: {output_team[output_team['C']]['Player'].tolist()}")
            logging.info(f"Vice-Captain: {output_team[output_team['VC']]['Player'].tolist()}")
            return output_team.reset_index(drop=True)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise