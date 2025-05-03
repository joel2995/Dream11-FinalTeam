import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataStandardizer:
    def __init__(self):
        self.player_id_map: Dict[str, int] = {}
        self.next_id = 1

    def _generate_player_id(self, player_name: str) -> int:
        """Generate or retrieve a unique ID for a player."""
        if player_name not in self.player_id_map:
            self.player_id_map[player_name] = self.next_id
            self.next_id += 1
        return self.player_id_map[player_name]

    def standardize_player_names(self, df: pd.DataFrame, name_column: str) -> pd.DataFrame:
        """Standardize player names by removing extra spaces and normalizing case."""
        df = df.copy()
        df[name_column] = df[name_column].str.strip()
        df[name_column] = df[name_column].str.title()
        return df

    def process_squad_data(self, squad_df: pd.DataFrame) -> pd.DataFrame:
        """Process squad data with standardized columns and player IDs."""
        df = squad_df.copy()
        
        # Standardize column names
        column_mapping = {
            'Player Name': 'Player',
            'Player Type': 'Player Role',
            'Team': 'Team',
            'IsPlaying': 'IsPlaying',
            'lineupOrder': 'LineupOrder'
        }
        df = df.rename(columns=column_mapping)
        
        # Standardize player names
        df = self.standardize_player_names(df, 'Player')
        
        # Add player IDs
        df['PlayerID'] = df['Player'].apply(self._generate_player_id)
        
        # Ensure consistent role names
        role_mapping = {
            'WK': 'WK',
            'BAT': 'BAT',
            'BOWL': 'BOWL',
            'ALL': 'ALL'
        }
        df['Player Role'] = df['Player Role'].map(role_mapping)
        
        return df

    def process_match_data(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """Process match data with standardized columns and metrics."""
        df = match_df.copy()
        
        # Standardize player names
        df = self.standardize_player_names(df, 'Player')
        
        # Add player IDs
        df['PlayerID'] = df['Player'].apply(self._generate_player_id)
        
        # Convert numeric columns
        numeric_columns = ['Runs', 'Balls', 'Fours', 'Sixes', 'Strike Rate', 
                         'Overs', 'Runs Conceded', 'Wickets', 'Economy']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    def process_credit_data(self, credit_df: pd.DataFrame) -> pd.DataFrame:
        """Process credit data with standardized columns and priorities."""
        df = credit_df.copy()
        
        # Standardize column names
        column_mapping = {
            'Player Type': 'Player Role',
            'Player Name': 'Player',
            'Team': 'Team',
            'Priority': 'Priority',
            'Credits': 'Credits'
        }
        df = df.rename(columns=column_mapping)
        
        # Standardize player names
        df = self.standardize_player_names(df, 'Player')
        
        # Add player IDs
        df['PlayerID'] = df['Player'].apply(self._generate_player_id)
        
        # Ensure credits are numeric
        df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(0)
        
        # Standardize priority values (1 is highest priority)
        if 'Priority' in df.columns:
            df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce').fillna(3)
        
        return df