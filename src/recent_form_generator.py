import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class PlayerPerformanceAnalyzer:
    def __init__(self):
        self.venue_stats: Dict[str, Dict] = {}
        self.opposition_stats: Dict[str, Dict] = {}
        self.form_trends: Dict[str, List] = {}

    def calculate_venue_stats(self, df: pd.DataFrame, player: str) -> Dict:
        """Calculate player's performance statistics at different venues."""
        player_data = df[df['Player'] == player]
        venue_stats = {}
        
        for venue in player_data['Venue'].unique():
            venue_data = player_data[player_data['Venue'] == venue]
            venue_stats[venue] = {
                'matches': len(venue_data),
                'avg_points': venue_data['Total Points'].mean(),
                'std_points': venue_data['Total Points'].std(),
                'max_points': venue_data['Total Points'].max(),
                'min_points': venue_data['Total Points'].min()
            }
        return venue_stats

    def calculate_opposition_stats(self, df: pd.DataFrame, player: str) -> Dict:
        """Calculate player's performance against different teams."""
        player_data = df[df['Player'] == player]
        opposition_stats = {}
        
        for team in player_data['Opposition'].unique():
            team_data = player_data[player_data['Opposition'] == team]
            opposition_stats[team] = {
                'matches': len(team_data),
                'avg_points': team_data['Total Points'].mean(),
                'success_rate': len(team_data[team_data['Total Points'] > team_data['Total Points'].mean()]) / len(team_data)
            }
        return opposition_stats

    def calculate_form_trends(self, df: pd.DataFrame, player: str, window_sizes: List[int] = [3, 5, 10]) -> Dict:
        """Calculate player's form trends over different time windows."""
        player_data = df[df['Player'] == player].sort_values('Match Number')
        trends = {}
        
        for window in window_sizes:
            rolling_avg = player_data['Total Points'].rolling(window=window, min_periods=1).mean()
            rolling_std = player_data['Total Points'].rolling(window=window, min_periods=1).std()
            
            trends[f'{window}_match'] = {
                'recent_avg': rolling_avg.iloc[-1] if len(rolling_avg) > 0 else 0,
                'trend_slope': np.polyfit(range(len(rolling_avg[-window:])), rolling_avg[-window:], 1)[0] if len(rolling_avg) >= window else 0,
                'volatility': rolling_std.iloc[-1] if len(rolling_std) > 0 else 0
            }
        return trends

    def generate_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive performance metrics for all players."""
        metrics_list = []
        
        # Ensure all required columns exist
        required_columns = ['Player', 'Venue', 'Total Points', 'Match Number', 'Opposition']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Opposition':
                    # Infer Opposition if missing
                    if 'Team' in df.columns and 'Match Number' in df.columns:
                        df['Opposition'] = df.apply(lambda row: self._infer_opposition(row, df), axis=1)
                    else:
                        df['Opposition'] = 'Unknown'
                else:
                    df[col] = 'Unknown'

        # Debug: Check column names in the input DataFrame for generate_performance_metrics
        print("[DEBUG] Columns in input DataFrame for generate_performance_metrics:", df.columns.tolist())
        
        # Use 'Player' instead of 'Player Name'
        for player in df['Player'].unique():
            venue_stats = self.calculate_venue_stats(df, player)
            opposition_stats = self.calculate_opposition_stats(df, player)
            form_trends = self.calculate_form_trends(df, player)
            
            # Calculate consistency score (0-1)
            recent_volatility = form_trends['5_match']['volatility']
            max_volatility = df['Total Points'].std() * 2  # Benchmark
            consistency_score = 1 - min(recent_volatility / max_volatility, 1) if max_volatility > 0 else 0
            
            # Calculate form score (0-1)
            recent_avg = form_trends['5_match']['recent_avg']
            overall_avg = df['Total Points'].mean()
            form_score = min(recent_avg / (overall_avg * 1.5), 1) if overall_avg > 0 else 0
            
            metrics_list.append({
                'Player': player,
                'Recent Form': form_trends['5_match']['recent_avg'],
                'Form Trend': form_trends['5_match']['trend_slope'],
                'Consistency Score': consistency_score,
                'Form Score': form_score,
                'Overall Rating': (consistency_score * 0.4 + form_score * 0.6)
            })
        
        return pd.DataFrame(metrics_list)

    def _infer_opposition(self, row, df):
        # Try to infer opposition team based on match metadata if available
        if 'Match Number' in row and 'Team' in row:
            match_no = row['Match Number']
            player_team = row['Team']
            # Try to get home/away teams for this match
            if 'Home Team' in df.columns and 'Away Team' in df.columns:
                match_row = df[(df['Match Number'] == match_no)]
                if not match_row.empty:
                    home_team = match_row.iloc[0]['Home Team']
                    away_team = match_row.iloc[0]['Away Team']
                    if player_team == home_team:
                        return away_team
                    elif player_team == away_team:
                        return home_team
        return 'Unknown'

def generate_recent_form(fantasy_points_file: str, output_file: str = 'data/recent_form.csv'):
    """Generate comprehensive recent form analysis including venue and opposition stats."""
    df = pd.read_excel(fantasy_points_file)
    analyzer = PlayerPerformanceAnalyzer()
    
    # Generate comprehensive metrics
    performance_metrics = analyzer.generate_performance_metrics(df)
    
    # Save to CSV
    performance_metrics.to_csv(output_file, index=False)
    print(f"[✔] Comprehensive performance metrics written to {output_file}")
    
    return performance_metrics
