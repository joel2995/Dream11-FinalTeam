import pandas as pd
import numpy as np

# Fantasy point rules
BAT_RUN_POINTS = 1
FOUR_BONUS = 1
SIX_BONUS = 2
STRIKE_RATE_THRESHOLDS = [(170, 6), (150, 4), (130, 2)]  # min 10 balls

WICKET_POINTS = 30
FIVE_WICKET_BONUS = 8
FOUR_WICKET_BONUS = 4
THREE_WICKET_BONUS = 4
MAIDEN_OVER_BONUS = 4

ECONOMY_THRESHOLDS = [(6, 6), (7, 4), (8, 2)]  # min 2 overs

CATCH_POINTS = 8
STUMP_POINTS = 12
RUNOUT_DIRECT = 12
RUNOUT_ASSIST = 6

MILESTONE_BONUS = [(100, 16), (75, 8), (50, 4), (25, 2)]


def calculate_batting_points(row):
    # Convert string values to appropriate numeric types with robust error handling
    try:
        runs = int(float(row['Runs'])) if pd.notna(row['Runs']) else 0
    except (ValueError, TypeError):
        runs = 0
    
    try:
        balls = int(float(row['Balls Faced'])) if pd.notna(row['Balls Faced']) else 0
    except (ValueError, TypeError):
        balls = 0
        
    try:
        fours = int(float(row['Fours'])) if pd.notna(row['Fours']) else 0
    except (ValueError, TypeError):
        fours = 0
        
    try:
        sixes = int(float(row['Sixes'])) if pd.notna(row['Sixes']) else 0
    except (ValueError, TypeError):
        sixes = 0

    points = runs * BAT_RUN_POINTS
    for milestone, bonus in MILESTONE_BONUS:
        if runs >= milestone:
            points += bonus
            break
    
    if balls >= 10:
        sr = (runs / balls) * 100 if balls > 0 else 0
        for threshold, bonus in STRIKE_RATE_THRESHOLDS:
            if sr >= threshold:
                points += bonus
                break

    points += (fours * FOUR_BONUS) + (sixes * SIX_BONUS)
    return points


def calculate_bowling_points(row):
    # Convert string values to appropriate numeric types with robust error handling
    try:
        wickets = int(float(row['Wickets'])) if pd.notna(row['Wickets']) else 0
    except (ValueError, TypeError):
        wickets = 0
        
    try:
        overs = float(row['Overs']) if pd.notna(row['Overs']) else 0
    except (ValueError, TypeError):
        overs = 0
        
    try:
        maidens = int(float(row['Maidens'])) if pd.notna(row['Maidens']) else 0
    except (ValueError, TypeError):
        maidens = 0
        
    try:
        runs_given = int(float(row['Runs Given'])) if pd.notna(row['Runs Given']) else 0
    except (ValueError, TypeError):
        runs_given = 0

    points = wickets * WICKET_POINTS
    if wickets >= 5:
        points += FIVE_WICKET_BONUS
    elif wickets == 4:
        points += FOUR_WICKET_BONUS
    elif wickets == 3:
        points += THREE_WICKET_BONUS

    points += maidens * MAIDEN_OVER_BONUS

    if overs >= 2:
        eco = runs_given / overs
        for threshold, bonus in ECONOMY_THRESHOLDS:
            if eco <= threshold:
                points += bonus
                break

    return points


def calculate_fielding_points(row):
    # Convert string values to appropriate numeric types with robust error handling
    try:
        catches = int(float(row['Catches'])) if pd.notna(row['Catches']) else 0
    except (ValueError, TypeError):
        catches = 0
        
    try:
        stumpings = int(float(row['Stumpings'])) if pd.notna(row['Stumpings']) else 0
    except (ValueError, TypeError):
        stumpings = 0
        
    try:
        runouts_direct = int(float(row['Run Outs (Direct)'])) if pd.notna(row['Run Outs (Direct)']) else 0
    except (ValueError, TypeError):
        runouts_direct = 0
        
    try:
        runouts_assist = int(float(row['Run Outs (Assist)'])) if pd.notna(row['Run Outs (Assist)']) else 0
    except (ValueError, TypeError):
        runouts_assist = 0
    
    return (
        catches * CATCH_POINTS +
        stumpings * STUMP_POINTS +
        runouts_direct * RUNOUT_DIRECT +
        runouts_assist * RUNOUT_ASSIST
    )


def calculate_total_points(df):
    # Skip the header row if it's being treated as data
    if 'Runs' in df.columns and isinstance(df.iloc[0]['Runs'], str) and df.iloc[0]['Runs'].lower() == 'runs':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Ensure all columns exist
    required_columns = ['Runs', 'Balls Faced', 'Fours', 'Sixes', 'Wickets', 'Overs', 'Maidens', 
                       'Runs Given', 'Catches', 'Stumpings', 'Run Outs (Direct)', 'Run Outs (Assist)']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    df['Batting Points'] = df.apply(calculate_batting_points, axis=1)
    df['Bowling Points'] = df.apply(calculate_bowling_points, axis=1)
    df['Fielding Points'] = df.apply(calculate_fielding_points, axis=1)
    df['Total Points'] = df['Batting Points'] + df['Bowling Points'] + df['Fielding Points']
    
    # Return only the 'Total Points' column as a Series instead of the entire DataFrame
    return df['Total Points']


def generate_fantasy_points(input_path, output_path):
    df = pd.read_excel(input_path)
    df = calculate_total_points(df)
    df.to_csv(output_path, index=False)
    print(f"Fantasy points saved to {output_path}")