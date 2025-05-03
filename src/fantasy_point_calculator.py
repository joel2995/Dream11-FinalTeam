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
    runs = row['Runs']
    balls = row['Balls Faced']
    fours = row['Fours']
    sixes = row['Sixes']

    points = runs * BAT_RUN_POINTS
    for milestone, bonus in MILESTONE_BONUS:
        if runs >= milestone:
            points += bonus
            break
    
    if balls >= 10:
        sr = (runs / balls) * 100
        for threshold, bonus in STRIKE_RATE_THRESHOLDS:
            if sr >= threshold:
                points += bonus
                break

    points += (fours * FOUR_BONUS) + (sixes * SIX_BONUS)
    return points


def calculate_bowling_points(row):
    wickets = row['Wickets']
    overs = row['Overs']
    maidens = row['Maidens']
    runs_given = row['Runs Given']

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
    return (
        row['Catches'] * CATCH_POINTS +
        row['Stumpings'] * STUMP_POINTS +
        row['Run Outs (Direct)'] * RUNOUT_DIRECT +
        row['Run Outs (Assist)'] * RUNOUT_ASSIST
    )


def calculate_total_points(df):
    df['Batting Points'] = df.apply(calculate_batting_points, axis=1)
    df['Bowling Points'] = df.apply(calculate_bowling_points, axis=1)
    df['Fielding Points'] = df.apply(calculate_fielding_points, axis=1)
    df['Total Points'] = df['Batting Points'] + df['Bowling Points'] + df['Fielding Points']
    return df


def generate_fantasy_points(input_path, output_path):
    df = pd.read_excel(input_path)
    df = calculate_total_points(df)
    df.to_csv(output_path, index=False)
    print(f"Fantasy points saved to {output_path}")