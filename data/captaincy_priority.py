# Captain and Vice-Captain Priority Data
# This file contains the captaincy and vice-captaincy priorities for players based on user observations

def get_captaincy_priority_data():
    """
    Returns a dictionary mapping players to their teams and captaincy priorities.
    Priority 1 indicates primary captain material, Priority 2 indicates vice-captain material.
    
    This data is based on user observations of IPL matches and player performances.
    """
    return {
        # CSK players
        "Ravindra Jadeja": {"team": "CSK", "priority": 1},
        "Ayush Mhatre": {"team": "CSK", "priority": 2},
        
        # RCB players
        "Virat Kohli": {"team": "RCB", "priority": 1},
        "Philip Salt": {"team": "RCB", "priority": 2},
        
        # SRH players
        "Heinrich Klaasen": {"team": "SRH", "priority": 2},
        "Abhishek Sharma": {"team": "SRH", "priority": 1},
        
        # GT players
        "Jos Buttler": {"team": "GT", "priority": 1},
        "Sai Sudharsan": {"team": "GT", "priority": 1},
        "Shubman Gill": {"team": "GT", "priority": 1},
        
        # MI players
        "Suryakumar Yadav": {"team": "MI", "priority": 1},
        "Ryan Rickelton": {"team": "MI", "priority": 2},
        "Hardik Pandya": {"team": "MI", "priority": 2},
        
        # PBKS players
        "Shreyas Iyer": {"team": "PBKS", "priority": 1},
        "Priyansh Arya": {"team": "PBKS", "priority": 2},
        
        # DC players
        "Axar Patel": {"team": "DC", "priority": 1},
        "Lokesh Rahul": {"team": "DC", "priority": 1},
        
        # KKR players
        "Sunil Narine": {"team": "KKR", "priority": 1},
        
        # RR players
        "Yashasvi Jaiswal": {"team": "RR", "priority": 1},
        
        # LSG players
        "Aiden Markram": {"team": "LSG", "priority": 1},
        "Nicholas Pooran": {"team": "LSG", "priority": 2},
        "Mitchell Marsh": {"team": "LSG", "priority": 2}
    }

def get_captain_vice_captain_for_match(home_team, away_team):
    """
    Determines the best captain and vice-captain candidates for a match between two teams.
    Returns a tuple of (captain_candidates, vice_captain_candidates) where each is a list of player names.
    
    Args:
        home_team (str): The home team code (e.g., 'CSK', 'MI')
        away_team (str): The away team code (e.g., 'RCB', 'GT')
        
    Returns:
        tuple: (captain_candidates, vice_captain_candidates)
    """
    captaincy_data = get_captaincy_priority_data()
    
    # Get captain candidates (priority 1) from both teams
    captain_candidates = [player for player, data in captaincy_data.items() 
                         if data["team"] in [home_team, away_team] and data["priority"] == 1]
    
    # Get vice-captain candidates (priority 2) from both teams
    vice_captain_candidates = [player for player, data in captaincy_data.items() 
                              if data["team"] in [home_team, away_team] and data["priority"] == 2]
    
    return captain_candidates, vice_captain_candidates