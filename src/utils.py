def normalize_name(name):
    """
    Standardize player names (useful for merging datasets).
    """
    return name.strip().lower().replace(" ", "_")


def get_most_recent_match(df, match_no):
    """
    Filter and return only data from matches before the given match number.
    """
    return df[df['Match Number'] < match_no]
