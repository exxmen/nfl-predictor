"""
Canonical NFL team-name mappings.

Single source of truth for team abbreviation <-> full-name conversion.
nfl_data_py / nflverse key teams by ABBREVIATION (KC, PHI), while the
simulation engine and tiebreakers key teams by FULL NAME
(Kansas City Chiefs). The EPA loader and the scheduler/scraper therefore
produce mismatched keys; every consumer should resolve through this module
so a lookup can never silently miss again.
"""

# Abbreviation -> full name (32 teams)
ABBREV_TO_FULL = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LA': 'Los Angeles Rams', 'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}

# Full name -> abbreviation (auto-derived, safe)
FULL_TO_ABBREV = {full: abbr for abbr, full in ABBREV_TO_FULL.items()}


def to_full_name(team: str) -> str:
    """Return the canonical full name for a team key (abbrev or full)."""
    if team in ABBREV_TO_FULL:
        return ABBREV_TO_FULL[team]
    if team in FULL_TO_ABBREV:
        return team
    return team


def to_abbreviation(team: str) -> str:
    """Return the abbreviation for a team key (abbrev or full)."""
    if team in FULL_TO_ABBREV:
        return FULL_TO_ABBREV[team]
    if team in ABBREV_TO_FULL:
        return team
    return team
