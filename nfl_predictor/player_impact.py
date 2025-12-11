"""
Player Impact Calculator - Determine injury impact on team performance.

Calculates how much each injured player affects their team's win probability
based on position importance and playing time.
"""

import pandas as pd
from typing import Dict, Optional

# Position impact weights (0.0-1.0 scale, QB = most valuable)
POSITION_IMPACT = {
    # Offense - most valuable
    'QB':  1.00,   # Quarterback is most impactful

    # Offensive Line (protect QB, enable run)
    'LT':  0.55,   # Left Tackle (protects blind side)
    'RT':  0.45,
    'C':   0.40,
    'LG':  0.35,
    'RG':  0.35,

    # Skill positions
    'WR':  0.40,   # Wide Receiver
    'TE':  0.35,   # Tight End
    'RB':  0.30,   # Running Back
    'FB':  0.15,   # Fullback

    # Defense - Edge rushers most valuable
    'EDGE': 0.50,  # Edge Rusher / OLB
    'DE':   0.45,  # Defensive End
    'DT':   0.35,  # Defensive Tackle
    'NT':   0.30,  # Nose Tackle

    # Secondary
    'CB':  0.45,   # Cornerback
    'S':   0.35,   # Safety

    # Linebackers
    'LB':  0.30,
    'ILB': 0.30,
    'MLB': 0.30,

    # Special Teams
    'K':   0.20,   # Kicker
    'P':   0.15,   # Punter
    'LS':  0.05,   # Long Snapper

    # Unknown/Default
    'UNK': 0.20
}

# Injury status multipliers (how much of the player's value is lost)
# NOTE: IR/PUP are set to 0 because these long-term injuries are already
# reflected in the team's EPA stats - we only want to capture NEW game-week impacts
STATUS_MULTIPLIER = {
    'Out':          1.00,   # 100% of value lost (not playing this week)
    'Doubtful':     0.85,   # 85% likely not playing
    'Questionable': 0.40,   # 40% chance of sitting
    'Probable':     0.10,   # Usually plays, minor impact
    'IR':           0.00,   # Already baked into team EPA (long-term)
    'PUP':          0.00,   # Already baked into team EPA (long-term)
    'Unknown':      0.30    # Conservative default
}

# Offensive vs Defensive positions
OFFENSIVE_POSITIONS = {'QB', 'RB', 'FB', 'WR', 'TE', 'LT', 'RT', 'LG', 'RG', 'C', 'K', 'P', 'LS'}
DEFENSIVE_POSITIONS = {'EDGE', 'DE', 'DT', 'NT', 'LB', 'ILB', 'MLB', 'CB', 'S'}


def normalize_name(name: str) -> str:
    """Normalize player name for matching (lowercase, remove suffixes)."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' ii', ' iii', ' iv']:
        name = name.replace(suffix, '')
    return name


def get_player_snap_share_by_name(
    player_name: str,
    team_abbrev: str,
    snap_counts_df: pd.DataFrame,
    position: str
) -> Optional[float]:
    """
    Look up player's snap share by name matching.
    
    Args:
        player_name: Player's full name from injury report
        team_abbrev: Team abbreviation (e.g., 'ARI', 'KC')
        snap_counts_df: Snap count DataFrame with 'player', 'team' columns
        position: Player's position
    
    Returns:
        Snap share (0.0-1.0) if found, None if no match
    """
    if snap_counts_df.empty or not player_name:
        return None
    
    # Filter to team's players only
    team_snaps = snap_counts_df[snap_counts_df['team'] == team_abbrev]
    if team_snaps.empty:
        return None
    
    # Normalize names for matching
    normalized_target = normalize_name(player_name)
    
    # Try exact match first
    for _, row in team_snaps.iterrows():
        if normalize_name(row['player']) == normalized_target:
            # Found match - get snap percentage
            if position in OFFENSIVE_POSITIONS:
                snap_pct = row.get('offense_pct', 0)
            else:
                snap_pct = row.get('defense_pct', 0)
            
            # Values are already 0-1 scale
            return min(1.0, max(0.0, float(snap_pct)))
    
    # Try partial match (first + last name)
    name_parts = normalized_target.split()
    if len(name_parts) >= 2:
        first_name = name_parts[0]
        last_name = name_parts[-1]
        
        for _, row in team_snaps.iterrows():
            row_parts = normalize_name(row['player']).split()
            if len(row_parts) >= 2:
                if row_parts[0] == first_name and row_parts[-1] == last_name:
                    if position in OFFENSIVE_POSITIONS:
                        snap_pct = row.get('offense_pct', 0)
                    else:
                        snap_pct = row.get('defense_pct', 0)
                    return min(1.0, max(0.0, float(snap_pct)))
    
    return None  # No match found


def get_player_snap_share(gsis_id: str, snap_counts_df: pd.DataFrame, position: str) -> float:
    """
    Calculate player's snap share by GSIS ID (0.0-1.0).
    
    NOTE: This is the ID-based version. For name-based matching,
    use get_player_snap_share_by_name() instead.

    Args:
        gsis_id: Player's GSIS ID
        snap_counts_df: Snap count DataFrame
        position: Player's position

    Returns:
        Snap share (0.0 = backup, 1.0 = every-down player)
    """
    if snap_counts_df.empty:
        # No snap data available - assume starter if reported as injured
        return 0.8

    # Check if gsis_id column exists in snap counts
    if 'gsis_id' not in snap_counts_df.columns:
        # Column doesn't exist, return sentinel value to trigger name-based fallback
        return 0.3

    # Find player's snap data
    player_snaps = snap_counts_df[snap_counts_df['gsis_id'] == gsis_id]

    if player_snaps.empty:
        # Player not in snap data - might be backup or data issue
        return 0.3  # Conservative estimate

    # Get the most recent week's data
    latest_week = player_snaps['week'].max()
    recent_snaps = player_snaps[player_snaps['week'] == latest_week]

    if recent_snaps.empty:
        return 0.3

    # Calculate snap percentage based on position
    snap_pct = 0.0

    if position in OFFENSIVE_POSITIONS:
        # Offensive snap percentage
        if 'offense_pct' in recent_snaps.columns:
            snap_pct = recent_snaps['offense_pct'].iloc[0]  # Already 0-1 scale
        elif 'offense_snaps' in recent_snaps.columns:
            # Estimate percentage (rough calculation)
            offense_snaps = recent_snaps['offense_snaps'].iloc[0]
            snap_pct = min(1.0, offense_snaps / 70.0)  # Assume ~70 offensive snaps per game
    else:
        # Defensive snap percentage
        if 'defense_pct' in recent_snaps.columns:
            snap_pct = recent_snaps['defense_pct'].iloc[0]  # Already 0-1 scale
        elif 'defense_snaps' in recent_snaps.columns:
            defense_snaps = recent_snaps['defense_snaps'].iloc[0]
            snap_pct = min(1.0, defense_snaps / 70.0)  # Assume ~70 defensive snaps per game

    # Ensure valid range
    snap_pct = max(0.0, min(1.0, snap_pct))

    # If no snap data, assume starter status for injured players
    if snap_pct == 0.0:
        snap_pct = 0.8

    return snap_pct

def get_player_value(position: str, snap_share: float) -> float:
    """
    Calculate player's value to their team (0.0-1.0 scale).

    Args:
        position: Player's position
        snap_share: Player's snap share (0.0-1.0)

    Returns:
        Player value multiplier (0.0-1.0)
    """
    # Base value from position
    base_value = POSITION_IMPACT.get(position, 0.20)

    # Adjust for playing time (starter vs backup)
    # - 0.90+ snap share = 1.0x (primary starter)
    # - 0.50 snap share = 0.6x (rotational)
    # - 0.20 snap share = 0.3x (backup)
    starter_multiplier = min(1.0, snap_share * 1.2)

    return base_value * starter_multiplier

def calculate_injury_impact(
    player_row: pd.Series,
    snap_counts_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate the impact of a single injured player.

    Args:
        player_row: Row from injuries DataFrame
        snap_counts_df: Snap count DataFrame

    Returns:
        Dict with 'offensive_impact' and 'defensive_impact' (0.0-1.0)
    """
    position = player_row.get('position', 'UNK')
    status = player_row.get('report_status', 'Unknown')
    gsis_id = player_row.get('gsis_id', None)
    player_name = player_row.get('full_name', '')
    team = player_row.get('team', '')

    # Get player's snap share using multiple methods
    snap_share = None
    
    # Method 1: Try GSIS ID lookup (most reliable)
    if gsis_id and snap_share is None:
        snap_share = get_player_snap_share(gsis_id, snap_counts_df, position)
        # Only accept if it's not the default fallback value
        if snap_share == 0.3:
            snap_share = None  # ID not found, try other methods
    
    # Method 2: Try name-based matching (for ESPN scraped data)
    if snap_share is None and player_name and team:
        snap_share = get_player_snap_share_by_name(player_name, team, snap_counts_df, position)
    
    # Method 3: Fall back to position-based defaults
    if snap_share is None:
        # Conservative defaults when no snap data available
        if position == 'QB':
            snap_share = 0.9  # QB on injury report is likely the starter
        elif position in {'LT', 'WR', 'EDGE', 'CB'}:
            snap_share = 0.5  # Key position but could be backup
        else:
            snap_share = 0.3  # Conservative - may not be a starter

    # Get player's base value
    player_value = get_player_value(position, snap_share)

    # Apply injury status multiplier
    status_factor = STATUS_MULTIPLIER.get(status, 0.5)
    injury_impact = player_value * status_factor

    # Determine if offensive or defensive impact
    if position in OFFENSIVE_POSITIONS:
        return {
            'offensive_impact': injury_impact,
            'defensive_impact': 0.0
        }
    elif position in DEFENSIVE_POSITIONS:
        return {
            'offensive_impact': 0.0,
            'defensive_impact': injury_impact
        }
    else:
        # Special teams or unknown - split impact
        return {
            'offensive_impact': injury_impact * 0.6,
            'defensive_impact': injury_impact * 0.4
        }

def calculate_team_injury_impact(
    team: str,
    injuries_df: pd.DataFrame,
    snap_counts_df: pd.DataFrame,
    week: int
) -> Dict[str, float]:
    """
    Calculate total injury impact on a team's performance.

    Args:
        team: Team name
        injuries_df: Full injury DataFrame
        snap_counts_df: Snap count DataFrame
        week: NFL week

    Returns:
        Dict with 'offensive_impact', 'defensive_impact', 'total_impact'
    """
    # Filter injuries for this team and week
    team_injuries = injuries_df[
        (injuries_df['team'] == team) &
        (injuries_df['week'] == week) &
        (injuries_df['game_type'] == 'REG')
    ]

    if team_injuries.empty:
        return {
            'offensive_impact': 0.0,
            'defensive_impact': 0.0,
            'total_impact': 0.0
        }

    total_offensive = 0.0
    total_defensive = 0.0

    # Calculate impact for each injured player
    for _, player in team_injuries.iterrows():
        impact = calculate_injury_impact(player, snap_counts_df)
        total_offensive += impact['offensive_impact']
        total_defensive += impact['defensive_impact']

    # Cap at reasonable maximums (losing whole offense unlikely)
    total_offensive = min(total_offensive, 0.60)  # Max 60% offensive impact
    total_defensive = min(total_defensive, 0.60)  # Max 60% defensive impact

    total_impact = total_offensive + total_defensive

    return {
        'offensive_impact': total_offensive,
        'defensive_impact': total_defensive,
        'total_impact': total_impact
    }

def get_all_team_impacts(
    injuries_df: pd.DataFrame,
    snap_counts_df: pd.DataFrame,
    week: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate injury impacts for all teams.

    Args:
        injuries_df: Full injury DataFrame
        snap_counts_df: Snap count DataFrame
        week: NFL week

    Returns:
        Dict of team -> impact dict
    """
    # Get all teams with injuries
    injured_teams = injuries_df[
        (injuries_df['week'] == week) &
        (injuries_df['game_type'] == 'REG')
    ]['team'].unique()

    impacts = {}
    for team in injured_teams:
        impacts[team] = calculate_team_injury_impact(team, injuries_df, snap_counts_df, week)

    return impacts

def print_team_impact_summary(
    team_impacts: Dict[str, Dict[str, float]],
    injuries_df: pd.DataFrame,
    week: int
):
    """Print a summary of injury impacts by team."""
    if not team_impacts:
        print("✅ No significant injury impacts this week")
        return

    print(f"\n📊 Injury Impact Summary - Week {week}")
    print("=" * 60)

    # Sort by total impact
    sorted_teams = sorted(
        team_impacts.items(),
        key=lambda x: x[1]['total_impact'],
        reverse=True
    )

    for team, impact in sorted_teams:
        if impact['total_impact'] < 0.01:  # Skip negligible impacts
            continue

        off_pct = impact['offensive_impact'] * 100
        def_pct = impact['defensive_impact'] * 100
        total_pct = impact['total_impact'] * 100

        print(f"\n{team}:")
        print(f"  Offensive Impact: {off_pct:.1f}%")
        print(f"  Defensive Impact: {def_pct:.1f}%")
        print(f"  Total Impact:     {total_pct:.1f}%")

        # Show key injuries
        team_injuries = injuries_df[
            (injuries_df['team'] == team) &
            (injuries_df['week'] == week) &
            (injuries_df['game_type'] == 'REG')
        ]

        for _, injury in team_injuries.iterrows():
            status = injury['report_status']
            player = f"{injury['full_name']} ({injury['position']})"
            injury_desc = injury.get('report_primary_injury', 'Unknown')

            status_icon = {
                'Out': '❌',
                'Doubtful': '⚠️',
                'Questionable': '❓',
                'Probable': '✅'
            }.get(status, '❓')

            print(f"  {status_icon} {player} - {status} ({injury_desc})")

if __name__ == "__main__":
    try:
        from .injuries import load_injury_data, load_snap_counts, get_current_nfl_week
    except ImportError:
        # Just for type hinting/IDE
        import pandas as pd
        def load_injury_data(*args, **kwargs): return {}
        def load_snap_counts(*args, **kwargs): return pd.DataFrame()
        def get_current_nfl_week(*args, **kwargs): return 1 # Default to week 1 if not found

    # Load data
    injuries_df = load_injury_data(2025)
    snap_counts_df = load_snap_counts(2025)
    current_week = get_current_nfl_week()

    # Calculate impacts
    impacts = get_all_team_impacts(injuries_df, snap_counts_df, current_week)

    # Print summary
    print_team_impact_summary(impacts, injuries_df, current_week)

    # Show position weights
    print(f"\n🏈 Position Impact Weights:")
    print("-" * 40)
    for pos, weight in sorted(POSITION_IMPACT.items(), key=lambda x: x[1], reverse=True):
        print(f"{pos}: {weight:.2f}")
