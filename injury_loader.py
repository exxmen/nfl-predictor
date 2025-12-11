"""
Injury Data Loader - Fetch and cache NFL injury reports.

Integrates player injuries into playoff predictions by adjusting team EPA
based on injured players' positions and playing time.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import re

# Cache directory
CACHE_DIR = Path("data")

def get_cache_files(season: int) -> tuple:
    """Get cache file paths for a specific season."""
    return (
        CACHE_DIR / f"injuries_{season}.parquet",
        CACHE_DIR / f"snap_counts_{season}.parquet",
        CACHE_DIR / f"injuries_{season}_meta.json"
    )

# NFL week start dates for 2025 (for cache invalidation)
NFL_2025_WEEK_STARTS = {
    1: "2025-09-04", 2: "2025-09-11", 3: "2025-09-18", 4: "2025-09-25",
    5: "2025-10-02", 6: "2025-10-09", 7: "2025-10-16", 8: "2025-10-23",
    9: "2025-10-30", 10: "2025-11-06", 11: "2025-11-13", 12: "2025-11-20",
    13: "2025-11-27", 14: "2025-12-04", 15: "2025-12-11", 16: "2025-12-18",
    17: "2025-12-25", 18: "2026-01-03"
}

def get_current_nfl_week() -> int:
    """Determine the current NFL week based on date."""
    today = datetime.now().strftime("%Y-%m-%d")
    current_week = 1
    for week, start_date in NFL_2025_WEEK_STARTS.items():
        if today >= start_date:
            current_week = week
    return current_week

def is_cache_valid(season: int) -> bool:
    """Check if injury cache is still valid for the given season."""
    injuries_cache, snap_cache, meta_file = get_cache_files(season)

    if not meta_file.exists():
        return False

    try:
        with open(meta_file) as f:
            meta = json.load(f)

        cached_season = meta.get("season", 0)

        # For past seasons, cache is always valid (data won't change)
        current_year = datetime.now().year
        if season < current_year:
            return cached_season == season

        # For current season, check week-based invalidation
        cached_week = meta.get("week", 0)
        current_week = get_current_nfl_week()

        # Cache is valid if we're still in the same week
        return cached_week >= current_week
    except:
        return False

def scrape_espn_injuries(season: int = 2025) -> pd.DataFrame:
    """
    Scrape current NFL injury data from ESPN.com.

    Returns DataFrame with columns similar to nflverse format:
    - season, game_type, team, week, full_name, position, report_status, report_primary_injury

    Args:
        season: NFL season year

    Returns:
        DataFrame with injury data
    """
    url = "https://www.espn.com/nfl/injuries"
    print(f"🌐 Scraping injury data from ESPN...")

    # Map full team names to abbreviations
    team_mapping = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        injuries_data = []
        current_week = get_current_nfl_week()

        # Find all team injury table containers (one per team)
        team_divs = soup.find_all('div', class_='Table__league-injuries')
        print(f"Found {len(team_divs)} team sections")

        # Teams are listed in alphabetical order on ESPN
        team_order = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
        ]

        for idx, team_div in enumerate(team_divs):
            # Use positional index to determine team (ESPN lists alphabetically)
            if idx >= len(team_order):
                continue  # Skip if more divs than expected teams

            team_abbrev = team_order[idx]

            # Find the table within this div
            table = team_div.find('table')
            if not table:
                continue

            # Process table rows (skip header row)
            rows = table.find_all('tr')[1:]

            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 4:
                    continue

                # Extract player data
                # Column order: NAME, POS, EST. RETURN DATE, STATUS, COMMENT
                player_cell = cols[0]
                player_link = player_cell.find('a')
                if not player_link:
                    continue

                full_name = player_link.get_text().strip()
                position = cols[1].get_text().strip() if len(cols) > 1 else 'UNK'
                return_date = cols[2].get_text().strip() if len(cols) > 2 else ''
                status = cols[3].get_text().strip() if len(cols) > 3 else 'Questionable'
                comment = cols[4].get_text().strip() if len(cols) > 4 else ''

                # Map ESPN status to our format
                status_mapping = {
                    'Out': 'Out',
                    'Doubtful': 'Doubtful',
                    'Questionable': 'Questionable',
                    'Injured Reserve': 'IR',
                    'Physically Unable to Perform': 'PUP'
                }
                report_status = status_mapping.get(status, status)  # Keep original if not mapped

                # Extract primary injury from comment
                primary_injury = "Unknown"
                if comment:
                    # Look for injury patterns in comments like "(knee)" or "knee injury"
                    injury_patterns = [
                        r'\(([^)]+)\)',  # (knee)
                        r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:injury|surgery)',  # "Knee injury"
                        r'(?:suffering|sustained|underwent)\s+([^,\.]+)',  # "suffering from knee injury"
                    ]
                    for pattern in injury_patterns:
                        match = re.search(pattern, comment, re.IGNORECASE)
                        if match:
                            primary_injury = match.group(1).strip()
                            break

                # Create injury record
                injury_record = {
                    'season': season,
                    'game_type': 'REG',
                    'team': team_abbrev,
                    'week': current_week,
                    'full_name': full_name,
                    'position': position,
                    'report_status': report_status,
                    'report_primary_injury': primary_injury,
                    'report_secondary_injury': None,
                    'practice_status': report_status,
                    'practice_primary_injury': primary_injury,
                    'practice_secondary_injury': None,
                    'date_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'ESPN'
                }

                injuries_data.append(injury_record)

        if injuries_data:
            df = pd.DataFrame(injuries_data)
            unique_teams = df['team'].nunique()
            print(f"✅ Scraped {len(df)} injury records from ESPN ({unique_teams} teams)")
            return df
        else:
            print("⚠️ No injury data found on ESPN")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Failed to scrape ESPN injuries: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_injury_data(season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load injury data with fallback hierarchy: ESPN → nfl_data_py → cache.

    Returns DataFrame with columns:
    - season, game_type, team, week, gsis_id, position, full_name
    - report_primary_injury, report_secondary_injury, report_status
    - practice_primary_injury, practice_secondary_injury, practice_status
    - date_modified

    Args:
        season: NFL season year
        force_refresh: Force re-fetch even if cache is valid

    Returns:
        DataFrame with injury data
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Get season-specific cache files
    injuries_cache, snap_cache, meta_file = get_cache_files(season)

    # Check cache
    if not force_refresh and injuries_cache.exists() and is_cache_valid(season):
        print(f"🏥 Loading {season} injury data from cache...")
        return pd.read_parquet(injuries_cache)

    # Try ESPN scraper first (for current season)
    injuries_df = pd.DataFrame()
    if season >= datetime.now().year:
        print(f"🌐 Trying ESPN scraper for {season}...")
        injuries_df = scrape_espn_injuries(season)

    # Fallback to nfl_data_py if ESPN failed or for past seasons
    if injuries_df.empty:
        print(f"🏥 Falling back to nfl_data_py for {season}...")
        try:
            injuries_df = nfl.import_injuries([season])
            print(f"✅ Loaded {len(injuries_df):,} injury records from nfl_data_py")
        except Exception as e:
            print(f"❌ Failed to fetch injury data from nfl_data_py: {e}")
            # Try to return stale cache if available
            if injuries_cache.exists():
                print("⚠️  Returning stale injury cache...")
                return pd.read_parquet(injuries_cache)
            print("❌ No injury data available, returning empty DataFrame")
            return pd.DataFrame()

    # Clean and standardize data
    injuries_df = injuries_df.copy()

    # Ensure position column exists and is clean
    if 'position' not in injuries_df.columns:
        print("⚠️  Position column missing from injury data")
        injuries_df['position'] = 'UNK'

    # Standardize position abbreviations
    position_mapping = {
        'QB': 'QB', 'RB': 'RB', 'FB': 'FB', 'WR': 'WR', 'TE': 'TE',
        'LT': 'LT', 'RT': 'RT', 'LG': 'LG', 'RG': 'RG', 'C': 'C',
        'LE': 'EDGE', 'RE': 'EDGE', 'DT': 'DT', 'NT': 'NT', 'DE': 'DE',
        'OLB': 'EDGE', 'ILB': 'LB', 'MLB': 'LB', 'LB': 'LB',
        'CB': 'CB', 'S': 'S', 'FS': 'S', 'SS': 'S',
        'K': 'K', 'P': 'P', 'LS': 'LS'
    }
    injuries_df['position'] = injuries_df['position'].map(position_mapping).fillna('UNK')

    # Ensure report_status is standardized
    status_mapping = {
        'Out': 'Out',
        'Doubtful': 'Doubtful',
        'Questionable': 'Questionable',
        'Probable': 'Probable',
        'IR': 'IR',
        'PUP': 'PUP',
        'DNP': 'Out',  # Did Not Participate = Out
        'Limited': 'Questionable',
        'Full': 'Probable'
    }
    injuries_df['report_status'] = injuries_df['report_status'].map(status_mapping).fillna('Unknown')

    # Save cache
    injuries_df.to_parquet(injuries_cache, index=False)

    # Save metadata
    meta = {
        "season": season,
        "week": get_current_nfl_week() if season >= datetime.now().year else 18,
        "updated": datetime.now().isoformat(),
        "records": len(injuries_df),
        "teams": injuries_df['team'].nunique(),
        "source": injuries_df['source'].iloc[0] if 'source' in injuries_df.columns and not injuries_df.empty else 'nfl_data_py'
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Cached {len(injuries_df)} injury records for {season}")

    return injuries_df

def load_snap_counts(season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load snap count data to determine player importance.

    Returns DataFrame with columns:
    - player, team, position, week, offense_snaps, offense_pct, etc.

    Args:
        season: NFL season year
        force_refresh: Force re-fetch even if cache is valid
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Get season-specific cache files
    injuries_cache, snap_cache, meta_file = get_cache_files(season)

    # Check cache
    if not force_refresh and snap_cache.exists() and is_cache_valid(season):
        print(f"📊 Loading {season} snap count data from cache...")
        return pd.read_parquet(snap_cache)

    # Fetch fresh data
    print(f"📊 Fetching {season} NFL snap counts...")
    try:
        snap_df = nfl.import_snap_counts([season])
        print(f"✅ Loaded {len(snap_df):,} snap count records for {season}")
    except Exception as e:
        print(f"❌ Failed to fetch snap count data: {e}")
        # Try to return stale cache if available
        if snap_cache.exists():
            print("⚠️  Returning stale snap count cache...")
            return pd.read_parquet(snap_cache)
        raise

    # Clean data
    snap_df = snap_df.copy()

    # Ensure we have the key columns
    required_cols = ['player', 'team', 'position', 'week']
    if not all(col in snap_df.columns for col in required_cols):
        print(f"⚠️  Missing required columns in snap data: {required_cols}")
        return pd.DataFrame()

    # Save cache
    snap_df.to_parquet(snap_cache, index=False)

    print(f"💾 Cached {len(snap_df)} snap count records for {season}")

    return snap_df

def get_current_week_injuries(injuries_df: pd.DataFrame, week: int = None) -> pd.DataFrame:
    """
    Get injuries for the current/specified week.

    Args:
        injuries_df: Full injury DataFrame
        week: Specific week (default: current NFL week)

    Returns:
        Filtered DataFrame with current week injuries
    """
    if week is None:
        week = get_current_nfl_week()

    # Filter to current week and regular season
    current_injuries = injuries_df[
        (injuries_df['week'] == week) &
        (injuries_df['game_type'] == 'REG')
    ].copy()

    return current_injuries

def print_injury_summary(injuries_df: pd.DataFrame, week: int = None):
    """Print a summary of current injuries by team."""
    current_injuries = get_current_week_injuries(injuries_df, week)

    if current_injuries.empty:
        print("✅ No injuries reported for current week")
        return

    print(f"\n🏥 NFL Injury Report - Week {week or get_current_nfl_week()}")
    print("=" * 60)

    # Group by team
    team_injuries = {}
    for _, injury in current_injuries.iterrows():
        team = injury['team']
        if team not in team_injuries:
            team_injuries[team] = []

        status = injury['report_status']
        player = f"{injury['full_name']} ({injury['position']})"
        injury_desc = injury.get('report_primary_injury', 'Unknown')

        team_injuries[team].append({
            'player': player,
            'status': status,
            'injury': injury_desc
        })

    # Print by team
    for team in sorted(team_injuries.keys()):
        injuries = team_injuries[team]
        print(f"\n{team}:")
        for injury in injuries:
            status_icon = {
                'Out': '❌',
                'Doubtful': '⚠️',
                'Questionable': '❓',
                'Probable': '✅',
                'IR': '🚫',
                'PUP': '🏥'
            }.get(injury['status'], '❓')

            print(f"  {status_icon} {injury['player']} - {injury['status']} ({injury['injury']})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load NFL injury data")
    parser.add_argument("--season", type=int, default=2025, help="NFL season year")
    parser.add_argument("--force", action="store_true", help="Force refresh from API")
    parser.add_argument("--summary", action="store_true", help="Show injury summary")
    args = parser.parse_args()

    # Load injury data
    injuries_df = load_injury_data(season=args.season, force_refresh=args.force)

    if args.summary:
        print_injury_summary(injuries_df)

    print(f"\n📊 Injury data shape: {injuries_df.shape}")
    print(f"📅 Weeks covered: {injuries_df['week'].min()} - {injuries_df['week'].max()}")
    print(f"🏈 Teams with injuries: {injuries_df['team'].nunique()}")

    # Show sample
    if not injuries_df.empty:
        print("\n📋 Sample injury records:")
        sample_cols = ['team', 'week', 'full_name', 'position', 'report_status', 'report_primary_injury']
        print(injuries_df[sample_cols].head().to_string(index=False))
