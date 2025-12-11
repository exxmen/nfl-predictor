"""
EPA Loader - Fetch team-level EPA stats from nfl_data_py.

EPA (Expected Points Added) adjusts for game situation (down, distance, field position)
and is the gold standard for NFL analytics. This provides more accurate inputs for
game simulation than raw scoring averages.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


# Cache directory
CACHE_DIR = Path("data")

def get_cache_files(season: int) -> tuple:
    """Get cache file paths for a specific season."""
    return (
        CACHE_DIR / f"team_epa_{season}.parquet",
        CACHE_DIR / f"team_epa_{season}_meta.json"
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
    """Check if EPA cache is still valid for the given season."""
    cache_file, meta_file = get_cache_files(season)
    
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


def calculate_team_momentum(pbp: pd.DataFrame, n_recent_games: int = 4) -> pd.DataFrame:
    """
    Calculate team momentum based on recent performance vs season average.
    
    Momentum is a z-score indicating how much better/worse a team is performing
    recently compared to their season baseline.
    
    Args:
        pbp: Play-by-play DataFrame
        n_recent_games: Number of recent games to use for "recent form" (default: 4)
    
    Returns:
        DataFrame with columns:
        - team: Team abbreviation
        - recent_off_epa: Offensive EPA over last N games
        - recent_def_epa: Defensive EPA over last N games
        - off_momentum: Z-score for offensive momentum
        - def_momentum: Z-score for defensive momentum
        - total_momentum: Combined momentum score
    """
    # Filter to real plays
    real_plays = pbp[
        (pbp['play_type'].isin(['pass', 'run', 'qb_kneel', 'qb_spike'])) &
        (pbp['epa'].notna()) &
        (pbp['week'].notna())
    ].copy()
    
    if real_plays.empty:
        return pd.DataFrame()
    
    # Get unique games per team to determine recent games
    games = pbp[pbp['game_id'].notna()][['game_id', 'week', 'home_team', 'away_team']].drop_duplicates()
    
    # Build team-game mapping
    team_games = []
    for _, game in games.iterrows():
        team_games.append({'team': game['home_team'], 'game_id': game['game_id'], 'week': game['week']})
        team_games.append({'team': game['away_team'], 'game_id': game['game_id'], 'week': game['week']})
    
    team_games_df = pd.DataFrame(team_games).drop_duplicates()
    
    # For each team, identify their most recent N games
    recent_game_ids = {}
    for team in team_games_df['team'].unique():
        if pd.isna(team) or team == '':
            continue
        team_schedule = team_games_df[team_games_df['team'] == team].sort_values('week', ascending=False)
        recent_game_ids[team] = set(team_schedule.head(n_recent_games)['game_id'].tolist())
    
    # Calculate season-wide EPA per team (offense)
    season_off_epa = real_plays.groupby('posteam')['epa'].agg(['mean', 'std']).rename(
        columns={'mean': 'season_off_epa', 'std': 'season_off_std'}
    )
    
    # Calculate season-wide EPA per team (defense) - flip sign
    season_def_epa = real_plays.groupby('defteam')['epa'].agg(['mean', 'std'])
    season_def_epa['season_def_epa'] = -season_def_epa['mean']
    season_def_epa['season_def_std'] = season_def_epa['std']
    season_def_epa = season_def_epa[['season_def_epa', 'season_def_std']]
    
    # Calculate recent EPA for each team
    momentum_data = []
    
    for team, game_ids in recent_game_ids.items():
        if not game_ids:
            continue
        
        # Recent offensive plays
        recent_off_plays = real_plays[
            (real_plays['posteam'] == team) & 
            (real_plays['game_id'].isin(game_ids))
        ]
        
        # Recent defensive plays
        recent_def_plays = real_plays[
            (real_plays['defteam'] == team) & 
            (real_plays['game_id'].isin(game_ids))
        ]
        
        if recent_off_plays.empty and recent_def_plays.empty:
            continue
        
        # Calculate recent EPA
        recent_off_epa = recent_off_plays['epa'].mean() if not recent_off_plays.empty else 0
        recent_def_epa = -recent_def_plays['epa'].mean() if not recent_def_plays.empty else 0
        
        # Get season stats for this team
        season_off = season_off_epa.loc[team] if team in season_off_epa.index else {'season_off_epa': 0, 'season_off_std': 0.1}
        season_def = season_def_epa.loc[team] if team in season_def_epa.index else {'season_def_epa': 0, 'season_def_std': 0.1}
        
        # Calculate z-scores (momentum)
        off_std = season_off['season_off_std'] if season_off['season_off_std'] > 0.01 else 0.1
        def_std = season_def['season_def_std'] if season_def['season_def_std'] > 0.01 else 0.1
        
        off_momentum = (recent_off_epa - season_off['season_off_epa']) / off_std
        def_momentum = (recent_def_epa - season_def['season_def_epa']) / def_std
        
        # Clip extreme values to avoid outliers dominating
        off_momentum = max(-3.0, min(3.0, off_momentum))
        def_momentum = max(-3.0, min(3.0, def_momentum))
        
        momentum_data.append({
            'team': team,
            'recent_off_epa': round(recent_off_epa, 4),
            'recent_def_epa': round(recent_def_epa, 4),
            'off_momentum': round(off_momentum, 3),
            'def_momentum': round(def_momentum, 3),
            'total_momentum': round((off_momentum + def_momentum) / 2, 3),
            'recent_games': len(game_ids)
        })
    
    return pd.DataFrame(momentum_data)


def load_team_epa(season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load team-level EPA stats from play-by-play data.
    
    Returns DataFrame with columns:
    - team: Team abbreviation (KC, PHI, etc.)
    - off_epa: Offensive EPA per play (higher = better offense)
    - def_epa: Defensive EPA per play (higher = better defense, flipped sign)
    - total_epa: Combined EPA rating
    - off_plays: Number of offensive plays
    - def_plays: Number of defensive plays
    - ppg: Points per game (from PBP)
    - ppg_allowed: Points allowed per game
    
    Args:
        season: NFL season year
        force_refresh: Force re-fetch even if cache is valid
    
    Returns:
        DataFrame with team EPA stats
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Get season-specific cache files
    cache_file, meta_file = get_cache_files(season)
    
    # Check cache
    if not force_refresh and cache_file.exists() and is_cache_valid(season):
        print(f"📊 Loading {season} EPA data from cache...")
        return pd.read_parquet(cache_file)
    
    # Try to fetch the requested season, fall back to most recent available
    seasons_to_try = [season, season - 1, 2024, 2023]
    pbp = None
    actual_season = season
    
    for try_season in seasons_to_try:
        try:
            print(f"🏈 Fetching {try_season} NFL play-by-play data (this may take 2-3 minutes)...")
            pbp = nfl.import_pbp_data([try_season])
            actual_season = try_season
            print(f"✅ Loaded {len(pbp):,} plays from {try_season}")
            break
        except Exception as e:
            if try_season == season:
                print(f"⚠️  {try_season} data not available, trying previous season...")
            continue
    
    if pbp is None:
        print(f"❌ Failed to fetch PBP data for any season")
        # Try to return stale cache if available
        if cache_file.exists():
            print("⚠️  Returning stale cache...")
            return pd.read_parquet(cache_file)
        raise RuntimeError("No PBP data available and no cache exists")
    
    # Filter to real plays (exclude penalties, timeouts, etc.)
    real_plays = pbp[
        (pbp['play_type'].isin(['pass', 'run', 'qb_kneel', 'qb_spike'])) &
        (pbp['epa'].notna())
    ].copy()
    
    print(f"📈 Analyzing {len(real_plays):,} real plays...")
    
    # Offensive EPA (when team has the ball)
    off_stats = real_plays.groupby('posteam').agg({
        'epa': 'mean',
        'play_id': 'count',
        'posteam_score_post': 'max'  # Rough proxy for scoring
    }).rename(columns={
        'epa': 'off_epa',
        'play_id': 'off_plays',
        'posteam_score_post': 'max_score'
    })
    
    # Defensive EPA (when team is defending - flip sign so positive = good)
    def_stats = real_plays.groupby('defteam').agg({
        'epa': 'mean',
        'play_id': 'count'
    }).rename(columns={
        'epa': 'def_epa_raw',
        'play_id': 'def_plays'
    })
    def_stats['def_epa'] = -def_stats['def_epa_raw']  # Flip: positive = good defense
    def_stats = def_stats.drop(columns=['def_epa_raw'])
    
    # Calculate points per game from PBP scoring plays
    games = pbp[pbp['game_id'].notna()].groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_score': 'max',
        'away_score': 'max'
    }).reset_index()
    
    # Home team PPG
    home_ppg = games.groupby('home_team').agg({
        'home_score': 'mean',
        'away_score': 'mean',
        'game_id': 'count'
    }).rename(columns={
        'home_score': 'home_ppg',
        'away_score': 'home_ppg_allowed',
        'game_id': 'home_games'
    })
    
    # Away team PPG
    away_ppg = games.groupby('away_team').agg({
        'away_score': 'mean',
        'home_score': 'mean',
        'game_id': 'count'
    }).rename(columns={
        'away_score': 'away_ppg',
        'home_score': 'away_ppg_allowed',
        'game_id': 'away_games'
    })
    
    # Merge offensive and defensive stats
    team_epa = off_stats.join(def_stats, how='outer').fillna(0)
    team_epa = team_epa.join(home_ppg, how='left').join(away_ppg, how='left').fillna(0)
    
    # Calculate combined PPG
    team_epa['games'] = team_epa['home_games'] + team_epa['away_games']
    team_epa['ppg'] = (
        (team_epa['home_ppg'] * team_epa['home_games'] + 
         team_epa['away_ppg'] * team_epa['away_games']) / 
        team_epa['games'].replace(0, 1)
    )
    team_epa['ppg_allowed'] = (
        (team_epa['home_ppg_allowed'] * team_epa['home_games'] + 
         team_epa['away_ppg_allowed'] * team_epa['away_games']) / 
        team_epa['games'].replace(0, 1)
    )
    
    # Total EPA
    team_epa['total_epa'] = team_epa['off_epa'] + team_epa['def_epa']
    
    # Calculate and merge momentum data
    print("📈 Calculating team momentum (recent form vs season average)...")
    momentum_df = calculate_team_momentum(pbp, n_recent_games=4)
    
    # Clean up and select final columns
    team_epa = team_epa.reset_index().rename(columns={'index': 'team', 'posteam': 'team'})
    if 'posteam' in team_epa.columns:
        team_epa = team_epa.rename(columns={'posteam': 'team'})
    
    # Ensure team column exists
    if 'team' not in team_epa.columns:
        team_epa = team_epa.reset_index()
        team_epa.columns = ['team'] + list(team_epa.columns[1:])
    
    # Merge momentum data
    if not momentum_df.empty:
        team_epa = team_epa.merge(
            momentum_df[['team', 'recent_off_epa', 'recent_def_epa', 'off_momentum', 'def_momentum', 'total_momentum']],
            on='team',
            how='left'
        )
        # Fill missing momentum with 0 (neutral)
        for col in ['off_momentum', 'def_momentum', 'total_momentum']:
            team_epa[col] = team_epa[col].fillna(0)
        print(f"   ✅ Added momentum data for {len(momentum_df)} teams")
    
    # Select and order columns
    final_cols = ['team', 'off_epa', 'def_epa', 'total_epa', 'off_plays', 'def_plays', 
                  'ppg', 'ppg_allowed', 'games', 'off_momentum', 'def_momentum', 'total_momentum']
    team_epa = team_epa[[c for c in final_cols if c in team_epa.columns]]
    
    # Remove any invalid team entries
    team_epa = team_epa[team_epa['team'].notna() & (team_epa['team'] != '')]
    
    # Round for readability
    for col in ['off_epa', 'def_epa', 'total_epa', 'ppg', 'ppg_allowed']:
        if col in team_epa.columns:
            team_epa[col] = team_epa[col].round(3)
    
    # Save cache (season-specific)
    team_epa.to_parquet(cache_file, index=False)
    
    # Save metadata
    meta = {
        "season": actual_season,
        "week": get_current_nfl_week() if actual_season >= datetime.now().year else 18,
        "updated": datetime.now().isoformat(),
        "teams": len(team_epa)
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"💾 Cached {actual_season} EPA data for {len(team_epa)} teams")
    
    return team_epa


def get_league_averages(epa_df: pd.DataFrame) -> dict:
    """Get league average stats for baseline calculations."""
    return {
        'avg_off_epa': epa_df['off_epa'].mean(),
        'avg_def_epa': epa_df['def_epa'].mean(),
        'avg_ppg': epa_df['ppg'].mean() if 'ppg' in epa_df.columns else 21.0,
        'avg_ppg_allowed': epa_df['ppg_allowed'].mean() if 'ppg_allowed' in epa_df.columns else 21.0,
    }


def print_epa_rankings(epa_df: pd.DataFrame):
    """Print EPA rankings for quick overview."""
    print("\n" + "=" * 60)
    print("  NFL EPA RANKINGS")
    print("=" * 60)
    
    print("\n🏈 BEST OFFENSES (EPA/play):")
    print("-" * 40)
    off_ranked = epa_df.nlargest(10, 'off_epa')[['team', 'off_epa', 'ppg']]
    for i, row in enumerate(off_ranked.itertuples(), 1):
        ppg_str = f"{row.ppg:.1f}" if hasattr(row, 'ppg') else "N/A"
        print(f"  {i:2}. {row.team:4} {row.off_epa:+.3f} EPA/play  ({ppg_str} PPG)")
    
    print("\n🛡️  BEST DEFENSES (EPA/play):")
    print("-" * 40)
    def_ranked = epa_df.nlargest(10, 'def_epa')[['team', 'def_epa', 'ppg_allowed']]
    for i, row in enumerate(def_ranked.itertuples(), 1):
        ppg_str = f"{row.ppg_allowed:.1f}" if hasattr(row, 'ppg_allowed') else "N/A"
        print(f"  {i:2}. {row.team:4} {row.def_epa:+.3f} EPA/play  ({ppg_str} PPG allowed)")
    
    print("\n⭐ OVERALL BEST TEAMS (Total EPA):")
    print("-" * 40)
    total_ranked = epa_df.nlargest(10, 'total_epa')[['team', 'total_epa', 'off_epa', 'def_epa']]
    for i, row in enumerate(total_ranked.itertuples(), 1):
        print(f"  {i:2}. {row.team:4} {row.total_epa:+.3f} total  (Off: {row.off_epa:+.3f}, Def: {row.def_epa:+.3f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load NFL team EPA stats")
    parser.add_argument("--season", type=int, default=2025, help="NFL season year")
    parser.add_argument("--force", action="store_true", help="Force refresh from API")
    args = parser.parse_args()
    
    epa_df = load_team_epa(season=args.season, force_refresh=args.force)
    print_epa_rankings(epa_df)
    
    print("\n📊 League Averages:")
    avgs = get_league_averages(epa_df)
    for k, v in avgs.items():
        print(f"  {k}: {v:.3f}")
