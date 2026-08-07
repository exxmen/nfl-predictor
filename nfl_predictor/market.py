"""
Market (Vegas) data integration via nfl_data_py.

Free source of consensus closing spreads: the `spread_line` column of the
schedule feed (nfl_data_py / nflverse), which we already depend on. No API
key, no paid tier.

Convention: `spread_line` is expressed from the HOME team's perspective
(POSITIVE = home team favored, e.g. +2.5 means home favored by 2.5).
This was verified empirically against 285 real 2024 games.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import nfl_data_py as nfl
import pandas as pd

from .team_names import to_full_name

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


# League-average expected total for market anchoring when no total line exists
DEFAULT_TOTAL = 43.0
# Weight applied to market-implied score vs the EPA model (tune via backtest)
MARKET_WEIGHT = 0.30


def _cache_paths(season: int) -> Tuple[Path, Path]:
    CACHE_DIR.mkdir(exist_ok=True)
    return (
        CACHE_DIR / f"spreads_{season}.parquet",
        CACHE_DIR / f"spreads_{season}_meta.json",
    )


def _cache_valid(season: int, meta_file: Path) -> bool:
    if not meta_file.exists():
        return False
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        cached_season = meta.get("season", 0)
        # Use get_current_season() (handles the Jan/Feb boundary where the
        # active NFL season is the previous calendar year) to decide whether
        # this is a historical season that never changes.
        from .config import get_current_season
        current_season = get_current_season()
        if season < current_season:
            return cached_season == season
        # Current season: refresh at least once per NFL week. Use the shared
        # week-based validity helper so a cache fetched before a week's games
        # start (Thursday kickoff) is not wrongly kept through game time.
        from .config import get_current_nfl_week, is_cache_valid_for_week
        current_week = get_current_nfl_week()
        if cached_season != season:
            return False
        if meta.get("week", 0) < current_week:
            return False
        # Same week: honor the shared kickoff-aware invalidation rule.
        updated = meta.get("updated")
        try:
            ts = datetime.fromisoformat(updated).timestamp() if updated else 0.0
        except (TypeError, ValueError):
            return False
        return is_cache_valid_for_week(ts, int(meta.get("week", 0)))
    except Exception:
        return False


def fetch_spreads(season: int, force_refresh: bool = False) -> Dict[Tuple[str, str], Optional[float]]:
    """
    Load closing consensus spreads keyed by (home_full_name, away_full_name).

    Returns a dict mapping the (home, away) full-name tuple to the home
    spread (POSITIVE = home favored; matches the verified nflverse
    convention used everywhere else). Games without a line map to None.
    """
    cache_file, meta_file = _cache_paths(season)
    if not force_refresh and _cache_valid(season, meta_file) and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return _df_to_spreads(df)

    print(f"📈 Fetching {season} consensus spreads from schedule feed...")
    schedule = nfl.import_schedules([season])
    if schedule is None or schedule.empty:
        logger.warning("No schedule data for %s; no market lines available", season)
        return {}

    rows = []
    for _, row in schedule.iterrows():
        home, away = row.get("home_team"), row.get("away_team")
        if not home or not away:
            continue
        spread = row.get("spread_line")
        rows.append({
            "home_full": to_full_name(home),
            "away_full": to_full_name(away),
            "spread": float(spread) if pd.notna(spread) else None,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(cache_file, index=False)
    with open(meta_file, "w") as f:
        json.dump({"season": season, "week": _current_week(season),
                   "updated": datetime.now().isoformat()}, f)

    spreads = _df_to_spreads(df)
    found = sum(1 for v in spreads.values() if v is not None)
    print(f"   ✅ Loaded {found} market lines for {len(spreads)} scheduled games")
    return spreads


def _current_week(season: int) -> int:
    from .config import get_current_nfl_week, get_current_season
    if season < get_current_season():
        return 18  # historical seasons fully resolved
    return get_current_nfl_week()


def _df_to_spreads(df: pd.DataFrame) -> Dict[Tuple[str, str], Optional[float]]:
    out = {}
    for _, row in df.iterrows():
        spread = row.get("spread")
        # DataFrame float columns turn missing values into NaN, not None;
        # pd.isna covers built-in float AND numpy.float64 (e.g. from parquet).
        if spread is None or pd.isna(spread):
            out[(row["home_full"], row["away_full"])] = None
        else:
            out[(row["home_full"], row["away_full"])] = float(spread)
    return out


def attach_spreads_to_games(remaining_games, spreads: Dict[Tuple[str, str], Optional[float]]) -> int:
    """
    Set `game.home_spread` on each remaining Game using the spread dict.
    Missing lines leave home_spread as None (game just skips market anchoring).

    Returns the number of games that received a line.
    """
    attached = 0
    for game in remaining_games:
        spread = spreads.get((game.home_team, game.away_team))
        if spread is not None:
            game.home_spread = spread
            attached += 1
    return attached


def market_implied_scores(home_spread: float, total: float = DEFAULT_TOTAL) -> Tuple[float, float]:
    """
    Convert a home-perspective spread into implied home/away expected points.

    VERIFIED empirically against 285 real 2024 games: nflverse `spread_line`
    uses POSITIVE = home favored (corr +0.44 with home win; 70.5% accuracy).
    So a positive spread gives the home team MORE expected points:
        home_expected = (total + spread) / 2
        away_expected = (total - spread) / 2
    e.g. spread = +3, total = 43 -> home = 23.0, away = 20.0.
    """
    home_expected = (total + home_spread) / 2.0
    away_expected = (total - home_spread) / 2.0
    return max(7.0, home_expected), max(7.0, away_expected)
