"""
Configuration settings for NFL Predictor.
Centralizes season/week logic to avoid hardcoding years.
"""

from datetime import datetime, date

# NFL Regular Season Week Start Dates (Thursday of each week)
# Games typically start Thursday evening and run through Monday night
# TODO: Update these annually or move to a dynamic source
NFL_WEEK_STARTS = {
    1: "2025-09-04", 2: "2025-09-11", 3: "2025-09-18", 4: "2025-09-25",
    5: "2025-10-02", 6: "2025-10-09", 7: "2025-10-16", 8: "2025-10-23",
    9: "2025-10-30", 10: "2025-11-06", 11: "2025-11-13", 12: "2025-11-20",
    13: "2025-11-27", 14: "2025-12-04", 15: "2025-12-11", 16: "2025-12-18",
    17: "2025-12-25", 18: "2026-01-03"
}

def get_current_season() -> int:
    """
    Determine the current NFL season year.
    If it's Jan or Feb, it's the previous year's season.
    """
    today = datetime.now()
    # In Jan/Feb, we are still completing the previous year's season
    if today.month < 3:
        return today.year - 1
    return today.year

def get_current_nfl_week() -> int:
    """Determine the current NFL week based on date."""
    today = datetime.now().strftime("%Y-%m-%d")
    current_week = 1
    for week, start_date in NFL_WEEK_STARTS.items():
        if today >= start_date:
            current_week = week
    return current_week


def is_cache_valid_for_week(cache_timestamp: float, cached_week: int) -> bool:
    """
    Check if cache is still valid based on NFL week schedule.
    
    Cache should be invalidated when:
    1. A new week's games have started (Thursday)
    2. We're in the same week but games might have been played
    
    Logic:
    - If we're in a later week than when cache was created, invalidate
    - If we're in the same week, check if games have likely started
    """
    current_week = get_current_nfl_week()
    
    # If current week is ahead of cached week, cache is stale
    if current_week > cached_week:
        return False
    
    # If we're in the same week, check if enough time has passed
    today = date.today()
    
    if current_week > 0 and current_week in NFL_WEEK_STARTS:
        week_start_str = NFL_WEEK_STARTS[current_week]
        week_start = datetime.strptime(week_start_str, "%Y-%m-%d").date()
        
        # If today is on or after the week start date,
        # check if cache was created before games started
        if today >= week_start:
            cache_date = datetime.fromtimestamp(cache_timestamp).date()
            # Cache is stale if it was created before this week's games started
            if cache_date < week_start:
                return False
        
    return True
