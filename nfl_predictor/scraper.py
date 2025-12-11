"""
Pro-Football-Reference Scraper for NFL Game Data

Scrapes completed games and remaining schedule from Pro-Football-Reference.com
Uses browser-use for reliable data extraction.
"""

import os
import re
import json
import asyncio
from typing import List, Tuple, Optional
from dataclasses import dataclass

from browser_use import Agent, ChatGoogle, Browser
from .tiebreakers import (
    Game,
    TEAM_TO_CONFERENCE,
    TEAM_TO_DIVISION,
    get_current_nfl_week,
    is_cache_valid_for_week,
    save_schedule_cache, load_schedule_cache,
    save_games_cache, load_games_cache
)


# Team name normalization (PFR sometimes uses different names)
TEAM_NAME_ALIASES = {
    'Kansas City': 'Kansas City Chiefs',
    'Buffalo': 'Buffalo Bills', 
    'Baltimore': 'Baltimore Ravens',
    'Cincinnati': 'Cincinnati Bengals',
    'Cleveland': 'Cleveland Browns',
    'Pittsburgh': 'Pittsburgh Steelers',
    'Houston': 'Houston Texans',
    'Indianapolis': 'Indianapolis Colts',
    'Jacksonville': 'Jacksonville Jaguars',
    'Tennessee': 'Tennessee Titans',
    'Denver': 'Denver Broncos',
    'Las Vegas': 'Las Vegas Raiders',
    'Los Angeles Chargers': 'Los Angeles Chargers',
    'LA Chargers': 'Los Angeles Chargers',
    'Miami': 'Miami Dolphins',
    'New England': 'New England Patriots',
    'New York Jets': 'New York Jets',
    'NY Jets': 'New York Jets',
    'Philadelphia': 'Philadelphia Eagles',
    'Dallas': 'Dallas Cowboys',
    'Washington': 'Washington Commanders',
    'New York Giants': 'New York Giants',
    'NY Giants': 'New York Giants',
    'Chicago': 'Chicago Bears',
    'Detroit': 'Detroit Lions',
    'Green Bay': 'Green Bay Packers',
    'Minnesota': 'Minnesota Vikings',
    'Atlanta': 'Atlanta Falcons',
    'Carolina': 'Carolina Panthers',
    'New Orleans': 'New Orleans Saints',
    'Tampa Bay': 'Tampa Bay Buccaneers',
    'Arizona': 'Arizona Cardinals',
    'Los Angeles Rams': 'Los Angeles Rams',
    'LA Rams': 'Los Angeles Rams',
    'San Francisco': 'San Francisco 49ers',
    'Seattle': 'Seattle Seahawks',
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to standard format"""
    name = name.strip()
    
    # Check aliases
    if name in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[name]
    
    # Check if already a full name
    if name in TEAM_TO_CONFERENCE:
        return name
    
    # Try partial match
    for full_name in TEAM_TO_CONFERENCE.keys():
        if name in full_name or full_name.endswith(name):
            return full_name
    
    return name


def parse_pfr_game_line(line: str) -> Optional[Game]:
    """
    Parse a single game line from PFR format.
    
    Completed game format:
    | Week | Day | Date | Time | Winner/tie | @ or blank | Loser/tie | boxscore | PtsW | PtsL | ...
    
    Future game format:
    | Week | Day | Date | Time | Visitor | @ | Home | preview | ...
    """
    # Clean up the line
    line = line.strip()
    if not line or line.startswith('Week'):
        return None
    
    # Split by pipe
    parts = [p.strip() for p in line.split('|')]
    parts = [p for p in parts if p]  # Remove empty parts
    
    if len(parts) < 7:
        return None
    
    try:
        week_str = parts[0]
        
        # Skip header rows and playoff games
        if week_str in ['Week', 'WildCard', 'Division', 'ConfChamp', 'SuperBowl', '']:
            return None
        
        # Handle preseason
        if week_str.isdigit():
            week = int(week_str)
        else:
            return None  # Skip non-regular-season games
        
        # parts[4] is first team, parts[5] is @ or blank, parts[6] is second team
        team1 = normalize_team_name(parts[4])
        location_indicator = parts[5] if len(parts) > 5 else ''
        team2 = normalize_team_name(parts[6]) if len(parts) > 6 else ''
        
        # parts[7] is 'boxscore' (completed) or 'preview' (future)
        game_status = parts[7] if len(parts) > 7 else ''
        
        if game_status == 'boxscore':
            # Completed game: team1 is winner, team2 is loser
            # @ means winner was away team
            winner = team1
            loser = team2
            
            if location_indicator == '@':
                # Winner was away, so loser is home
                home_team = loser
                away_team = winner
            else:
                # Winner was home
                home_team = winner
                away_team = loser
            
            # Get scores
            pts_w = int(parts[8]) if len(parts) > 8 and parts[8].isdigit() else 0
            pts_l = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
            
            if location_indicator == '@':
                home_score = pts_l
                away_score = pts_w
            else:
                home_score = pts_w
                away_score = pts_l
            
            return Game(
                week=week,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                completed=True
            )
        
        elif game_status == 'preview':
            # Future game: team1 is visitor, team2 is home
            # @ confirms visitor/home relationship
            away_team = team1
            home_team = team2
            
            return Game(
                week=week,
                home_team=home_team,
                away_team=away_team,
                completed=False
            )
        
    except (ValueError, IndexError) as e:
        pass
    
    return None


def parse_pfr_schedule_data(content: str) -> Tuple[List[Game], List[Game]]:
    """
    Parse PFR schedule page content.
    Returns (completed_games, remaining_games)
    """
    completed = []
    remaining = []
    
    lines = content.split('\n')
    
    for line in lines:
        game = parse_pfr_game_line(line)
        if game:
            if game.completed:
                completed.append(game)
            else:
                remaining.append(game)
    
    return completed, remaining


async def scrape_pfr_schedule(model: str = "gemini-2.5-flash", 
                               vision: bool = True,
                               season: int = 2025) -> Tuple[List[Game], List[Game]]:
    """
    Scrape NFL schedule from Pro-Football-Reference.
    Returns (completed_games, remaining_games)
    """
    print(f"\n📅 Scraping {season} NFL schedule from Pro-Football-Reference...")
    
    # Check cache first
    cached_games = load_games_cache()
    cached_schedule = load_schedule_cache()
    
    if cached_games is not None and cached_schedule is not None:
        print("✅ Using cached game data")
        return cached_games, cached_schedule
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    llm = ChatGoogle(
        model=model,
        temperature=0.0,
        api_key=api_key,
    )
    
    browser = Browser(headless=True)
    
    try:
        task = f"""Go to https://www.pro-football-reference.com/years/{season}/games.htm

Extract ALL regular season games from the schedule table. For each game, extract:
- Week number (1-18, skip playoff games)
- Winner/Visitor team name
- Loser/Home team name  
- Whether the @ symbol is present (indicates away game for winner)
- Whether it's completed (has "boxscore" link) or future ("preview" link)
- For completed games: Winner points (PtsW) and Loser points (PtsL)

Return the data in this exact format, one game per line:
Week|Day|Date|Time|Team1|@_or_blank|Team2|boxscore_or_preview|PtsW|PtsL

Example completed game (home win):
1|Thu|2025-09-04|8:20PM|Philadelphia Eagles||Dallas Cowboys|boxscore|24|20

Example completed game (away win):
1|Sun|2025-09-07|1:00PM|Tampa Bay Buccaneers|@|Atlanta Falcons|boxscore|23|20

Example future game:
15|Thu|2025-12-11|8:15PM|Atlanta Falcons|@|Tampa Bay Buccaneers|preview||

Scroll down to capture ALL games from weeks 1-18.
Return ONLY the game data, no other text."""

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            use_vision=vision,
            max_actions_per_step=2,
        )
        
        result = await agent.run()
        
        print("✅ PFR scrape completed!")
        
        # Extract content from result
        content = ""
        if hasattr(result, 'final_result') and callable(result.final_result):
            content = result.final_result() or ""
        elif hasattr(result, 'extracted_content'):
            content = result.extracted_content or ""
        elif isinstance(result, str):
            content = result
        
        if not content:
            print("⚠️ No content extracted, trying history...")
            if hasattr(result, 'history') and result.history:
                for item in reversed(result.history):
                    if hasattr(item, 'result') and item.result:
                        if hasattr(item.result, 'extracted_content'):
                            content = item.result.extracted_content
                            break
        
        print(f"📄 Extracted content length: {len(content)}")
        
        # Parse the content
        completed, remaining = parse_pfr_schedule_data(content)
        
        print(f"✅ Parsed {len(completed)} completed games, {len(remaining)} remaining games")
        
        # Save to cache
        if completed:
            save_games_cache(completed)
        if remaining:
            save_schedule_cache(remaining)
        
        return completed, remaining
        
    finally:
        await browser.stop()


async def scrape_pfr_schedule_simple(season: int = 2025) -> Tuple[List[Game], List[Game]]:
    """
    Simple scraper using direct HTTP request with BeautifulSoup.
    Primary method - faster and more reliable than browser scraping.
    """
    import urllib.request
    from bs4 import BeautifulSoup
    
    print(f"\n📅 Fetching {season} NFL schedule (direct request)...")
    
    # Check cache first
    cached_games = load_games_cache()
    cached_schedule = load_schedule_cache()
    
    if cached_games is not None and cached_schedule is not None:
        print("✅ Using cached game data")
        return cached_games, cached_schedule
    
    url = f"https://www.pro-football-reference.com/years/{season}/games.htm"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find the games table
        games_table = soup.find('table', id='games')
        if not games_table:
            print("⚠️ Could not find games table")
            return [], []
        
        completed = []
        remaining = []
        
        tbody = games_table.find('tbody')
        if not tbody:
            print("⚠️ Could not find table body")
            return [], []
        
        for row in tbody.find_all('tr'):
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class', []):
                continue
            
            cells = row.find_all(['td', 'th'])
            if len(cells) < 7:
                continue
            
            try:
                # Get week number
                week_cell = row.find('th', {'data-stat': 'week_num'})
                if not week_cell:
                    continue
                
                week_text = week_cell.get_text(strip=True)
                if not week_text.isdigit():
                    continue  # Skip playoff weeks or header rows
                week = int(week_text)
                
                # Check if this is a future game (preview) or completed (boxscore)
                boxscore_cell = row.find('td', {'data-stat': 'boxscore_word'})
                is_future = boxscore_cell and 'preview' in boxscore_cell.get_text(strip=True).lower()
                
                # Get winner/visitor and loser/home (PFR uses same columns for both)
                winner_cell = row.find('td', {'data-stat': 'winner'})
                loser_cell = row.find('td', {'data-stat': 'loser'})
                
                if not winner_cell or not loser_cell:
                    continue
                
                team1 = normalize_team_name(winner_cell.get_text(strip=True))
                team2 = normalize_team_name(loser_cell.get_text(strip=True))
                
                if not team1 or not team2:
                    continue
                
                # Check if away game (@ symbol indicates team1 was visitor)
                game_location = row.find('td', {'data-stat': 'game_location'})
                is_away = game_location and '@' in game_location.get_text()
                
                if is_future:
                    # Future game: team1 is visitor, team2 is home
                    # (when @ is present, it confirms team1 was away)
                    if is_away:
                        away_team = team1
                        home_team = team2
                    else:
                        # No @ means team1 listed first is home (in winner column)
                        home_team = team1
                        away_team = team2
                    
                    remaining.append(Game(
                        week=week,
                        home_team=home_team,
                        away_team=away_team,
                        completed=False
                    ))
                else:
                    # Completed game
                    pts_win_cell = row.find('td', {'data-stat': 'pts_win'})
                    pts_lose_cell = row.find('td', {'data-stat': 'pts_lose'})
                    
                    if pts_win_cell and pts_lose_cell:
                        pts_win_text = pts_win_cell.get_text(strip=True)
                        pts_lose_text = pts_lose_cell.get_text(strip=True)
                        
                        if pts_win_text.isdigit() and pts_lose_text.isdigit():
                            pts_win = int(pts_win_text)
                            pts_lose = int(pts_lose_text)
                            
                            if is_away:
                                # Winner was away team
                                home_team = team2
                                away_team = team1
                                home_score = pts_lose
                                away_score = pts_win
                            else:
                                # Winner was home team
                                home_team = team1
                                away_team = team2
                                home_score = pts_win
                                away_score = pts_lose
                            
                            completed.append(Game(
                                week=week,
                                home_team=home_team,
                                away_team=away_team,
                                home_score=home_score,
                                away_score=away_score,
                                completed=True
                            ))
                
            except (ValueError, AttributeError) as e:
                continue
        
        print(f"✅ Parsed {len(completed)} completed games, {len(remaining)} remaining")
        
        # Save to cache
        if completed:
            save_games_cache(completed)
        if remaining:
            save_schedule_cache(remaining)
        
        return completed, remaining
        
    except Exception as e:
        print(f"❌ Direct request failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []


# ==========================================
# STANDINGS SCRAPER (HTTP-based)
# ==========================================

STANDINGS_CACHE_FILE = "cache/nfl_standings_cache.json"


def load_standings_cache():
    """Load standings from cache if valid based on NFL week schedule"""
    import time
    from .config import get_current_nfl_week, is_cache_valid_for_week
    
    if not os.path.exists(STANDINGS_CACHE_FILE):
        return None
    
    try:
        with open(STANDINGS_CACHE_FILE, 'r') as f:
            data = json.load(f)
        
        # Handle old cache format (just a list) vs new format (dict with metadata)
        if isinstance(data, list):
            # Old format - invalidate to refresh with new format
            return None
        
        timestamp = data.get('timestamp', 0)
        cached_week = data.get('cached_week', 0)
        
        # Check if cache is still valid based on NFL week
        if not is_cache_valid_for_week(timestamp, cached_week):
            current_week = get_current_nfl_week()
            print(f"🔄 Standings cache outdated (cached week {cached_week}, current week {current_week})")
            return None
            
        return data.get('standings', [])
    except Exception as e:
        print(f"⚠️ Failed to load standings cache: {e}")
        return None


def save_standings_cache(teams_data):
    """Save standings to cache with week metadata"""
    import time
    from .tiebreakers import get_current_nfl_week
    
    try:
        current_week = get_current_nfl_week()
        data = {
            'timestamp': time.time(),
            'cached_week': current_week,
            'standings': teams_data
        }
        import os
        os.makedirs(os.path.dirname(STANDINGS_CACHE_FILE), exist_ok=True)
        with open(STANDINGS_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save standings cache: {e}")


def scrape_pfr_standings(season: int = 2025):
    """
    Scrape NFL standings from Pro-Football-Reference using HTTP.
    Returns list of team dicts with: name, w, l, t, div, conf, pf, pa
    """
    import urllib.request
    from bs4 import BeautifulSoup
    
    print(f"\n🏈 Fetching {season} NFL standings from Pro-Football-Reference...")
    
    # Check cache first
    cached_data = load_standings_cache()
    if cached_data:
        print(f"✅ Using cached standings ({len(cached_data)} teams)")
        return cached_data
    
    url = f"https://www.pro-football-reference.com/years/{season}/"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        soup = BeautifulSoup(html, 'html.parser')
        
        teams_data = []
        
        # PFR has separate tables for AFC and NFC with IDs
        for conf in ['AFC', 'NFC']:
            table = soup.find('table', id=conf)
            if not table:
                print(f"⚠️ Could not find {conf} standings table")
                continue
            
            tbody = table.find('tbody')
            if not tbody:
                continue
            
            current_division = None
            
            for row in tbody.find_all('tr'):
                # Check for division header rows (class='thead onecell')
                row_classes = row.get('class', [])
                if 'thead' in row_classes:
                    # This is a division header - text is in td, not th
                    td = row.find('td')
                    if td:
                        div_text = td.get_text(strip=True)
                        # Extract just the division name (e.g., "AFC East" -> "East")
                        if 'East' in div_text:
                            current_division = 'East'
                        elif 'North' in div_text:
                            current_division = 'North'
                        elif 'South' in div_text:
                            current_division = 'South'
                        elif 'West' in div_text:
                            current_division = 'West'
                    continue
                
                # Regular team row
                cells = row.find_all(['td', 'th'])
                if len(cells) < 7:
                    continue
                
                try:
                    # Team name is in the first th with data-stat='team'
                    team_cell = row.find('th', {'data-stat': 'team'})
                    if not team_cell:
                        continue
                    
                    # Get team name (remove * or + markers for playoff indicators)
                    team_link = team_cell.find('a')
                    if team_link:
                        team_name = team_link.get_text(strip=True)
                    else:
                        team_name = team_cell.get_text(strip=True)
                    
                    # Remove playoff indicators
                    team_name = team_name.rstrip('*+')
                    team_name = normalize_team_name(team_name)
                    
                    if not team_name or team_name not in TEAM_TO_CONFERENCE:
                        continue
                    
                    # Get W, L, T, PF, PA
                    wins_cell = row.find('td', {'data-stat': 'wins'})
                    losses_cell = row.find('td', {'data-stat': 'losses'})
                    ties_cell = row.find('td', {'data-stat': 'ties'})
                    pf_cell = row.find('td', {'data-stat': 'points'})
                    pa_cell = row.find('td', {'data-stat': 'points_opp'})
                    
                    wins = int(wins_cell.get_text(strip=True)) if wins_cell else 0
                    losses = int(losses_cell.get_text(strip=True)) if losses_cell else 0
                    ties = int(ties_cell.get_text(strip=True)) if ties_cell else 0
                    pf = int(pf_cell.get_text(strip=True)) if pf_cell else 0
                    pa = int(pa_cell.get_text(strip=True)) if pa_cell else 0
                    
                    # Determine division - use current_division if set from table header,
                    # otherwise fall back to TEAM_TO_DIVISION mapping
                    division = current_division
                    if not division:
                        div_full = TEAM_TO_DIVISION.get(team_name, '')
                        if div_full and ' ' in div_full:
                            division = div_full.split()[1]  # "AFC East" -> "East"
                        else:
                            division = 'Unknown'
                    
                    teams_data.append({
                        'name': team_name,
                        'w': wins,
                        'l': losses,
                        't': ties,
                        'div': division,
                        'conf': conf,
                        'pf': pf,
                        'pa': pa
                    })
                    
                except (ValueError, AttributeError) as e:
                    continue
        
        if len(teams_data) >= 30:
            print(f"✅ Scraped standings for {len(teams_data)} teams")
            save_standings_cache(teams_data)
            return teams_data
        else:
            print(f"⚠️ Only found {len(teams_data)} teams, expected 32")
            return teams_data
            
    except Exception as e:
        print(f"❌ Failed to scrape standings: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==========================================
# STANDALONE TEST
# ==========================================

if __name__ == "__main__":
    async def test():
        # Test standings scraper
        print("Testing standings scraper...")
        standings = scrape_pfr_standings()
        print(f"\nStandings: {len(standings)} teams")
        for team in standings[:5]:
            print(f"  {team['name']}: {team['w']}-{team['l']}-{team['t']} ({team['conf']} {team['div']}) PF:{team['pf']} PA:{team['pa']}")
        
        print("\n" + "="*50 + "\n")
        
        # Test games scraper
        completed, remaining = await scrape_pfr_schedule_simple()
        print(f"\nCompleted games: {len(completed)}")
        print(f"Remaining games: {len(remaining)}")
        
        if completed:
            print("\nFirst 3 completed games:")
            for game in completed[:3]:
                print(f"  Week {game.week}: {game.away_team} @ {game.home_team} ({game.away_score}-{game.home_score})")
        
        if remaining:
            print("\nFirst 3 remaining games:")
            for game in remaining[:3]:
                print(f"  Week {game.week}: {game.away_team} @ {game.home_team}")
    
    asyncio.run(test())
