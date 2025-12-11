import os
import asyncio
import json
import re
import sys
import argparse
import numpy as np
import pandas as pd

# Advanced simulation with real tiebreakers
try:
    from nfl_predictor.simulation import (
        run_advanced_simulation,
        print_simulation_results,
        build_season_data_from_standings
    )
    from nfl_predictor.scraper import (
        scrape_pfr_schedule_simple,
        scrape_pfr_standings
    )
    from nfl_predictor.tiebreakers import Game
    ADVANCED_MODE = True
except ImportError as e:
    print(f"⚠️ Advanced mode not available: {e}")
    ADVANCED_MODE = False

# ==========================================
# CONFIGURATION
# ==========================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='NFL Playoff Predictor')
    parser.add_argument('--show-overall', action='store_true',
                       help='Show overall playoff probabilities table')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple simulation without real tiebreakers')
    parser.add_argument('--simulations', '-n', type=int, default=10000,
                       help='Number of Monte Carlo simulations (default: 10000)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--show-injuries', action='store_true',
                       help='Show current injury report and impacts')
    parser.add_argument('--no-injuries', action='store_true',
                       help='Disable injury adjustments in simulation')
    parser.add_argument('--no-momentum', action='store_true',
                       help='Disable momentum adjustments in simulation')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# ==========================================
# CACHING LOGIC (legacy - now handled in pfr_scraper)
# ==========================================
CACHE_FILE = "cache/nfl_standings_cache.json"

def load_cached_standings():
    """Load standings from cache if valid (less than 24 hours old)"""
    if not os.path.exists(CACHE_FILE):
        print("🗃️ No cache file found.")
        return None
    
    try:
        import time
        # Check file age (24 hours = 86400 seconds)
        file_age = time.time() - os.path.getmtime(CACHE_FILE)
        if file_age > 86400:
            print("🕒 Cache expired (older than 24h).")
            return None
            
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
            print(f"📂 Loaded cached standings ({len(data)} teams) from {CACHE_FILE}")
            return data
    except Exception as e:
        print(f"⚠️ Failed to load cache: {e}")
        return None

def save_cached_standings(teams_data):
    """Save standings to cache"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(teams_data, f, indent=2)
        print(f"💾 Saved standings to {CACHE_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

# ==========================================
# SCRAPER & PREDICTOR LOGIC
# ==========================================

def get_nfl_standings():
    """Get NFL standings using HTTP-based scraper from Pro-Football-Reference"""
    if ADVANCED_MODE:
        return scrape_pfr_standings()
    else:
        # Fallback to cached data if available
        cached_data = load_cached_standings()
        if cached_data:
            return cached_data
        print("❌ Advanced mode not available and no cached data")
        return []


def extract_json_from_text(text):
    """Extract JSON array of NFL team data from text"""
    # Look for JSON arrays in the text
    json_patterns = [
        r'\[\s*\{.*?\}\s*\]',  # Standard JSON array of objects
        r'\[.*?\]',            # Generic array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)

                # Check if it's a list of teams
                if isinstance(data, list) and len(data) > 0:
                    valid_teams = []
                    for item in data:
                        if isinstance(item, dict) and 'name' in item:
                            
                            # CLEANUP: Handle "AFC East" -> "East"
                            div = item.get('div', 'Unknown')
                            if ' ' in div:
                                div = div.split(' ')[-1] # Take last word "East" from "AFC East"
                            
                            team = {
                                'name': item.get('name', ''),
                                'w': int(item.get('w', 0)),
                                'l': int(item.get('l', 0)),
                                't': int(item.get('t', 0)),
                                'div': div,
                                'conf': item.get('conf', 'Unknown'),
                                'pf': int(item.get('pf', 0)),
                                'pa': int(item.get('pa', 0))
                            }

                            if is_valid_nfl_team(team['name']):
                                # Auto-fill if still unknown or verify
                                conf_real, div_real = get_team_conference_division(team['name'])
                                
                                # Trust the API but fallback/normalize to known values
                                if team['conf'] not in ['AFC', 'NFC']:
                                    team['conf'] = conf_real
                                if team['div'] not in ['East', 'North', 'South', 'West']:
                                    team['div'] = div_real
                                    
                                valid_teams.append(team)

                    if len(valid_teams) == 32:
                        print(f"✅ Successfully parsed {len(valid_teams)} teams from JSON")
                        return valid_teams
                    else:
                        print(f"⚠️ Found {len(valid_teams)} valid teams in JSON, expected 32.")

            except json.JSONDecodeError:
                continue

    return extract_teams_from_text(text)

def extract_teams_from_text(text):
    """Extract team data from plain text when JSON fails"""
    print("🔍 Trying text-based extraction...")

    # Look for patterns like "Team Name: W-L, Division"
    lines = text.split('\n')
    teams_data = []

    for line in lines:
        # Try various patterns
        patterns = [
            r'([A-Za-z\s&]+):\s*(\d+)-(\d+)-?(\d*)',  # Team: W-L-T
            r'([A-Za-z\s&]+)\s+(\d+)-(\d+)-?(\d*)',   # Team W-L-T
        ]

        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                team_name = match[0].strip()
                wins = int(match[1])
                losses = int(match[2])
                ties = int(match[3]) if match[3] and match[3].isdigit() else 0

                if is_valid_nfl_team(team_name):
                    conf, div = get_team_conference_division(team_name)
                    teams_data.append({
                        'name': team_name,
                        'w': wins,
                        'l': losses,
                        't': ties,
                        'div': div,
                        'conf': conf,
                        'pf': 0,  # Will be calculated from games data
                        'pa': 0   # Will be calculated from games data
                    })
                    print(f"✅ Found team: {team_name} ({wins}-{losses}-{ties})")

    return teams_data if len(teams_data) >= 30 else None

def is_valid_nfl_team(name):
    """Check if the name is a valid NFL team"""
    if not name:
        return False

    name_upper = name.upper().strip()
    nfl_teams = [
        'BUFFALO BILLS', 'MIAMI DOLPHINS', 'NEW ENGLAND PATRIOTS', 'NEW YORK JETS',
        'BALTIMORE RAVENS', 'CINCINNATI BENGALS', 'CLEVELAND BROWNS', 'PITTSBURGH STEELERS',
        'HOUSTON TEXANS', 'INDIANAPOLIS COLTS', 'JACKSONVILLE JAGUARS', 'TENNESSEE TITANS',
        'DENVER BRONCOS', 'KANSAS CITY CHIEFS', 'LAS VEGAS RAIDERS', 'LOS ANGELES CHARGERS',
        'DALLAS COWBOYS', 'NEW YORK GIANTS', 'PHILADELPHIA EAGLES', 'WASHINGTON COMMANDERS',
        'CHICAGO BEARS', 'DETROIT LIONS', 'GREEN BAY PACKERS', 'MINNESOTA VIKINGS',
        'ATLANTA FALCONS', 'CAROLINA PANTHERS', 'NEW ORLEANS SAINTS', 'TAMPA BAY BUCCANEERS',
        'ARIZONA CARDINALS', 'LOS ANGELES RAMS', 'SAN FRANCISCO 49ERS', 'SEATTLE SEAHAWKS'
    ]

    return name_upper in nfl_teams

def get_team_conference_division(team_name):
    """Get conference and division for a team"""
    afc_teams = {
        'East': ['Buffalo Bills', 'Miami Dolphins', 'New England Patriots', 'New York Jets'],
        'North': ['Baltimore Ravens', 'Cincinnati Bengals', 'Cleveland Browns', 'Pittsburgh Steelers'],
        'South': ['Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Tennessee Titans'],
        'West': ['Denver Broncos', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers']
    }

    for division, teams in afc_teams.items():
        if team_name in teams:
            return 'AFC', division

    # If not AFC, it's NFC
    nfc_divisions = {
        'East': ['Dallas Cowboys', 'New York Giants', 'Philadelphia Eagles', 'Washington Commanders'],
        'North': ['Chicago Bears', 'Detroit Lions', 'Green Bay Packers', 'Minnesota Vikings'],
        'South': ['Atlanta Falcons', 'Carolina Panthers', 'New Orleans Saints', 'Tampa Bay Buccaneers'],
        'West': ['Arizona Cardinals', 'Los Angeles Rams', 'San Francisco 49ers', 'Seattle Seahawks']
    }

    for division, teams in nfc_divisions.items():
        if team_name in teams:
            return 'NFC', division

    # Fallback
    return 'Unknown', 'Unknown'

# Old extraction functions removed - using improved LLM-based parsing

# Sample data removed - application now requires real scraped NFL data
# This ensures predictions are based on actual current standings

def run_simulation(teams_data, n_simulations=10000):
    print(f"\n🎲 [Step 2] Running {n_simulations} Monte Carlo simulations...")
    results = {team['name']: 0 for team in teams_data}
    
    # Pre-calculate stats
    team_stats = []
    for t in teams_data:
        games_played = t['w'] + t['l'] + t['t']
        win_pct = t['w'] / games_played if games_played > 0 else 0.5
        remaining_games = 17 - games_played
        
        team_stats.append({
            'name': t['name'],
            'current_w': t['w'],
            'remaining': remaining_games,
            'win_pct': win_pct,
            'div': t['div'],
            'conf': t['conf']
        })

    for _ in range(n_simulations):
        sim_season = []
        for t in team_stats:
            new_wins = np.random.binomial(n=t['remaining'], p=t['win_pct'])
            # Score = Wins + Random Tiebreaker (0.0-0.9)
            score = t['current_w'] + new_wins + np.random.uniform(0, 0.9)
            sim_season.append({'name': t['name'], 'div': t['div'], 'conf': t['conf'], 'score': score})
            
        df = pd.DataFrame(sim_season)
        
        # Determine Playoff Qualifiers
        for conference in ['AFC', 'NFC']:
            conf_teams = df[df['conf'] == conference]
            div_winners = []
            wild_card_pool = []
            
            for division in ['North', 'South', 'East', 'West']:
                div_group = conf_teams[conf_teams['div'] == division]
                if len(div_group) > 0:
                    sorted_group = div_group.sort_values('score', ascending=False)
                    div_winners.append(sorted_group.iloc[0]['name'])
                    if len(sorted_group) > 1:
                        wild_card_pool.extend(sorted_group.iloc[1:].to_dict('records'))
            
            if wild_card_pool:
                wc_df = pd.DataFrame(wild_card_pool).sort_values('score', ascending=False)
                qualifiers = div_winners + wc_df.head(3)['name'].tolist()
            else:
                qualifiers = div_winners
            
            for team in qualifiers:
                results[team] += 1

    return results

async def main():
    teams_data = get_nfl_standings()
    if not teams_data: return

    # Load injury data if requested
    injury_impacts = None
    if not args.no_injuries and ADVANCED_MODE:
        try:
            from nfl_predictor.injuries import load_injury_data, load_snap_counts, get_current_nfl_week
            from nfl_predictor.player_impact import get_all_team_impacts

            print("\n🏥 Loading injury data...")
            injuries_df = load_injury_data(season=2025)
            snap_counts_df = load_snap_counts(season=2025)
            current_week = get_current_nfl_week()

            # Calculate injury impacts
            injury_impacts = get_all_team_impacts(injuries_df, snap_counts_df, current_week)

            if injury_impacts:
                print(f"📋 Calculated injury impacts for {len(injury_impacts)} teams")
                # Show summary of impacts
                significant_impacts = [
                    (team, impact) for team, impact in injury_impacts.items()
                    if impact['total_impact'] > 0.01
                ]
                if significant_impacts:
                    print("   Teams with significant injury impacts:")
                    for team, impact in sorted(significant_impacts, key=lambda x: x[1]['total_impact'], reverse=True):
                        total_pct = impact['total_impact'] * 100
                        print(f"     {team}: {total_pct:.1f}% impact")
            else:
                print("✅ No significant injuries this week")

            # Show detailed injury summary if requested
            if args.show_injuries:
                from nfl_predictor.injuries import print_injury_summary
                print_injury_summary(injuries_df, current_week)

                from nfl_predictor.player_impact import print_team_impact_summary
                print_team_impact_summary(injury_impacts, injuries_df, current_week)

        except Exception as e:
            print(f"⚠️ Failed to load injury data: {e}")
            print("   Continuing without injury adjustments")
            injury_impacts = None

    # Check if we should use advanced mode
    use_advanced = ADVANCED_MODE and not args.simple

    if use_advanced:
        print("\n🔄 Fetching game data from Pro-Football-Reference...")
        try:
            # Use HTTP scraper (fast and reliable)
            completed_games, remaining_games = await scrape_pfr_schedule_simple(season=2025)

            if not completed_games:
                print("⚠️ No game data available, falling back to simple mode")
                use_advanced = False
            else:
                print(f"✅ Loaded {len(completed_games)} completed games, {len(remaining_games)} remaining")
        except Exception as e:
            print(f"⚠️ Failed to fetch game data: {e}")
            print("   Falling back to simple simulation mode")
            use_advanced = False

    if use_advanced:
        # Run advanced simulation with real tiebreakers
        results = run_advanced_simulation(
            standings=teams_data,
            completed_games=completed_games,
            remaining_games=remaining_games,
            n_simulations=args.simulations,
            show_progress=not args.no_progress,
            injury_impacts=injury_impacts,
            use_momentum=not args.no_momentum
        )

        print_simulation_results(results, teams_data, n_simulations=args.simulations)
        return  # Advanced mode handles its own output
    
    # Fall back to simple simulation
    odds = run_simulation(teams_data, n_simulations=args.simulations)

    # Separate teams by conference
    afc_teams = {}
    nfc_teams = {}

    for team_data in teams_data:
        team_name = team_data['name']
        if team_data['conf'] == 'AFC':
            afc_teams[team_name] = odds.get(team_name, 0)
        else:
            nfc_teams[team_name] = odds.get(team_name, 0)

    print("\n========================================")
    print("      NFL PLAYOFF PROBABILITIES         ")
    print("========================================")

    # AFC Playoffs (7 teams: 4 division winners + 3 wild cards)
    print("\n🏈 AFC PLAYOFF PICTURE")
    print("-" * 40)
    print("Division Winners:")
    afc_div_winners = []
    for div in ['East', 'North', 'South', 'West']:
        div_teams = [(t, p) for t, p in afc_teams.items()
                    if next((td for td in teams_data if td['name'] == t and td['div'] == div), None)]
        if div_teams:
            best_team = max(div_teams, key=lambda x: x[1])
            afc_div_winners.append(best_team[0])
            prob = (best_team[1] / 10000) * 100
            print(f"  {div} Division: {best_team[0]:<20} {prob:>5.1f}%")

    print("\nWild Card Teams:")
    # Remove division winners from wild card consideration
    afc_wc_candidates = [(t, p) for t, p in afc_teams.items() if t not in afc_div_winners]
    afc_wc_sorted = sorted(afc_wc_candidates, key=lambda x: x[1], reverse=True) # Full sorted list
    
    # Top 3 are wild cards
    for i, (team, count) in enumerate(afc_wc_sorted[:3], 1):
        prob = (count / 10000) * 100
        print(f"  WC #{i}: {team:<20} {prob:>5.1f}%")
        
    # Next 3 are outside looking in
    print("\nOutside Looking In:")
    afc_outside = afc_wc_sorted[3:6]
    if not afc_outside:
        print("  (No other teams)")
    else:
        for team, count in afc_outside:
            prob = (count / 10000) * 100
            print(f"  {team:<25} {prob:>5.1f}%")

    # NFC Playoffs (7 teams: 4 division winners + 3 wild cards)
    print("\n🏈 NFC PLAYOFF PICTURE")
    print("-" * 40)
    print("Division Winners:")
    nfc_div_winners = []
    for div in ['East', 'North', 'South', 'West']:
        div_teams = [(t, p) for t, p in nfc_teams.items()
                    if next((td for td in teams_data if td['name'] == t and td['div'] == div), None)]
        if div_teams:
            best_team = max(div_teams, key=lambda x: x[1])
            nfc_div_winners.append(best_team[0])
            prob = (best_team[1] / 10000) * 100
            print(f"  {div} Division: {best_team[0]:<20} {prob:>5.1f}%")

    print("\nWild Card Teams:")
    # Remove division winners from wild card consideration
    nfc_wc_candidates = [(t, p) for t, p in nfc_teams.items() if t not in nfc_div_winners]
    nfc_wc_sorted = sorted(nfc_wc_candidates, key=lambda x: x[1], reverse=True) # Full sorted list
    
    # Top 3 are wild cards
    for i, (team, count) in enumerate(nfc_wc_sorted[:3], 1):
        prob = (count / 10000) * 100
        print(f"  WC #{i}: {team:<20} {prob:>5.1f}%")
        
    # Next 3 are outside looking in
    print("\nOutside Looking In:")
    nfc_outside = nfc_wc_sorted[3:6]
    if not nfc_outside:
        print("  (No other teams)")
    else:
        for team, count in nfc_outside:
            prob = (count / 10000) * 100
            print(f"  {team:<25} {prob:>5.1f}%")

    # Overall playoff probabilities (for teams that made playoffs) - OPTIONAL
    if args.show_overall:
        print(f"\n🏆 OVERALL PLAYOFF PROBABILITIES")
        print("-" * 40)

        # Extract just team names from wild card sorted list
        afc_wc_teams = [team for team, _ in afc_wc_sorted[:3]]
        nfc_wc_teams = [team for team, _ in nfc_wc_sorted[:3]]

        all_playoff_teams = afc_div_winners + afc_wc_teams + nfc_div_winners + nfc_wc_teams
        all_playoff_odds = [(t, odds[t]) for t in all_playoff_teams]
        all_playoff_odds_sorted = sorted(all_playoff_odds, key=lambda x: x[1], reverse=True)

        for team, count in all_playoff_odds_sorted:
            prob = (count / 10000) * 100
            print(f"{team:<25} {prob:>5.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
