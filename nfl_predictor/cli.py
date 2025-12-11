import sys
import argparse
import asyncio
from datetime import datetime

from .config import get_current_season, get_current_nfl_week
from .simulation import run_advanced_simulation
from .scraper import scrape_pfr_standings, scrape_pfr_schedule_simple
from .injuries import load_injury_data, load_snap_counts
from .player_impact import get_all_team_impacts

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

def main():
    args = parse_arguments()
    
    print(f"🏈 NFL Playoff Predictor - Week {get_current_nfl_week()}")
    season = get_current_season()
    
    # Load Standings
    print(f"📊 Fetching {season} NFL standings...")
    standings = scrape_pfr_standings()
    if not standings:
        print("❌ Failed to load standings")
        sys.exit(1)
        
    # Load Schedule
    print(f"📅 Fetching {season} NFL schedule...")
    completed_games, remaining_games = asyncio.run(scrape_pfr_schedule_simple(season))
    
    print(f"📊 Data: {len(standings)} teams, {len(completed_games)} completed games, {len(remaining_games)} remaining")
    
    # Load Injuries (if not disabled)
    injury_impacts = None
    if not args.no_injuries:
        print(f"🏥 Loading {season} injury data...")
        try:
            injuries_df = load_injury_data(season)
            snap_counts_df = load_snap_counts(season)
            current_week = get_current_nfl_week()
            injury_impacts = get_all_team_impacts(injuries_df, snap_counts_df, current_week)
            print(f"🏥 Loaded injury impacts for {len(injury_impacts)} teams")
            
            if args.show_injuries:
                # TODO: Print injury summary if requested
                pass
        except Exception as e:
            print(f"⚠️ Failed to load injury data: {e}")

    # Run Simulation
    print(f"\n🚀 Running {args.simulations:,} simulations...")
    
    results = run_advanced_simulation(
        standings=standings,
        completed_games=completed_games,
        remaining_games=remaining_games,
        n_simulations=args.simulations,
        show_progress=not args.no_progress,
        injury_impacts=injury_impacts
    )
    
    # Print Results
    from .simulation import print_simulation_results
    print_simulation_results(results, standings, args.simulations)

if __name__ == "__main__":
    main()
