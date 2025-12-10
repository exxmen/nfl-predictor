#!/usr/bin/env python3
"""
Scheduled NFL Playoff Predictor

Runs high-iteration simulations and saves results to a dated file.
Designed to be run via cron Tuesday-Thursday (PHT) during NFL season.

Cron example (run at 8:00 AM PHT on Tue, Wed, Thu):
0 8 * * 2,3,4 cd /path/to/nfl-predictor && /path/to/uv run python scheduled_run.py

PHT (UTC+8) timing considerations:
- Thursday Night Football starts ~8:15 PM ET = Friday 9:15 AM PHT
- Monday Night Football ends ~midnight ET = Tuesday 1:00 PM PHT
- So Tuesday-Thursday mornings PHT are ideal for updated predictions
"""

import os
import sys
import json
from datetime import datetime, date

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfl_tiebreakers import get_current_nfl_week, NFL_2025_WEEK_STARTS


def should_run_today() -> tuple[bool, str]:
    """
    Check if we should run today based on:
    1. Day of week (Tuesday=1, Wednesday=2, Thursday=3 in Python's weekday())
    2. NFL season dates
    """
    today = date.today()
    weekday = today.weekday()  # Monday=0, Tuesday=1, Wednesday=2, Thursday=3
    
    # Only run Tuesday, Wednesday, Thursday
    if weekday not in [1, 2, 3]:
        day_name = today.strftime('%A')
        return False, f"Skipping: Today is {day_name}, only runs Tue-Thu"
    
    # Check if we're in NFL season (Week 1 start to Week 18 end)
    current_week = get_current_nfl_week()
    
    if current_week < 1:
        return False, "Skipping: NFL season hasn't started yet"
    
    if current_week > 18:
        return False, "Skipping: NFL regular season is over"
    
    return True, f"Running: Week {current_week}, {today.strftime('%A %Y-%m-%d')}"


def clear_caches():
    """Clear all caches to force fresh data fetch"""
    cache_files = [
        "nfl_standings_cache.json",
        "nfl_games_cache.json", 
        "nfl_schedule_cache.json"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"🗑️  Cleared {cache_file}")


def run_simulation(n_simulations: int = 100000) -> dict:
    """Run the simulation and return results"""
    import asyncio
    from advanced_simulation import run_advanced_simulation, build_season_data_from_standings
    from pfr_scraper import scrape_pfr_schedule_simple, scrape_pfr_standings
    from nfl_tiebreakers import Game
    
    # Get fresh standings
    teams_data = scrape_pfr_standings()
    if not teams_data:
        raise RuntimeError("Failed to fetch standings")
    
    # Get fresh game data
    async def fetch_games():
        return await scrape_pfr_schedule_simple(season=2025)
    
    completed_games, remaining_games = asyncio.run(fetch_games())
    
    if not completed_games:
        raise RuntimeError("Failed to fetch game data")
    
    print(f"📊 Data: {len(teams_data)} teams, {len(completed_games)} completed games, {len(remaining_games)} remaining")
    
    # Run simulation
    results = run_advanced_simulation(
        standings=teams_data,
        completed_games=completed_games,
        remaining_games=remaining_games,
        n_simulations=n_simulations,
        show_progress=True
    )
    
    return {
        'teams_data': teams_data,
        'results': results,
        'completed_games': len(completed_games),
        'remaining_games': len(remaining_games)
    }


def format_results(results: dict, n_simulations: int) -> str:
    """Format results as a string report"""
    from nfl_tiebreakers import TEAM_TO_DIVISION
    
    teams_data = results['teams_data']
    sim_results = results['results']
    
    lines = []
    lines.append("=" * 60)
    lines.append("    NFL PLAYOFF PROBABILITIES")
    lines.append(f"    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"    Simulations: {n_simulations:,}")
    lines.append(f"    Games: {results['completed_games']} completed, {results['remaining_games']} remaining")
    lines.append("=" * 60)
    
    # Separate by conference
    for conf in ['AFC', 'NFC']:
        conf_teams = {t['name']: t for t in teams_data if t['conf'] == conf}
        conf_results = {name: sim_results[name] for name in conf_teams if name in sim_results}
        
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {conf} PLAYOFF PICTURE")
        lines.append("=" * 60)
        
        # Sort by playoff probability
        sorted_teams = sorted(
            conf_results.items(),
            key=lambda x: x[1]['playoff_count'],
            reverse=True
        )
        
        # Division leaders
        div_leaders = {}
        for div in ['East', 'North', 'South', 'West']:
            div_teams = [
                (name, r) for name, r in conf_results.items()
                if TEAM_TO_DIVISION.get(name) == div
            ]
            if div_teams:
                leader = max(div_teams, key=lambda x: x[1]['division_winner_count'])
                div_leaders[div] = leader[0]
        
        lines.append("\nDivision Leaders:")
        lines.append(f"  {'Team':<26} {'Div%':>7} {'Playoff%':>9} {'Avg Wins':>9}")
        lines.append("  " + "-" * 53)
        
        for div in ['East', 'North', 'South', 'West']:
            if div in div_leaders:
                name = div_leaders[div]
                r = conf_results[name]
                div_pct = (r['division_winner_count'] / n_simulations) * 100
                playoff_pct = (r['playoff_count'] / n_simulations) * 100
                avg_wins = r.get('avg_wins', 0)
                lines.append(f"  {name:<26} {div_pct:>6.1f}% {playoff_pct:>8.1f}% {avg_wins:>9.1f}")
        
        # Wild Card
        lines.append("\nWild Card Contenders:")
        lines.append(f"  {'Team':<26} {'WC%':>7} {'Playoff%':>9} {'Avg Wins':>9}")
        lines.append("  " + "-" * 53)
        
        wc_candidates = [
            (name, r) for name, r in sorted_teams
            if name not in div_leaders.values()
        ]
        
        for i, (name, r) in enumerate(wc_candidates[:6], 1):
            wc_pct = (r['wildcard_count'] / n_simulations) * 100
            playoff_pct = (r['playoff_count'] / n_simulations) * 100
            avg_wins = r.get('avg_wins', 0)
            marker = "  " if i <= 3 else "  "  # Top 3 are in WC spots
            lines.append(f"{marker}{name:<26} {wc_pct:>6.1f}% {playoff_pct:>8.1f}% {avg_wins:>9.1f}")
        
        # Seed distribution
        lines.append("\nSeed Distribution:")
        lines.append(f"  {'Team':<22} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6} {'5th':>6} {'6th':>6} {'7th':>6}")
        lines.append("  " + "-" * 64)
        
        for name, r in sorted_teams[:10]:
            seed_pcts = []
            for seed in range(1, 8):
                pct = (r['seed_counts'][seed] / n_simulations) * 100
                seed_pcts.append(f"{pct:>5.1f}%")
            lines.append(f"  {name:<22} {' '.join(seed_pcts)}")
    
    return '\n'.join(lines)


def format_results_markdown(results: dict, n_simulations: int) -> str:
    """Format results as a properly formatted Markdown document"""
    from nfl_tiebreakers import TEAM_TO_DIVISION
    
    teams_data = results['teams_data']
    sim_results = results['results']
    current_week = get_current_nfl_week()
    
    lines = []
    
    # Header
    lines.append(f"# 🏈 NFL Playoff Probabilities - Week {current_week}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    lines.append(f"**Simulations:** {n_simulations:,}")
    lines.append(f"**Games:** {results['completed_games']} completed, {results['remaining_games']} remaining")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Process each conference
    for conf in ['AFC', 'NFC']:
        conf_teams = {t['name']: t for t in teams_data if t['conf'] == conf}
        conf_results = {name: sim_results[name] for name in conf_teams if name in sim_results}
        
        lines.append(f"## {conf} Playoff Picture")
        lines.append("")
        
        # Sort by playoff probability
        sorted_teams = sorted(
            conf_results.items(),
            key=lambda x: x[1]['playoff_count'],
            reverse=True
        )
        
        # Division leaders
        div_leaders = {}
        for div in ['East', 'North', 'South', 'West']:
            div_teams = [
                (name, r) for name, r in conf_results.items()
                if TEAM_TO_DIVISION.get(name) == div
            ]
            if div_teams:
                leader = max(div_teams, key=lambda x: x[1]['division_winner_count'])
                div_leaders[div] = leader[0]
        
        # Division Leaders Table
        lines.append("### 🏆 Division Leaders")
        lines.append("")
        lines.append("| Division | Team | Div % | Playoff % | Avg Wins |")
        lines.append("|:---------|:-----|------:|----------:|---------:|")
        
        for div in ['East', 'North', 'South', 'West']:
            if div in div_leaders:
                name = div_leaders[div]
                r = conf_results[name]
                div_pct = (r['division_winner_count'] / n_simulations) * 100
                playoff_pct = (r['playoff_count'] / n_simulations) * 100
                avg_wins = r.get('avg_wins', 0)
                lines.append(f"| {div} | {name} | {div_pct:.1f}% | {playoff_pct:.1f}% | {avg_wins:.1f} |")
        
        lines.append("")
        
        # Wild Card Race
        lines.append("### 🎯 Wild Card Race")
        lines.append("")
        lines.append("| # | Team | WC % | Playoff % | Avg Wins |")
        lines.append("|:-:|:-----|-----:|----------:|---------:|")
        
        wc_candidates = [
            (name, r) for name, r in sorted_teams
            if name not in div_leaders.values()
        ]
        
        for i, (name, r) in enumerate(wc_candidates[:3], 1):
            wc_pct = (r['wildcard_count'] / n_simulations) * 100
            playoff_pct = (r['playoff_count'] / n_simulations) * 100
            avg_wins = r.get('avg_wins', 0)
            lines.append(f"| {i} | {name} | {wc_pct:.1f}% | {playoff_pct:.1f}% | {avg_wins:.1f} |")
        
        lines.append("")
        
        # Outside Looking In
        lines.append("### 👀 Outside Looking In")
        lines.append("")
        lines.append("| # | Team | WC % | Playoff % | Avg Wins |")
        lines.append("|:-:|:-----|-----:|----------:|---------:|")
        
        for i, (name, r) in enumerate(wc_candidates[3:6], 1):
            wc_pct = (r['wildcard_count'] / n_simulations) * 100
            playoff_pct = (r['playoff_count'] / n_simulations) * 100
            avg_wins = r.get('avg_wins', 0)
            lines.append(f"| {i} | {name} | {wc_pct:.1f}% | {playoff_pct:.1f}% | {avg_wins:.1f} |")
        
        lines.append("")
        
        # Seed Distribution
        lines.append("### 📊 Seed Distribution")
        lines.append("")
        lines.append("| Team | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th |")
        lines.append("|:-----|----:|----:|----:|----:|----:|----:|----:|")
        
        for name, r in sorted_teams[:10]:
            seed_pcts = []
            for seed in range(1, 8):
                pct = (r['seed_counts'][seed] / n_simulations) * 100
                seed_pcts.append(f"{pct:.1f}%")
            lines.append(f"| {name} | {' | '.join(seed_pcts)} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("*Probabilities based on Monte Carlo simulation with full NFL tiebreaker rules.*")
    
    return '\n'.join(lines)


def save_results(report: str, markdown_report: str, results: dict, n_simulations: int):
    """Save results to dated files"""
    today = date.today()
    current_week = get_current_nfl_week()
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save text report
    report_file = f"{results_dir}/nfl_predictions_week{current_week}_{today.strftime('%Y%m%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"📄 Saved text report to {report_file}")
    
    # Save markdown report
    md_file = f"{results_dir}/nfl_predictions_week{current_week}_{today.strftime('%Y%m%d')}.md"
    with open(md_file, 'w') as f:
        f.write(markdown_report)
    print(f"📝 Saved markdown report to {md_file}")
    
    # Save JSON data for further analysis
    json_file = f"{results_dir}/nfl_predictions_week{current_week}_{today.strftime('%Y%m%d')}.json"
    
    # Convert results to JSON-serializable format
    json_data = {
        'date': today.isoformat(),
        'week': current_week,
        'n_simulations': n_simulations,
        'completed_games': results['completed_games'],
        'remaining_games': results['remaining_games'],
        'standings': results['teams_data'],
        'probabilities': {
            name: {
                'playoff_pct': (r['playoff_count'] / n_simulations) * 100,
                'division_winner_pct': (r['division_winner_count'] / n_simulations) * 100,
                'wildcard_pct': (r['wildcard_count'] / n_simulations) * 100,
                'avg_wins': r.get('avg_wins', 0),
                'seed_distribution': {
                    str(seed): (r['seed_counts'][seed] / n_simulations) * 100
                    for seed in range(1, 8)
                }
            }
            for name, r in results['results'].items()
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📊 Saved JSON data to {json_file}")


def main():
    """Main entry point for scheduled runs"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scheduled NFL Playoff Predictor')
    parser.add_argument('--force', action='store_true', 
                       help='Run even if not Tuesday-Thursday')
    parser.add_argument('--simulations', '-n', type=int, default=100000,
                       help='Number of simulations (default: 100000)')
    parser.add_argument('--no-clear-cache', action='store_true',
                       help='Do not clear cache before running')
    args = parser.parse_args()
    
    print("🏈 NFL Playoff Predictor - Scheduled Run")
    print("=" * 50)
    
    # Check if we should run
    should_run, reason = should_run_today()
    print(f"📅 {reason}")
    
    if not should_run and not args.force:
        print("Use --force to run anyway")
        return
    
    current_week = get_current_nfl_week()
    print(f"🗓️  NFL Week: {current_week}")
    print(f"🎲 Simulations: {args.simulations:,}")
    
    # Clear caches for fresh data
    if not args.no_clear_cache:
        print("\n🔄 Clearing caches for fresh data...")
        clear_caches()
    
    # Run simulation
    print(f"\n🚀 Running {args.simulations:,} simulations...")
    try:
        results = run_simulation(n_simulations=args.simulations)
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Format results
    report = format_results(results, args.simulations)
    markdown_report = format_results_markdown(results, args.simulations)
    
    # Display text report
    print("\n" + report)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(report, markdown_report, results, args.simulations)
    
    print("\n✅ Scheduled run complete!")


if __name__ == "__main__":
    main()
