"""
NFL Playoff Predictor - Backtest Validation

This module validates the prediction model by comparing simulated playoff
probabilities against actual historical outcomes.

Key metrics:
- Brier Score: Measures probability calibration (lower = better, target <0.22)
- Win Prediction Accuracy: % of games correctly predicted
- Playoff Prediction Accuracy: How well we predicted playoff teams
- Calibration: Are 70% predictions correct 70% of the time?
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from epa_loader import load_team_epa
from nfl_tiebreakers import Game, TEAM_TO_CONFERENCE, TEAM_TO_DIVISION

# Try to import injury modules (optional enhancement for current season)
try:
    from injury_loader import load_injury_data, load_snap_counts
    from player_impact import get_all_team_impacts
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False


# Team abbreviation to conference/division mapping
# nfl_data_py uses abbreviations, our tiebreakers use full names
ABBREV_TO_CONF = {
    # AFC East
    'BUF': 'AFC', 'MIA': 'AFC', 'NE': 'AFC', 'NYJ': 'AFC',
    # AFC North
    'BAL': 'AFC', 'CIN': 'AFC', 'CLE': 'AFC', 'PIT': 'AFC',
    # AFC South
    'HOU': 'AFC', 'IND': 'AFC', 'JAX': 'AFC', 'TEN': 'AFC',
    # AFC West
    'DEN': 'AFC', 'KC': 'AFC', 'LV': 'AFC', 'LAC': 'AFC',
    # NFC East
    'DAL': 'NFC', 'NYG': 'NFC', 'PHI': 'NFC', 'WAS': 'NFC',
    # NFC North
    'CHI': 'NFC', 'DET': 'NFC', 'GB': 'NFC', 'MIN': 'NFC',
    # NFC South
    'ATL': 'NFC', 'CAR': 'NFC', 'NO': 'NFC', 'TB': 'NFC',
    # NFC West
    'ARI': 'NFC', 'LA': 'NFC', 'LAR': 'NFC', 'SF': 'NFC', 'SEA': 'NFC',
}

ABBREV_TO_DIV = {
    # AFC East
    'BUF': 'East', 'MIA': 'East', 'NE': 'East', 'NYJ': 'East',
    # AFC North
    'BAL': 'North', 'CIN': 'North', 'CLE': 'North', 'PIT': 'North',
    # AFC South
    'HOU': 'South', 'IND': 'South', 'JAX': 'South', 'TEN': 'South',
    # AFC West
    'DEN': 'West', 'KC': 'West', 'LV': 'West', 'LAC': 'West',
    # NFC East
    'DAL': 'East', 'NYG': 'East', 'PHI': 'East', 'WAS': 'East',
    # NFC North
    'CHI': 'North', 'DET': 'North', 'GB': 'North', 'MIN': 'North',
    # NFC South
    'ATL': 'South', 'CAR': 'South', 'NO': 'South', 'TB': 'South',
    # NFC West
    'ARI': 'West', 'LA': 'West', 'LAR': 'West', 'SF': 'West', 'SEA': 'West',
}

# Map abbreviations to full names for simulation compatibility
ABBREV_TO_FULL = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LA': 'Los Angeles Rams', 'LAR': 'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints',
    'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    season: int
    week: int
    brier_score: float
    win_accuracy: float
    playoff_accuracy: float
    n_games: int
    predictions: List[Dict]
    
    def to_dict(self) -> dict:
        return {
            'season': self.season,
            'week': self.week,
            'brier_score': round(self.brier_score, 4),
            'win_accuracy': round(self.win_accuracy, 4),
            'playoff_accuracy': round(self.playoff_accuracy, 4),
            'n_games': self.n_games
        }


class NFLBacktester:
    """
    Validates NFL prediction models against historical data.
    
    Uses nfl_data_py to fetch historical schedules, standings, and results.
    Compares model predictions to actual outcomes.
    """
    
    # Actual playoff teams for validation (using full team names)
    ACTUAL_PLAYOFFS = {
        2024: {
            'AFC': ['Kansas City Chiefs', 'Buffalo Bills', 'Baltimore Ravens', 'Houston Texans', 
                    'Los Angeles Chargers', 'Pittsburgh Steelers', 'Denver Broncos'],
            'NFC': ['Detroit Lions', 'Philadelphia Eagles', 'Tampa Bay Buccaneers', 'Los Angeles Rams',
                    'Minnesota Vikings', 'Washington Commanders', 'Green Bay Packers']
        },
        2023: {
            'AFC': ['Baltimore Ravens', 'Buffalo Bills', 'Kansas City Chiefs', 'Houston Texans',
                    'Cleveland Browns', 'Miami Dolphins', 'Pittsburgh Steelers'],
            'NFC': ['San Francisco 49ers', 'Dallas Cowboys', 'Detroit Lions', 'Tampa Bay Buccaneers',
                    'Philadelphia Eagles', 'Los Angeles Rams', 'Green Bay Packers']
        }
    }
    
    def __init__(self, use_epa: bool = True):
        """
        Initialize backtester.
        
        Args:
            use_epa: Whether to use EPA-based model (True) or traditional model (False)
        """
        self.use_epa = use_epa
        self.results: List[BacktestResult] = []
    
    def fetch_season_data(self, season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch schedule and standings for a season.
        
        Returns:
            Tuple of (schedule_df, standings_df)
        """
        print(f"📥 Fetching {season} season data...")
        
        # Fetch schedule
        schedule = nfl.import_schedules([season])
        print(f"  ✅ Loaded {len(schedule)} games")
        
        return schedule
    
    def get_standings_at_week(self, schedule: pd.DataFrame, up_to_week: int) -> Dict[str, Dict]:
        """
        Calculate standings as of a specific week.
        
        Args:
            schedule: Full season schedule
            up_to_week: Calculate standings up to this week
        
        Returns:
            Dict of team -> standings info (using full team names)
        """
        # Filter to completed games up to this week
        completed = schedule[
            (schedule['week'] <= up_to_week) & 
            (schedule['home_score'].notna()) &
            (schedule['away_score'].notna())
        ].copy()
        
        standings = {}
        all_teams = set(schedule['home_team'].unique()) | set(schedule['away_team'].unique())
        
        for abbrev in all_teams:
            # Convert abbreviation to full name for simulation
            full_name = ABBREV_TO_FULL.get(abbrev, abbrev)
            
            home_games = completed[completed['home_team'] == abbrev]
            away_games = completed[completed['away_team'] == abbrev]
            
            home_wins = len(home_games[home_games['home_score'] > home_games['away_score']])
            home_losses = len(home_games[home_games['home_score'] < home_games['away_score']])
            home_ties = len(home_games[home_games['home_score'] == home_games['away_score']])
            
            away_wins = len(away_games[away_games['away_score'] > away_games['home_score']])
            away_losses = len(away_games[away_games['away_score'] < away_games['home_score']])
            away_ties = len(away_games[away_games['away_score'] == away_games['home_score']])
            
            wins = home_wins + away_wins
            losses = home_losses + away_losses
            ties = home_ties + away_ties
            
            # Points
            home_pf = home_games['home_score'].sum() if len(home_games) > 0 else 0
            home_pa = home_games['away_score'].sum() if len(home_games) > 0 else 0
            away_pf = away_games['away_score'].sum() if len(away_games) > 0 else 0
            away_pa = away_games['home_score'].sum() if len(away_games) > 0 else 0
            
            standings[full_name] = {
                'name': full_name,
                'w': wins,
                'l': losses,
                't': ties,
                'pf': int(home_pf + away_pf),
                'pa': int(home_pa + away_pa),
                'conf': ABBREV_TO_CONF.get(abbrev, 'Unknown'),
                'div': ABBREV_TO_DIV.get(abbrev, 'Unknown')
            }
        
        return standings
    
    def get_remaining_games(self, schedule: pd.DataFrame, from_week: int) -> List[Game]:
        """Get games from a specific week onwards (using full team names)."""
        remaining = schedule[schedule['week'] > from_week]
        
        games = []
        for _, row in remaining.iterrows():
            home_full = ABBREV_TO_FULL.get(row['home_team'], row['home_team'])
            away_full = ABBREV_TO_FULL.get(row['away_team'], row['away_team'])
            games.append(Game(
                week=int(row['week']),
                home_team=home_full,
                away_team=away_full,
                home_score=None,
                away_score=None,
                completed=False
            ))
        
        return games
    
    def get_completed_games(self, schedule: pd.DataFrame, up_to_week: int) -> List[Game]:
        """Get completed games up to a specific week (using full team names)."""
        completed = schedule[
            (schedule['week'] <= up_to_week) & 
            (schedule['home_score'].notna())
        ]
        
        games = []
        for _, row in completed.iterrows():
            home_full = ABBREV_TO_FULL.get(row['home_team'], row['home_team'])
            away_full = ABBREV_TO_FULL.get(row['away_team'], row['away_team'])
            games.append(Game(
                week=int(row['week']),
                home_team=home_full,
                away_team=away_full,
                home_score=int(row['home_score']),
                away_score=int(row['away_score']),
                completed=True
            ))
        
        return games
        
        return games
    
    def simulate_from_week(
        self, 
        season: int, 
        schedule: pd.DataFrame,
        from_week: int,
        n_simulations: int = 1000
    ) -> Dict[str, Dict]:
        """
        Run simulation from a specific point in the season.
        
        Args:
            season: NFL season year
            schedule: Full schedule DataFrame
            from_week: Week to simulate from
            n_simulations: Number of Monte Carlo simulations
        
        Returns:
            Simulation results dict
        """
        from advanced_simulation import run_advanced_simulation, build_season_data_from_standings
        
        # Get standings and games at this point
        standings_dict = self.get_standings_at_week(schedule, from_week)
        standings_list = list(standings_dict.values())
        
        completed_games = self.get_completed_games(schedule, from_week)
        remaining_games = self.get_remaining_games(schedule, from_week)
        
        if len(remaining_games) == 0:
            print(f"  ⚠️  No remaining games from week {from_week}")
            return {}
        
        # Load injury data for current season (not available for historical)
        injury_impacts = None
        if INJURIES_AVAILABLE:
            try:
                injuries_df = load_injury_data(season=season)
                snap_counts_df = load_snap_counts(season=season)
                injury_impacts = get_all_team_impacts(injuries_df, snap_counts_df, from_week)
                if injury_impacts:
                    print(f"  📋 Loaded injury impacts for {len(injury_impacts)} teams")
            except Exception as e:
                print(f"  ⚠️ Injury data unavailable: {e}")

        # Run simulation with correct season for EPA data
        results = run_advanced_simulation(
            standings=standings_list,
            completed_games=completed_games,
            remaining_games=remaining_games,
            n_simulations=n_simulations,
            show_progress=False,
            use_epa=self.use_epa,
            season=season,
            injury_impacts=injury_impacts
        )
        
        return results
    
    def calculate_playoff_accuracy(
        self, 
        predictions: Dict[str, Dict], 
        actual_playoffs: Dict[str, List[str]],
        n_sims: int
    ) -> float:
        """
        Calculate how accurately we predicted playoff teams.
        
        For each team that actually made playoffs, check our predicted probability.
        Higher probabilities for actual playoff teams = better.
        """
        correct = 0
        total = 0
        
        for conf, teams in actual_playoffs.items():
            for team in teams:
                if team in predictions:
                    prob = predictions[team]['playoff_count'] / n_sims
                    # Consider "correct" if we gave them >50% playoff odds
                    if prob > 0.5:
                        correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_brier_score(
        self, 
        game_predictions: List[Dict]
    ) -> float:
        """
        Calculate Brier score for game predictions.
        
        Brier = mean((prob - outcome)^2)
        - 0.0 = perfect predictions
        - 0.25 = random guessing (50/50)
        - <0.22 = good NFL prediction model
        """
        if not game_predictions:
            return 0.25
        
        squared_errors = []
        for pred in game_predictions:
            prob = pred['home_win_prob']
            outcome = 1.0 if pred['home_won'] else 0.0
            squared_errors.append((prob - outcome) ** 2)
        
        return np.mean(squared_errors)
    
    def backtest_season(
        self, 
        season: int, 
        from_week: int = 14,
        n_simulations: int = 1000
    ) -> BacktestResult:
        """
        Run full backtest for a season.
        
        Args:
            season: NFL season year
            from_week: Week to start predictions from
            n_simulations: Simulations per prediction
        
        Returns:
            BacktestResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"  BACKTESTING {season} SEASON (from Week {from_week})")
        print(f"{'='*60}")
        
        # Fetch season data
        schedule = self.fetch_season_data(season)
        
        # Run simulation from this week
        print(f"\n🎲 Simulating from Week {from_week}...")
        predictions = self.simulate_from_week(season, schedule, from_week, n_simulations)
        
        if not predictions:
            return BacktestResult(
                season=season,
                week=from_week,
                brier_score=0.25,
                win_accuracy=0.5,
                playoff_accuracy=0.0,
                n_games=0,
                predictions=[]
            )
        
        # Calculate game-by-game accuracy
        remaining = schedule[schedule['week'] > from_week]
        game_preds = []
        correct_picks = 0
        total_picks = 0
        
        from advanced_simulation import EPAGameSimulator, EPA_AVAILABLE
        
        # Load EPA for game predictions
        if self.use_epa and EPA_AVAILABLE:
            epa_df = load_team_epa(season=season, force_refresh=False)
            simulator = EPAGameSimulator(epa_df=epa_df)
        else:
            simulator = None
        
        for _, row in remaining.iterrows():
            if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                continue
            
            home = row['home_team']
            away = row['away_team']
            home_won = row['home_score'] > row['away_score']
            
            # Get win probability
            if simulator:
                probs = simulator.get_win_probability(home, away, n_sims=100)
                home_prob = probs['home_win']
            else:
                # Simple baseline: home team wins ~57% historically
                home_prob = 0.57
            
            game_preds.append({
                'home': home,
                'away': away,
                'home_win_prob': home_prob,
                'home_won': home_won,
                'week': int(row['week'])
            })
            
            # Count correct picks (predict winner with >50%)
            if (home_prob > 0.5 and home_won) or (home_prob < 0.5 and not home_won):
                correct_picks += 1
            total_picks += 1
        
        # Calculate metrics
        brier = self.calculate_brier_score(game_preds)
        win_acc = correct_picks / total_picks if total_picks > 0 else 0.5
        
        # Playoff accuracy
        actual = self.ACTUAL_PLAYOFFS.get(season, {'AFC': [], 'NFC': []})
        playoff_acc = self.calculate_playoff_accuracy(predictions, actual, n_simulations)
        
        result = BacktestResult(
            season=season,
            week=from_week,
            brier_score=brier,
            win_accuracy=win_acc,
            playoff_accuracy=playoff_acc,
            n_games=total_picks,
            predictions=game_preds
        )
        
        self.results.append(result)
        
        # Print results
        print(f"\n📊 BACKTEST RESULTS ({season} from Week {from_week}):")
        print("-" * 40)
        print(f"  Games predicted: {total_picks}")
        print(f"  Win accuracy:    {win_acc*100:.1f}% ({correct_picks}/{total_picks})")
        print(f"  Brier score:     {brier:.4f} (lower = better, <0.22 = good)")
        print(f"  Playoff accuracy: {playoff_acc*100:.1f}% of playoff teams predicted >50%")
        
        return result
    
    def compare_models(self, season: int, from_week: int = 14, n_simulations: int = 1000):
        """
        Compare EPA model vs traditional model.
        """
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON: {season} from Week {from_week}")
        print(f"{'='*60}")
        
        # EPA model
        print("\n🔬 Testing EPA-based Poisson model...")
        self.use_epa = True
        epa_result = self.backtest_season(season, from_week, n_simulations)
        
        # Traditional model
        print("\n🔬 Testing traditional Gaussian model...")
        self.use_epa = False
        trad_result = self.backtest_season(season, from_week, n_simulations)
        
        # Compare
        print(f"\n{'='*60}")
        print(f"  COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"\n{'Metric':<20} {'EPA Model':>15} {'Traditional':>15} {'Winner':>12}")
        print("-" * 62)
        
        # Brier (lower = better)
        brier_winner = "EPA" if epa_result.brier_score < trad_result.brier_score else "Traditional"
        print(f"{'Brier Score':<20} {epa_result.brier_score:>15.4f} {trad_result.brier_score:>15.4f} {brier_winner:>12}")
        
        # Win accuracy (higher = better)
        win_winner = "EPA" if epa_result.win_accuracy > trad_result.win_accuracy else "Traditional"
        print(f"{'Win Accuracy':<20} {epa_result.win_accuracy*100:>14.1f}% {trad_result.win_accuracy*100:>14.1f}% {win_winner:>12}")
        
        # Playoff accuracy (higher = better)
        po_winner = "EPA" if epa_result.playoff_accuracy > trad_result.playoff_accuracy else "Traditional"
        print(f"{'Playoff Accuracy':<20} {epa_result.playoff_accuracy*100:>14.1f}% {trad_result.playoff_accuracy*100:>14.1f}% {po_winner:>12}")
        
        return epa_result, trad_result
    
    def save_results(self, filepath: str = "backtest_results.json"):
        """Save all backtest results to JSON."""
        data = {
            'generated': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved results to {filepath}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest NFL prediction model")
    parser.add_argument("--season", type=int, default=2024, help="Season to backtest")
    parser.add_argument("--week", type=int, default=14, help="Week to simulate from")
    parser.add_argument("--sims", type=int, default=1000, help="Simulations per run")
    parser.add_argument("--compare", action="store_true", help="Compare EPA vs traditional model")
    parser.add_argument("--no-epa", action="store_true", help="Use traditional model only")
    args = parser.parse_args()
    
    backtester = NFLBacktester(use_epa=not args.no_epa)
    
    if args.compare:
        backtester.compare_models(args.season, args.week, args.sims)
    else:
        backtester.backtest_season(args.season, args.week, args.sims)
    
    backtester.save_results()


if __name__ == "__main__":
    main()
