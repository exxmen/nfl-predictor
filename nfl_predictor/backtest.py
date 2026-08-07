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
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

from .epa import load_team_epa
from .tiebreakers import Game, TEAM_TO_CONFERENCE, TEAM_TO_DIVISION

# Try to import injury modules (optional enhancement for current season)
try:
    from .injuries import load_injury_data, load_snap_counts
    from .player_impact import get_all_team_impacts
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False

try:
    from .simulation import run_advanced_simulation, build_season_data_from_standings
    from .simulation import EPAGameSimulator, EPA_AVAILABLE
    from .intangibles import IntangiblesConfig
    INTANGIBLES_AVAILABLE = True
    SIMULATION_AVAILABLE = True
except ImportError:
    INTANGIBLES_AVAILABLE = False
    SIMULATION_AVAILABLE = False


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
    log_loss: Optional[float] = None
    ece: Optional[float] = None
    market_brier: Optional[float] = None
    market_win_accuracy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'season': self.season,
            'week': self.week,
            'brier_score': round(self.brier_score, 4),
            'win_accuracy': round(self.win_accuracy, 4),
            'playoff_accuracy': round(self.playoff_accuracy, 4),
            'log_loss': round(self.log_loss, 4) if self.log_loss is not None else None,
            'ece': round(self.ece, 4) if self.ece is not None else None,
            'market_brier': round(self.market_brier, 4) if self.market_brier is not None else None,
            'market_win_accuracy': round(self.market_win_accuracy, 4) if self.market_win_accuracy is not None else None,
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
        },
        2022: {
            'AFC': ['Kansas City Chiefs', 'Buffalo Bills', 'Cincinnati Bengals', 'Jacksonville Jaguars',
                    'Los Angeles Chargers', 'Miami Dolphins', 'Baltimore Ravens'],
            'NFC': ['Philadelphia Eagles', 'San Francisco 49ers', 'Minnesota Vikings', 'Tampa Bay Buccaneers',
                    'Dallas Cowboys', 'New York Giants', 'Seattle Seahawks']
        },
        2021: {
            'AFC': ['Tennessee Titans', 'Kansas City Chiefs', 'Buffalo Bills', 'Cincinnati Bengals',
                    'Las Vegas Raiders', 'New England Patriots', 'Pittsburgh Steelers'],
            'NFC': ['Green Bay Packers', 'Tampa Bay Buccaneers', 'Dallas Cowboys', 'Los Angeles Rams',
                    'Arizona Cardinals', 'San Francisco 49ers', 'Philadelphia Eagles']
        },
    }
    
    def __init__(self, use_epa: bool = True, use_intangibles: bool = False, intangibles_config: Optional[IntangiblesConfig] = None, market_weight: float = 0.0):
        """
        Initialize backtester.

        Args:
            use_epa: Whether to use EPA-based model (True) or traditional model (False)
            use_intangibles: Whether to use intangibles adjustments
            intangibles_config: Configuration for intangibles adjustments
            market_weight: Weight (0..1) to blend toward consensus spreads when
                computing win probabilities. 0 = pure EPA model (no anchoring).
        """
        self.use_epa = use_epa
        self.use_intangibles = use_intangibles
        self.intangibles_config = intangibles_config
        self.market_weight = market_weight
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

            # Determine if Thursday/Monday night from gametime and weekday
            is_thursday = row['weekday'] == 'Thu' and row['gametime'] == '20:15'
            is_monday = row['weekday'] == 'Mon' and row['gametime'] == '20:15'

            games.append(Game(
                week=int(row['week']),
                home_team=home_full,
                away_team=away_full,
                home_score=None,
                away_score=None,
                completed=False,
                gameday=str(row['gameday']),
                gametime=str(row['gametime']),
                home_rest=int(row['home_rest']) if pd.notna(row['home_rest']) else None,
                away_rest=int(row['away_rest']) if pd.notna(row['away_rest']) else None,
                is_thursday_night=is_thursday,
                is_monday_night=is_monday,
                is_division=bool(row['div_game']) if pd.notna(row['div_game']) else False,
                temp=int(row['temp']) if pd.notna(row['temp']) else None,
                wind=float(row['wind']) if pd.notna(row['wind']) else None,
                home_spread=float(row['spread_line']) if pd.notna(row.get('spread_line')) else None
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

            # Determine if Thursday/Monday night from gametime and weekday
            is_thursday = row['weekday'] == 'Thu' and row['gametime'] == '20:15'
            is_monday = row['weekday'] == 'Mon' and row['gametime'] == '20:15'

            games.append(Game(
                week=int(row['week']),
                home_team=home_full,
                away_team=away_full,
                home_score=int(row['home_score']),
                away_score=int(row['away_score']),
                completed=True,
                gameday=str(row['gameday']),
                gametime=str(row['gametime']),
                home_rest=int(row['home_rest']) if pd.notna(row['home_rest']) else None,
                away_rest=int(row['away_rest']) if pd.notna(row['away_rest']) else None,
                is_thursday_night=is_thursday,
                is_monday_night=is_monday,
                is_division=bool(row['div_game']) if pd.notna(row['div_game']) else False,
                temp=int(row['temp']) if pd.notna(row['temp']) else None,
                wind=float(row['wind']) if pd.notna(row['wind']) else None
            ))

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
        from .simulation import run_advanced_simulation, build_season_data_from_standings
        
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
            injury_impacts=injury_impacts,
            use_intangibles=self.use_intangibles,
            intangibles_config=self.intangibles_config,
            market_weight=self.market_weight
        )
        
        return results
    
    def get_actual_playoffs(self, season: int, schedule: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
        """
        Get the actual playoff field for a season.

        Prefers deriving from the schedule's game_type column (WC/DIV/CON/SB
        games identify the teams that made the playoffs) so any season works,
        and falls back to the hardcoded ACTUAL_PLAYOFFS dict when schedule data
        or the game_type column is unavailable.
        """
        hardcoded = self.ACTUAL_PLAYOFFS.get(season)
        if schedule is not None and 'game_type' in schedule.columns:
            post = schedule[schedule['game_type'].isin(['WC', 'DIV', 'CON', 'SB'])]
            if not post.empty:
                from .team_names import to_full_name
                teams_in_post = set(post['home_team'].tolist() + post['away_team'].tolist())
                afc, nfc = [], []
                skipped = 0
                for abbr in sorted(teams_in_post):
                    conf = ABBREV_TO_CONF.get(abbr)
                    if conf not in ('AFC', 'NFC'):
                        # Unknown abbreviation: skip rather than mis-bucket it
                        skipped += 1
                        continue
                    full = to_full_name(abbr)
                    (afc if conf == 'AFC' else nfc).append(full)
                if skipped:
                    logger.warning("get_actual_playoffs: skipped %d unknown team abbreviation(s)", skipped)
                if afc and nfc:
                    return {'AFC': afc, 'NFC': nfc}
        if hardcoded is not None:
            return hardcoded
        return {'AFC': [], 'NFC': []}

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

    def calculate_log_loss(self, game_predictions: List[Dict]) -> float:
        """
        Calculate log-loss for binary predictions.
        Log-loss = -mean(y·ln(p) + (1−y)·ln(1−p)).
        More sensitive than Brier to confident-but-wrong predictions.
        """
        if not game_predictions:
            return 0.693  # -ln(0.5), the coin-flip baseline
        losses = []
        for pred in game_predictions:
            p = min(0.999, max(0.001, pred['home_win_prob']))  # clip to avoid log(0)
            y = 1.0 if pred['home_won'] else 0.0
            losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean(losses))

    def calculate_ece(self, game_predictions: List[Dict], n_bins: int = 10) -> float:
        """
        Expected Calibration Error: |acc − conf| weighted by bin size.
        Measures whether 70% predictions are right 70% of the time.
        """
        if not game_predictions:
            return 0.0
        probs = np.array([p['home_win_prob'] for p in game_predictions])
        outcomes = np.array([1.0 if p['home_won'] else 0.0 for p in game_predictions])

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges[0] = -0.001  # include p=0.0
        total = len(probs)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs > lo) & (probs <= hi)
            n = int(np.sum(mask))
            if n == 0:
                continue
            conf = float(np.mean(probs[mask]))
            acc = float(np.mean(outcomes[mask]))
            ece += (n / total) * abs(acc - conf)
        return ece

    @staticmethod
    def market_prob_from_spread(home_spread: Optional[float]) -> Optional[float]:
        """Convert a closing spread to an implied home win probability.
        Uses a logistic approximation. VERIFIED empirically against 285 real
        2024 games: nflverse `spread_line` has POSITIVE = home favored, so
        p = 1 / (1 + exp(-0.294 * spread)) yields p > 0.5 when home favored."""
        if home_spread is None:
            return None
        return 1.0 / (1.0 + np.exp(-0.294 * home_spread))

    def calculate_market_benchmark(self, game_predictions: List[Dict]) -> Dict:
        """
        Compute Brier + win accuracy a market-implied baseline would achieve,
        using each game's home_spread. Missing spreads are skipped.
        """
        if not game_predictions:
            return {'market_brier': None, 'market_win_accuracy': None, 'n_with_spread': 0}
        brier_terms = []
        correct = 0
        n = 0
        for pred in game_predictions:
            p = self.market_prob_from_spread(pred.get('home_spread'))
            if p is None:
                continue
            y = 1.0 if pred['home_won'] else 0.0
            brier_terms.append((p - y) ** 2)
            if p == 0.5:
                # Pick'em: no favorite, so a correct/incorrect call is undefined.
                # Credit half so accuracy isn't biased downward by their presence.
                correct += 0.5
            elif (p > 0.5 and y == 1.0) or (p < 0.5 and y == 0.0):
                correct += 1
            n += 1
        return {
            'market_brier': float(np.mean(brier_terms)) if brier_terms else None,
            'market_win_accuracy': correct / n if n > 0 else None,
            'n_with_spread': n
        }
    
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
        
        from .simulation import EPAGameSimulator, EPA_AVAILABLE
        
        # Load EPA for game predictions
        if self.use_epa and EPA_AVAILABLE:
            epa_df = load_team_epa(season=season, force_refresh=False)
            simulator = EPAGameSimulator(epa_df=epa_df, market_weight=self.market_weight)
        else:
            simulator = None
        
        for _, row in remaining.iterrows():
            if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                continue
            
            home = row['home_team']
            away = row['away_team']
            home_won = row['home_score'] > row['away_score']
            spread = float(row['spread_line']) if pd.notna(row.get('spread_line')) else None
            
            # Get win probability (with market anchoring when a spread exists)
            if simulator:
                game_data = {'home_spread': spread} if spread is not None else {}
                probs = simulator.get_win_probability(home, away, n_sims=100, game_data=game_data)
                home_prob = probs['home_win']
            else:
                # Simple baseline: home team wins ~57% historically
                home_prob = 0.57
            
            game_preds.append({
                'home': home,
                'away': away,
                'home_win_prob': home_prob,
                'home_won': home_won,
                'week': int(row['week']),
                'home_spread': spread
            })
            
            # Count correct picks (predict winner with >50%)
            if (home_prob > 0.5 and home_won) or (home_prob < 0.5 and not home_won):
                correct_picks += 1
            total_picks += 1
        
        # Calculate metrics
        brier = self.calculate_brier_score(game_preds)
        win_acc = correct_picks / total_picks if total_picks > 0 else 0.5
        log_loss = self.calculate_log_loss(game_preds)
        ece = self.calculate_ece(game_preds)
        mkt = self.calculate_market_benchmark(game_preds)

        # Playoff accuracy
        actual = self.get_actual_playoffs(season, schedule)
        playoff_acc = self.calculate_playoff_accuracy(predictions, actual, n_simulations)

        result = BacktestResult(
            season=season,
            week=from_week,
            brier_score=brier,
            win_accuracy=win_acc,
            playoff_accuracy=playoff_acc,
            n_games=total_picks,
            predictions=game_preds,
            log_loss=log_loss,
            ece=ece,
            market_brier=mkt['market_brier'],
            market_win_accuracy=mkt['market_win_accuracy']
        )
        
        self.results.append(result)
        
        # Print results
        print(f"\n📊 BACKTEST RESULTS ({season} from Week {from_week}):")
        print("-" * 40)
        print(f"  Games predicted: {total_picks}")
        print(f"  Win accuracy:    {win_acc*100:.1f}% ({correct_picks}/{total_picks})")
        print(f"  Brier score:     {brier:.4f} (lower = better, <0.22 = good)")
        print(f"  Log-loss:        {log_loss:.4f} (lower = better, 0.693 = coin flip)")
        print(f"  ECE (calib):     {ece:.4f} (<0.08 = well calibrated)")
        print(f"  Playoff accuracy: {playoff_acc*100:.1f}% of playoff teams predicted >50%")
        if mkt['market_brier'] is not None:
            diff = brier - mkt['market_brier']
            print(f"  Market bench:    Brier {mkt['market_brier']:.4f} (vs model {brier:+.4f} "
                  f"{'BEATS' if diff < 0 else 'TRAILS'}), acc {mkt['market_win_accuracy']*100:.1f}% "
                  f"({mkt['n_with_spread']} games with spread)")
        else:
            print("  Market bench:    unavailable (no spread_line in schedule data)")
        
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
    
    def save_results(self, filepath: str = "results/backtest_results.json"):
        """Save all backtest results to JSON."""
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'generated': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved results to {filepath}")

    def backtest_multi_season(self, seasons: List[int], from_week: int = 14,
                              n_simulations: int = 1000) -> List[BacktestResult]:
        """Run backtests across multiple seasons and summarize."""
        for season in seasons:
            self.backtest_season(season, from_week, n_simulations)
        return self.results

    def write_markdown_summary(self, filepath: str = "results/backtest_summary.md") -> str:
        """Write a markdown summary table of all backtest results (and return it)."""
        import os
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        lines = []
        lines.append("# NFL Predictor Backtest Summary\n")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        lines.append("| Season | Week | Games | Win% | Brier | LogLoss | ECE | Market Brier | Market Win% | Playoff% |")
        lines.append("|:------:|:----:|:-----:|:----:|:-----:|:-------:|:---:|:------------:|:-----------:|:--------:|")
        for r in self.results:
            mkt_b = f"{r.market_brier:.4f}" if r.market_brier is not None else "—"
            mkt_a = f"{r.market_win_accuracy*100:.1f}%" if r.market_win_accuracy is not None else "—"
            ll = f"{r.log_loss:.4f}" if r.log_loss is not None else "—"
            ece = f"{r.ece:.4f}" if r.ece is not None else "—"
            lines.append(
                f"| {r.season} | {r.week} | {r.n_games} | {r.win_accuracy*100:.1f}% "
                f"| {r.brier_score:.4f} | {ll} | {ece} | {mkt_b} | {mkt_a} | {r.playoff_accuracy*100:.1f}% |"
            )
        lines.append("")
        lines.append("* Brier <0.22 = good, LogLoss 0.693 = coin flip, ECE <0.08 = well calibrated.")
        lines.append("* Model TRAILS the market benchmark when its Brier exceeds the closing-line Brier — a signal to blend harder toward consensus spreads.")

        markdown = "\n".join(lines)
        with open(filepath, 'w') as f:
            f.write(markdown)
        print(f"\n📄 Wrote markdown summary to {filepath}")
        return markdown


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest NFL prediction model")
    parser.add_argument("--season", type=int, default=2024, help="Season to backtest")
    parser.add_argument("--seasons", nargs="+", type=int, default=None,
                        help="Multiple seasons to backtest (e.g. --seasons 2021 2022 2023 2024)")
    parser.add_argument("--week", type=int, default=14, help="Week to simulate from")
    parser.add_argument("--sims", type=int, default=1000, help="Simulations per run")
    parser.add_argument("--compare", action="store_true", help="Compare EPA vs traditional model")
    parser.add_argument("--no-epa", action="store_true", help="Use traditional model only")
    parser.add_argument("--intangibles", action="store_true", help="Enable intangibles adjustments")
    parser.add_argument("--compare-intangibles", action="store_true", help="Compare with vs without intangibles")
    parser.add_argument("--market", type=float, metavar="W", default=0.0,
                        help="Blend weight toward consensus spreads (e.g. 0.30). 0 = pure EPA model")
    parser.add_argument("--compare-market", action="store_true",
                        help="Compare pure EPA model vs market-anchored model")
    args = parser.parse_args()

    intangibles_config = None
    if args.intangibles or args.compare_intangibles:
        intangibles_config = IntangiblesConfig(
            use_rest_days=True,
            use_turnover_luck=True,
            use_travel_adjustment=True,
            use_division_familiarity=True,
            use_weather=False  # Disabled for backtest (requires API)
        )

    backtester = NFLBacktester(
        use_epa=not args.no_epa,
        use_intangibles=args.intangibles,
        intangibles_config=intangibles_config,
        market_weight=args.market
    )

    if args.compare_market:
        # Compare pure EPA model vs market-anchored model
        print("\n" + "="*60)
        print("  MARKET ANCHORING COMPARISON")
        print("="*60)

        print("\n🔬 Running backtest WITHOUT market anchoring (pure EPA)...")
        backtester.market_weight = 0.0
        result_plain = backtester.backtest_season(args.season, args.week, args.sims)

        print("\n🔬 Running backtest WITH market anchoring...")
        # Use the user's --market W value (default 0.30) as the comparison weight
        anchored_weight = args.market if args.market > 0 else 0.30
        backtester.market_weight = anchored_weight
        result_market = backtester.backtest_season(args.season, args.week, args.sims)

        print(f"\n{'='*60}")
        print(f"  MARKET ANCHORING COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"\n{'Metric':<18} {'Pure EPA':>12} {'Anchored':>12} {'Winner':>12}")
        print("-" * 54)
        print(f"  (anchored weight: {anchored_weight:.2f})")
        brier_winner = "Anchored" if result_market.brier_score < result_plain.brier_score else "Pure EPA"
        ll_winner = "Anchored" if result_market.log_loss < result_plain.log_loss else "Pure EPA"
        win_winner = "Anchored" if result_market.win_accuracy > result_plain.win_accuracy else "Pure EPA"
        print(f"{'Brier Score':<18} {result_plain.brier_score:>12.4f} {result_market.brier_score:>12.4f} {brier_winner:>12}")
        print(f"{'Log-loss':<18} {result_plain.log_loss:>12.4f} {result_market.log_loss:>12.4f} {ll_winner:>12}")
        print(f"{'Win accuracy':<18} {result_plain.win_accuracy*100:>11.1f}% {result_market.win_accuracy*100:>11.1f}% {win_winner:>12}")
        print(f"{'ECE (calib)':<18} {result_plain.ece:>12.4f} {result_market.ece:>12.4f}")
        return
    elif args.compare_intangibles:
        # Compare with vs without intangibles
        print("\n" + "="*60)
        print("  INTANGIBLES COMPARISON")
        print("="*60)

        # Without intangibles
        print("\n🔬 Running backtest WITHOUT intangibles...")
        backtester.use_intangibles = False
        result_without = backtester.backtest_season(args.season, args.week, args.sims)

        # With intangibles
        print("\n🔬 Running backtest WITH intangibles...")
        backtester.use_intangibles = True
        result_with = backtester.backtest_season(args.season, args.week, args.sims)

        # Compare
        print(f"\n{'='*60}")
        print(f"  INTANGIBLES COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"\n{'Metric':<20} {'Without':>15} {'With':>15} {'Winner':>12}")
        print("-" * 62)

        # Brier (lower = better)
        brier_winner = "With" if result_with.brier_score < result_without.brier_score else "Without"
        print(f"{'Brier Score':<20} {result_without.brier_score:>15.4f} {result_with.brier_score:>15.4f} {brier_winner:>12}")

        # Win accuracy (higher = better)
        win_winner = "With" if result_with.win_accuracy > result_without.win_accuracy else "Without"
        print(f"{'Win Accuracy':<20} {result_without.win_accuracy*100:>14.1f}% {result_with.win_accuracy*100:>14.1f}% {win_winner:>12}")

        # Playoff accuracy (higher = better)
        po_winner = "With" if result_with.playoff_accuracy > result_without.playoff_accuracy else "Without"
        print(f"{'Playoff Accuracy':<20} {result_without.playoff_accuracy*100:>14.1f}% {result_with.playoff_accuracy*100:>14.1f}% {po_winner:>12}")
    elif args.compare:
        backtester.compare_models(args.season, args.week, args.sims)
    elif args.seasons:
        backtester.backtest_multi_season(args.seasons, args.week, args.sims)
    else:
        backtester.backtest_season(args.season, args.week, args.sims)

    backtester.save_results()
    backtester.write_markdown_summary()


if __name__ == "__main__":
    main()
