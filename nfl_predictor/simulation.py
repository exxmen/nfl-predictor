"""
Advanced NFL Playoff Simulation with Real Tiebreakers

This module provides the enhanced Monte Carlo simulation using:
- EPA (Expected Points Added) for matchup-adjusted scoring
- Poisson distribution for realistic NFL score modeling
- Real NFL tiebreaker rules
- Game-by-game simulation
- Progress tracking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from scipy.stats import poisson
import pandas as pd

from .tiebreakers import (
    Game,
    SeasonData,
    TeamStats,
    NFLTiebreaker,
    TEAM_TO_DIVISION,
    TEAM_TO_CONFERENCE,
    get_current_nfl_week
)

# Try to import EPA loader (optional enhancement)
try:
    from .epa import load_team_epa, get_league_averages
    EPA_AVAILABLE = True
except ImportError:
    EPA_AVAILABLE = False

# Try to import intangibles module
try:
    from .intangibles import IntangiblesCalculator, IntangiblesConfig
    INTANGIBLES_AVAILABLE = True
except ImportError:
    INTANGIBLES_AVAILABLE = False


# Momentum calculation constants
MIN_GAMES_FOR_MOMENTUM = 3  # Minimum games needed for meaningful momentum calculation
MIN_MARGIN_STD_DEV = 7.0  # Minimum std dev for point margins (prevents over-sensitivity)
MAX_MOMENTUM_ZSCORE = 3.0  # Clip momentum z-scores to prevent extreme outliers


def calculate_game_momentum(completed_games: List[Game], n_recent: int = 4) -> Dict[str, Dict[str, float]]:
    """
    Calculate team momentum from completed game margins.
    
    This provides TRUE current-season momentum using actual game results,
    rather than relying on EPA data which may not be available for current season.
    
    Momentum is calculated as z-score: (recent_margin - season_margin) / std_dev
    
    Args:
        completed_games: List of completed Game objects
        n_recent: Number of recent games to use for "hot/cold" detection (default: 4)
    
    Returns:
        Dict of team -> {'off_momentum': float, 'def_momentum': float, 'total_momentum': float}
    """
    from collections import defaultdict
    
    # Build per-team game history
    team_games: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)  # team -> [(week, points_for, points_against)]
    
    for game in completed_games:
        if not game.completed or game.home_score is None or game.away_score is None:
            continue
        
        # Home team's perspective
        team_games[game.home_team].append((game.week, game.home_score, game.away_score))
        # Away team's perspective
        team_games[game.away_team].append((game.week, game.away_score, game.home_score))
    
    momentum_data = {}
    
    for team, games in team_games.items():
        if len(games) < MIN_GAMES_FOR_MOMENTUM:
            continue
        
        # Sort by week (most recent last)
        games.sort(key=lambda x: x[0])
        
        # Calculate season averages
        all_margins = [pf - pa for week, pf, pa in games]
        all_pf = [pf for week, pf, pa in games]
        all_pa = [pa for week, pf, pa in games]
        
        season_margin = sum(all_margins) / len(all_margins)
        season_pf = sum(all_pf) / len(all_pf)
        season_pa = sum(all_pa) / len(all_pa)
        
        # Calculate std dev for z-score (use margin std dev, min threshold to avoid over-sensitivity)
        margin_std = max(MIN_MARGIN_STD_DEV, np.std(all_margins)) if len(all_margins) > 1 else MIN_MARGIN_STD_DEV
        
        # Get recent games (last n_recent)
        recent_games = games[-n_recent:]
        recent_margins = [pf - pa for week, pf, pa in recent_games]
        recent_pf = [pf for week, pf, pa in recent_games]
        recent_pa = [pa for week, pf, pa in recent_games]
        
        recent_margin = sum(recent_margins) / len(recent_margins)
        recent_pf_avg = sum(recent_pf) / len(recent_pf)
        recent_pa_avg = sum(recent_pa) / len(recent_pa)
        
        # Calculate momentum as z-score
        # Positive = team is outperforming their season average
        total_momentum = (recent_margin - season_margin) / margin_std
        
        # Offensive momentum: scoring more recently than usual
        off_momentum = (recent_pf_avg - season_pf) / max(MIN_MARGIN_STD_DEV, np.std(all_pf) if len(all_pf) > 1 else MIN_MARGIN_STD_DEV)
        
        # Defensive momentum: allowing fewer points recently than usual
        def_momentum = (season_pa - recent_pa_avg) / max(MIN_MARGIN_STD_DEV, np.std(all_pa) if len(all_pa) > 1 else MIN_MARGIN_STD_DEV)
        
        # Clip to reasonable range to prevent extreme outliers
        total_momentum = max(-MAX_MOMENTUM_ZSCORE, min(MAX_MOMENTUM_ZSCORE, total_momentum))
        off_momentum = max(-MAX_MOMENTUM_ZSCORE, min(MAX_MOMENTUM_ZSCORE, off_momentum))
        def_momentum = max(-MAX_MOMENTUM_ZSCORE, min(MAX_MOMENTUM_ZSCORE, def_momentum))
        
        momentum_data[team] = {
            'off_momentum': round(off_momentum, 3),
            'def_momentum': round(def_momentum, 3),
            'total_momentum': round(total_momentum, 3),
            'recent_games': len(recent_games),
            'season_games': len(games)
        }
    
    return momentum_data


class EPAGameSimulator:
    """
    Enhanced game simulator using EPA and Poisson scoring model.
    
    EPA (Expected Points Added) adjusts for game situation and opponent strength,
    providing more accurate matchup predictions than raw scoring averages.
    
    Poisson distribution models NFL scoring realistically (discrete, skewed toward
    lower scores) compared to Gaussian which can produce unrealistic scores.
    """
    
    # Home field advantage in points (historical NFL average ~2.5)
    HOME_ADVANTAGE = 2.5
    
    # League average PPG (used as baseline)
    LEAGUE_AVG_PPG = 22.0
    
    # EPA scaling factor: how much EPA/play affects expected points
    # ~0.1 EPA/play difference ≈ 3-4 points per game
    EPA_SCALING = 35.0  # Multiply EPA by this to get point adjustment
    
    # Momentum weight: how much recent form affects expected score
    # Value of 0.10 means momentum can adjust score by up to ±2 points
    MOMENTUM_WEIGHT = 0.10
    MOMENTUM_POINTS_PER_SIGMA = 2.0  # Points adjustment per standard deviation
    
    def __init__(self, epa_df: Optional[pd.DataFrame] = None, season_data: Optional['SeasonData'] = None, injury_impacts: Optional[Dict[str, Dict[str, float]]] = None, use_momentum: bool = True, game_momentum: Optional[Dict[str, Dict[str, float]]] = None, prefer_game_momentum: bool = False, intangibles_config: Optional[IntangiblesConfig] = None, intangibles_calculator: Optional['IntangiblesCalculator'] = None):
        """
        Initialize EPA-based simulator.

        Args:
            epa_df: DataFrame with team EPA stats (from epa_loader.py)
            season_data: Traditional SeasonData for fallback
            injury_impacts: Dict of team -> injury impact dict (from player_impact.py)
            use_momentum: Whether to apply momentum adjustments (default: True)
            game_momentum: Dict of team -> momentum from completed games
            prefer_game_momentum: If True, prefer game momentum over EPA momentum (for current season)
            intangibles_config: Configuration for intangibles adjustments
            intangibles_calculator: Pre-configured IntangiblesCalculator instance
        """
        self.epa_df = epa_df
        self.season_data = season_data
        self.injury_impacts = injury_impacts or {}
        self.use_momentum = use_momentum
        self.game_momentum = game_momentum or {}
        self.prefer_game_momentum = prefer_game_momentum  # True for current season
        self.intangibles_config = intangibles_config
        self.intangibles_calculator = intangibles_calculator

        # Build team lookup if EPA data available
        if epa_df is not None:
            self.team_epa = epa_df.set_index('team').to_dict('index')
            self.league_avg_ppg = epa_df['ppg'].mean() if 'ppg' in epa_df.columns else self.LEAGUE_AVG_PPG
            self.league_avg_off_epa = epa_df['off_epa'].mean()
            self.league_avg_def_epa = epa_df['def_epa'].mean()
            # Check if momentum data is available in EPA (from nflverse)
            self.has_epa_momentum = 'off_momentum' in epa_df.columns and 'def_momentum' in epa_df.columns
        else:
            self.team_epa = {}
            self.league_avg_ppg = self.LEAGUE_AVG_PPG
            self.league_avg_off_epa = 0.0
            self.league_avg_def_epa = 0.0
            self.has_epa_momentum = False
        
        # Determine which momentum source to use
        self.has_momentum = bool(self.game_momentum) if prefer_game_momentum else (self.has_epa_momentum or bool(self.game_momentum))
    
    def get_team_stats(self, team: str) -> dict:
        """Get EPA stats for a team, with fallback to league average."""
        if team in self.team_epa:
            return self.team_epa[team]
        
        # Fallback to league averages
        return {
            'off_epa': self.league_avg_off_epa,
            'def_epa': self.league_avg_def_epa,
            'ppg': self.league_avg_ppg,
            'ppg_allowed': self.league_avg_ppg,
            'off_momentum': 0.0,
            'def_momentum': 0.0
        }
    
    def get_momentum(self, team: str, side: str) -> float:
        """
        Get momentum score for a team (offensive or defensive).
        
        For current season: prefers game-based momentum (from actual completed games)
        For historical: uses EPA-based momentum (from that year's play-by-play)
        
        Args:
            team: Team abbreviation
            side: 'offense' or 'defense'
        
        Returns:
            Momentum z-score (positive = hot, negative = cold)
        """
        if not self.use_momentum:
            return 0.0
        
        # For current season, prefer game-based momentum (from actual 2025 games)
        if self.prefer_game_momentum and self.game_momentum and team in self.game_momentum:
            if side == 'offense':
                return self.game_momentum[team].get('off_momentum', 0.0)
            else:
                return self.game_momentum[team].get('def_momentum', 0.0)
        
        # For historical (backtest), use EPA-based momentum
        if self.has_epa_momentum:
            stats = self.team_epa.get(team, {})
            if side == 'offense':
                return stats.get('off_momentum', 0.0)
            else:
                return stats.get('def_momentum', 0.0)
        
        # Last resort fallback to game momentum
        if self.game_momentum and team in self.game_momentum:
            if side == 'offense':
                return self.game_momentum[team].get('off_momentum', 0.0)
            else:
                return self.game_momentum[team].get('def_momentum', 0.0)
        
        return 0.0
    
    def calculate_expected_score(self, offense_team: str, defense_team: str, is_home: bool = False) -> float:
        """
        Calculate expected score for a team using EPA adjustments and injury impacts.

        Formula:
        Expected = League_Avg + (Team_Off_EPA - Avg_Off_EPA) * Scale
                   - (Opp_Def_EPA - Avg_Def_EPA) * Scale + Home_Adv
                   + Injury_Adjustments

        Args:
            offense_team: Team on offense
            defense_team: Team on defense
            is_home: Whether offense team is home

        Returns:
            Expected points (lambda for Poisson)
        """
        off_stats = self.get_team_stats(offense_team)
        def_stats = self.get_team_stats(defense_team)

        # Start with team's average PPG (or league average if not available)
        base_ppg = off_stats.get('ppg', self.league_avg_ppg)

        # EPA adjustments relative to league average
        off_epa_adj = (off_stats['off_epa'] - self.league_avg_off_epa) * self.EPA_SCALING

        # Opponent's defense: higher def_epa means BETTER defense, reduces scoring
        def_epa_adj = (def_stats['def_epa'] - self.league_avg_def_epa) * self.EPA_SCALING

        # Expected score: base + offensive boost - defensive reduction
        expected = base_ppg + off_epa_adj - def_epa_adj

        # Apply injury adjustments
        if self.injury_impacts:
            # Offensive team's injuries reduce their expected score
            off_injuries = self.injury_impacts.get(offense_team, {})
            off_impact = off_injuries.get('offensive_impact', 0.0)
            expected *= (1 - off_impact)

            # Defensive team's injuries increase offensive team's expected score
            def_injuries = self.injury_impacts.get(defense_team, {})
            def_impact = def_injuries.get('defensive_impact', 0.0)
            expected *= (1 + def_impact * 0.5)  # Half effect for opponent injuries
        
        # Apply momentum adjustments (hot/cold streaks)
        if self.has_momentum and self.use_momentum:
            # Hot offense scores more, cold offense scores less
            off_momentum = self.get_momentum(offense_team, 'offense')
            # Hot defense reduces scoring, cold defense allows more
            def_momentum = self.get_momentum(defense_team, 'defense')
            
            # Apply momentum adjustment: ~2 points per standard deviation
            momentum_adj = off_momentum * self.MOMENTUM_POINTS_PER_SIGMA * self.MOMENTUM_WEIGHT
            momentum_adj -= def_momentum * self.MOMENTUM_POINTS_PER_SIGMA * self.MOMENTUM_WEIGHT
            expected += momentum_adj

        # Home field advantage
        if is_home:
            expected += self.HOME_ADVANTAGE

        # Apply intangibles adjustments (if available)
        # Note: Intangibles are applied per-game, not per-team
        # The game simulation will call this with additional intangibles context
        # For now, we apply intangibles at the game level in simulate_game()

        # Ensure minimum of 7 (a touchdown) - Poisson needs positive lambda
        return max(7.0, expected)

    def apply_intangibles_adjustment(
        self,
        home_team: str,
        away_team: str,
        home_lambda: float,
        away_lambda: float,
        game_data: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Apply intangibles adjustments to expected scores.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_lambda: Home team's expected score (pre-adjustment)
            away_lambda: Away team's expected score (pre-adjustment)
            game_data: Optional dict with game context (date, rest, flags, etc.)

        Returns:
            Tuple of (adjusted_home_lambda, adjusted_away_lambda)
        """
        if not self.intangibles_calculator or not self.intangibles_config:
            return home_lambda, away_lambda

        game_data = game_data or {}
        from datetime import date

        # Calculate total intangibles adjustment
        adjustments = self.intangibles_calculator.calculate_total_intangibles(
            home_team=home_team,
            away_team=away_team,
            game_date=game_data.get('date', date.today()),
            bye_teams=game_data.get('bye_teams', set()),
            is_thursday_night=game_data.get('is_thursday_night', False),
            is_monday_night=game_data.get('is_monday_night', False),
            is_early_et_game=game_data.get('is_early_et_game', False),
            is_division=game_data.get('is_division', False),
            home_spread=game_data.get('home_spread'),
            weather_data=game_data.get('weather_data'),
            home_rest=game_data.get('home_rest'),
            away_rest=game_data.get('away_rest')
        )

        # Apply point adjustment (positive = home advantage)
        total_adj = adjustments['total_adjustment']

        # Apply adjustment to home and away scores
        # If adjustment is positive, home gains, away loses (relative)
        # We split the adjustment and add/subtract from each team
        home_lambda += total_adj * 0.5
        away_lambda -= total_adj * 0.5

        # Apply weather scaling (reduces both teams' scoring)
        weather_scale = adjustments['weather_scaling']
        if weather_scale < 1.0:
            home_lambda *= weather_scale
            away_lambda *= weather_scale

        return max(7.0, home_lambda), max(7.0, away_lambda)
    
    def simulate_game(self, home_team: str, away_team: str, game_data: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Simulate a single game using Poisson scoring model.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_data: Optional dict with game context for intangibles

        Returns:
            Tuple of (home_score, away_score)
        """
        # Calculate expected scores
        home_lambda = self.calculate_expected_score(home_team, away_team, is_home=True)
        away_lambda = self.calculate_expected_score(away_team, home_team, is_home=False)

        # Apply intangibles adjustments
        if self.intangibles_calculator:
            home_lambda, away_lambda = self.apply_intangibles_adjustment(
                home_team, away_team, home_lambda, away_lambda, game_data
            )

        # Sample from Poisson distributions
        home_score = poisson.rvs(home_lambda)
        away_score = poisson.rvs(away_lambda)

        # Handle ties (rare in NFL, ~1% of games)
        # Simulate OT with 50/50 coinflip if tied
        if home_score == away_score:
            # In NFL, ~57% of OT games are won by receiving team
            # But for simplicity, just give slight home advantage
            if np.random.random() < 0.52:
                home_score += 3  # Home team wins with FG
            else:
                away_score += 3

        return int(home_score), int(away_score)
    
    def get_win_probability(self, home_team: str, away_team: str, n_sims: int = 1000) -> dict:
        """
        Calculate win probability for a matchup via simulation.
        
        Returns:
            Dict with 'home_win', 'away_win', 'home_expected', 'away_expected'
        """
        home_lambda = self.calculate_expected_score(home_team, away_team, is_home=True)
        away_lambda = self.calculate_expected_score(away_team, home_team, is_home=False)
        
        home_scores = poisson.rvs(home_lambda, size=n_sims)
        away_scores = poisson.rvs(away_lambda, size=n_sims)
        
        home_wins = np.sum(home_scores > away_scores)
        ties = np.sum(home_scores == away_scores)
        
        # Split ties evenly
        home_win_pct = (home_wins + ties * 0.5) / n_sims
        
        return {
            'home_win': home_win_pct,
            'away_win': 1 - home_win_pct,
            'home_expected': home_lambda,
            'away_expected': away_lambda
        }


def build_season_data_from_standings(
    standings: List[Dict],
    completed_games: List[Game],
    remaining_games: List[Game]
) -> SeasonData:
    """
    Build SeasonData object from scraped standings and games.
    
    Args:
        standings: List of team dicts with name, w, l, t, div, conf, pf, pa
        completed_games: List of completed Game objects
        remaining_games: List of remaining Game objects
    
    Returns:
        SeasonData object ready for simulation
    """
    teams: Dict[str, TeamStats] = {}
    
    # Create TeamStats for each team from standings
    for team_data in standings:
        name = team_data['name']
        games_played = team_data['w'] + team_data['l'] + team_data.get('t', 0)
        pf = team_data.get('pf', 0)
        pa = team_data.get('pa', 0)
        
        # Calculate average points if we have data
        avg_pf = pf / games_played if games_played > 0 and pf > 0 else 22.0
        avg_pa = pa / games_played if games_played > 0 and pa > 0 else 22.0
        
        teams[name] = TeamStats(
            name=name,
            wins=team_data['w'],
            losses=team_data['l'],
            ties=team_data.get('t', 0),
            points_for=pf,
            points_against=pa,
            division=TEAM_TO_DIVISION.get(name, team_data.get('div', 'Unknown')),
            conference=TEAM_TO_CONFERENCE.get(name, team_data.get('conf', 'Unknown')),
            div_wins=0, div_losses=0, div_ties=0,
            conf_wins=0, conf_losses=0, conf_ties=0,
            avg_points_for=avg_pf,
            avg_points_against=avg_pa
        )
    
    # Calculate division and conference records from completed games
    for game in completed_games:
        if not game.completed:
            continue
        
        home_team = teams.get(game.home_team)
        away_team = teams.get(game.away_team)
        
        if not home_team or not away_team:
            continue
        
        winner = game.winner
        loser = game.loser
        is_tie = game.is_tie
        
        # Update points (if not already in standings)
        if home_team.points_for == 0:
            home_team.points_for += game.home_score or 0
            home_team.points_against += game.away_score or 0
        if away_team.points_for == 0:
            away_team.points_for += game.away_score or 0
            away_team.points_against += game.home_score or 0
        
        # Division game
        if home_team.division == away_team.division and home_team.conference == away_team.conference:
            if is_tie:
                home_team.div_ties += 1
                away_team.div_ties += 1
            else:
                if winner == game.home_team:
                    home_team.div_wins += 1
                    away_team.div_losses += 1
                else:
                    away_team.div_wins += 1
                    home_team.div_losses += 1
        
        # Conference game
        if home_team.conference == away_team.conference:
            if is_tie:
                home_team.conf_ties += 1
                away_team.conf_ties += 1
            else:
                if winner == game.home_team:
                    home_team.conf_wins += 1
                    away_team.conf_losses += 1
                else:
                    away_team.conf_wins += 1
                    home_team.conf_losses += 1
    
    return SeasonData(
        teams=teams,
        completed_games=completed_games,
        remaining_games=remaining_games
    )


def run_advanced_simulation(
    standings: List[Dict],
    completed_games: List[Game],
    remaining_games: List[Game],
    n_simulations: int = 10000,
    show_progress: bool = True,
    use_epa: bool = True,
    season: int = 2025,
    injury_impacts: Optional[Dict[str, Dict[str, float]]] = None,
    use_momentum: bool = True,
    use_intangibles: bool = False,
    intangibles_config: Optional[IntangiblesConfig] = None
) -> Dict[str, Dict]:
    """
    Run Monte Carlo simulation with real NFL tiebreakers.

    Args:
        standings: Current standings data
        completed_games: Completed games this season
        remaining_games: Remaining games to simulate
        n_simulations: Number of simulations to run
        show_progress: Whether to show progress bar
        use_epa: Whether to use EPA-based Poisson model (if available)
        season: NFL season year (for loading correct EPA data)
        injury_impacts: Dict of team -> injury impact from player_impact.py
        use_momentum: Whether to apply momentum adjustments (default: True)
        use_intangibles: Whether to apply intangibles adjustments (default: False)
        intangibles_config: Configuration for intangibles adjustments

    Returns:
        Dictionary with simulation results for each team
    """
    # Try to load EPA data if requested
    epa_df = None
    if use_epa and EPA_AVAILABLE:
        try:
            epa_df = load_team_epa(season=season)
            print("📊 Using EPA-based Poisson scoring model")
        except Exception as e:
            print(f"⚠️  EPA data unavailable ({e}), using traditional model")
            epa_df = None
    elif use_epa:
        print("⚠️  EPA loader not available, using traditional model")
    
    print(f"\n🎲 Running {n_simulations:,} advanced simulations with NFL tiebreakers...")
    
    # Build initial season data
    base_season = build_season_data_from_standings(standings, completed_games, remaining_games)
    
    # Determine if this is current season or historical (backtest)
    from datetime import datetime
    current_year = datetime.now().year
    is_current_season = (season >= current_year)
    
    # Calculate game-based momentum from completed games
    # For current season: use game-based momentum (most accurate for live predictions)
    # For historical: use EPA-based momentum (from that year's play-by-play)
    game_momentum = None
    prefer_game_momentum = is_current_season  # Prefer game-based for current season
    
    if use_momentum and is_current_season:
        game_momentum = calculate_game_momentum(completed_games, n_recent=4)
        if game_momentum:
            print(f"📈 Calculated momentum from {len(game_momentum)} teams' recent games")

    # Setup intangibles calculator if requested
    intangibles_calculator = None
    if use_intangibles and INTANGIBLES_AVAILABLE:
        intangibles_config = intangibles_config or IntangiblesConfig()
        intangibles_calculator = IntangiblesCalculator(intangibles_config)

        # Set turnover margins from standings (if available)
        turnover_margin = {}
        for team_data in standings:
            name = team_data['name']
            # Use point differential as proxy for turnover margin
            pf = team_data.get('pf', 0)
            pa = team_data.get('pa', 0)
            # Rough estimate: 1 turnover = ~4 points, so convert point diff to turnover estimate
            diff = (pf - pa) / team_data.get('w', 1) if team_data.get('w', 0) > 0 else 0
            turnover_margin[name] = diff / 4  # Rough estimate

        intangibles_calculator.set_team_turnover_margin(turnover_margin)

        print(f"📊 Intangibles enabled: rest days, turnover luck, travel, division")

    # Initialize results tracking
    results: Dict[str, Dict] = {}
    for team_name in base_season.teams.keys():
        results[team_name] = {
            'playoff_count': 0,
            'division_winner_count': 0,
            'wildcard_count': 0,
            'seed_counts': {i: 0 for i in range(1, 8)},
            'total_wins': 0,
            'total_sims': n_simulations
        }
    
    # Create simulator - use EPA if available, with game-based momentum for current season
    if epa_df is not None:
        simulator = EPAGameSimulator(
            epa_df=epa_df,
            season_data=base_season,
            injury_impacts=injury_impacts,
            use_momentum=use_momentum,
            game_momentum=game_momentum,
            prefer_game_momentum=prefer_game_momentum,
            intangibles_config=intangibles_config,
            intangibles_calculator=intangibles_calculator
        )
        if simulator.has_momentum and use_momentum:
            if is_current_season and game_momentum:
                print(f"📈 Using game-based momentum (from {season} score margins)")
            elif simulator.has_epa_momentum:
                print(f"📈 Using EPA-based momentum (from {season} play-by-play)")
    else:
        simulator = GameSimulator(base_season)
    
    tiebreaker = NFLTiebreaker(base_season)
    
    # Progress bar
    iterator = tqdm(range(n_simulations), desc="Simulating seasons", disable=not show_progress)
    
    for _ in iterator:
        # Deep copy the season data for this simulation
        sim_teams = {}
        for name, team in base_season.teams.items():
            sim_teams[name] = TeamStats(
                name=team.name,
                wins=team.wins,
                losses=team.losses,
                ties=team.ties,
                points_for=team.points_for,
                points_against=team.points_against,
                division=team.division,
                conference=team.conference,
                div_wins=team.div_wins,
                div_losses=team.div_losses,
                div_ties=team.div_ties,
                conf_wins=team.conf_wins,
                conf_losses=team.conf_losses,
                conf_ties=team.conf_ties,
                avg_points_for=team.avg_points_for,
                avg_points_against=team.avg_points_against
            )
        
        sim_completed = list(completed_games)
        
        # Simulate remaining games
        for game in remaining_games:
            home_team = sim_teams.get(game.home_team)
            away_team = sim_teams.get(game.away_team)

            if not home_team or not away_team:
                continue

            # Build game data dict for intangibles
            from datetime import datetime
            game_data = {}
            if game.gameday:
                try:
                    game_data['date'] = datetime.strptime(game.gameday, '%Y-%m-%d').date()
                except:
                    game_data['date'] = datetime.now().date()
            else:
                game_data['date'] = datetime.now().date()

            # Rest days - use actual rest from Game if available
            if game.home_rest is not None and game.away_rest is not None:
                game_data['home_rest'] = game.home_rest
                game_data['away_rest'] = game.away_rest

            # TNF/MNF flags
            game_data['is_thursday_night'] = game.is_thursday_night
            game_data['is_monday_night'] = game.is_monday_night

            # Division game
            game_data['is_division'] = game.is_division

            # Weather
            if game.temp is not None or game.wind is not None:
                game_data['weather_data'] = {
                    'temp': game.temp or 70,
                    'wind': game.wind or 0
                }

            # Early ET game (1pm or 4:05pm ET games)
            if game.gametime:
                is_early = game.gametime.startswith('13:') or game.gametime.startswith('16:05')
                game_data['is_early_et_game'] = is_early

            # Simulate the game with intangibles context
            home_score, away_score = simulator.simulate_game(
                game.home_team,
                game.away_team,
                game_data=game_data
            )

            sim_game = Game(
                week=game.week,
                home_team=game.home_team,
                away_team=game.away_team,
                home_score=home_score,
                away_score=away_score,
                completed=True
            )
            sim_completed.append(sim_game)
            
            # Update stats
            if home_score > away_score:
                home_team.wins += 1
                away_team.losses += 1
                if home_team.division == away_team.division:
                    home_team.div_wins += 1
                    away_team.div_losses += 1
                if home_team.conference == away_team.conference:
                    home_team.conf_wins += 1
                    away_team.conf_losses += 1
            elif away_score > home_score:
                away_team.wins += 1
                home_team.losses += 1
                if home_team.division == away_team.division:
                    away_team.div_wins += 1
                    home_team.div_losses += 1
                if home_team.conference == away_team.conference:
                    away_team.conf_wins += 1
                    home_team.conf_losses += 1
            else:
                # Tie
                home_team.ties += 1
                away_team.ties += 1
                if home_team.division == away_team.division:
                    home_team.div_ties += 1
                    away_team.div_ties += 1
                if home_team.conference == away_team.conference:
                    home_team.conf_ties += 1
                    away_team.conf_ties += 1
            
            # Update points
            home_team.points_for += home_score
            home_team.points_against += away_score
            away_team.points_for += away_score
            away_team.points_against += home_score
        
        # Create simulated season data for tiebreaker
        sim_season = SeasonData(
            teams=sim_teams,
            completed_games=sim_completed,
            remaining_games=[]
        )
        
        # Track wins for ALL teams (not just playoff teams)
        for team_name, team in sim_teams.items():
            results[team_name]['total_wins'] += team.wins
        
        # Update tiebreaker with simulated data
        sim_tiebreaker = NFLTiebreaker(sim_season)
        
        # Determine playoffs for each conference
        for conf in ['AFC', 'NFC']:
            try:
                seeding = sim_tiebreaker.determine_playoff_seeding(conf)
                
                for seed, team_name in enumerate(seeding, 1):
                    results[team_name]['playoff_count'] += 1
                    results[team_name]['seed_counts'][seed] += 1
                    
                    if seed <= 4:
                        results[team_name]['division_winner_count'] += 1
                    else:
                        results[team_name]['wildcard_count'] += 1
            except Exception as e:
                # Fallback to simple win-based seeding
                conf_teams = [(name, t) for name, t in sim_teams.items() if t.conference == conf]
                conf_teams.sort(key=lambda x: x[1].win_pct, reverse=True)
                
                for seed, (team_name, _) in enumerate(conf_teams[:7], 1):
                    results[team_name]['playoff_count'] += 1
                    results[team_name]['seed_counts'][seed] += 1
    
    # Calculate averages
    for team_name in results:
        if results[team_name]['playoff_count'] > 0:
            results[team_name]['avg_wins'] = results[team_name]['total_wins'] / n_simulations
        else:
            results[team_name]['avg_wins'] = base_season.teams[team_name].wins
    
    return results


def print_simulation_results(results: Dict[str, Dict], teams_data: List[Dict], n_simulations: int = 10000):
    """Print formatted simulation results"""
    
    # Separate by conference
    afc_results = {}
    nfc_results = {}
    
    for team_data in teams_data:
        name = team_data['name']
        if name in results:
            if team_data['conf'] == 'AFC':
                afc_results[name] = results[name]
            else:
                nfc_results[name] = results[name]
    
    print("\n" + "=" * 50)
    print("    NFL PLAYOFF PROBABILITIES (Advanced Model)")
    print("=" * 50)
    
    for conf, conf_results in [('AFC', afc_results), ('NFC', nfc_results)]:
        print(f"\n🏈 {conf} PLAYOFF PICTURE")
        print("-" * 50)
        
        # Sort by playoff probability
        sorted_teams = sorted(
            conf_results.items(),
            key=lambda x: x[1]['playoff_count'],
            reverse=True
        )
        
        # Find division leaders
        div_leaders = {}
        for div in ['East', 'North', 'South', 'West']:
            div_teams = [
                (name, r) for name, r in conf_results.items()
                if TEAM_TO_DIVISION.get(name) == div
            ]
            if div_teams:
                leader = max(div_teams, key=lambda x: x[1]['division_winner_count'])
                div_leaders[div] = leader[0]
        
        print("\nDivision Leaders:")
        for div in ['East', 'North', 'South', 'West']:
            if div in div_leaders:
                name = div_leaders[div]
                r = conf_results[name]
                div_win_pct = (r['division_winner_count'] / n_simulations) * 100
                playoff_pct = (r['playoff_count'] / n_simulations) * 100
                avg_wins = r.get('avg_wins', 0)
                print(f"  {div}: {name:<22} Div: {div_win_pct:>5.1f}%  Playoff: {playoff_pct:>5.1f}%  Wins: {avg_wins:.1f}")
        
        print("\nWild Card Race:")
        wc_candidates = [
            (name, r) for name, r in sorted_teams
            if name not in div_leaders.values() and r['wildcard_count'] > 0
        ]
        wc_sorted = sorted(wc_candidates, key=lambda x: x[1]['wildcard_count'], reverse=True)
        
        for i, (name, r) in enumerate(wc_sorted[:3], 1):
            wc_pct = (r['wildcard_count'] / n_simulations) * 100
            playoff_pct = (r['playoff_count'] / n_simulations) * 100
            avg_wins = r.get('avg_wins', 0)
            print(f"  {i}. {name:<24} WC: {wc_pct:>5.1f}%  Playoff: {playoff_pct:>5.1f}%  Wins: {avg_wins:.1f}")
        
        # Outside Looking In - teams just outside playoff contention
        print("\nOutside Looking In:")
        # Get teams not in division leaders with lower playoff odds
        outside_candidates = [
            (name, r) for name, r in sorted_teams
            if name not in div_leaders.values()
        ]
        # Skip the top 3 wild card contenders, take next 3
        outside_looking_in = outside_candidates[3:6]
        
        for i, (name, r) in enumerate(outside_looking_in, 1):
            wc_pct = (r['wildcard_count'] / n_simulations) * 100
            playoff_pct = (r['playoff_count'] / n_simulations) * 100
            avg_wins = r.get('avg_wins', 0)
            print(f"  {i}. {name:<24} WC: {wc_pct:>5.1f}%  Playoff: {playoff_pct:>5.1f}%  Wins: {avg_wins:.1f}")
    
    # Seed distribution
    print("\n" + "=" * 50)
    print("       SEED PROBABILITY DISTRIBUTION")
    print("=" * 50)
    
    for conf, conf_results in [('AFC', afc_results), ('NFC', nfc_results)]:
        print(f"\n{conf}:")
        print(f"{'Team':<25} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6} {'5th':>6} {'6th':>6} {'7th':>6}")
        print("-" * 70)
        
        sorted_teams = sorted(
            conf_results.items(),
            key=lambda x: x[1]['seed_counts'][1],  # Sort by 1st seed probability
            reverse=True
        )[:10]
        
        for name, r in sorted_teams:
            seed_pcts = []
            for seed in range(1, 8):
                pct = (r['seed_counts'][seed] / n_simulations) * 100
                seed_pcts.append(f"{pct:>5.1f}%")
            print(f"{name:<25} {' '.join(seed_pcts)}")
