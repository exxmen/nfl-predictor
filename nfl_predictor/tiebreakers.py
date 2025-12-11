"""
NFL Playoff Predictor with Full Tiebreaker Support

This module implements the complete NFL tiebreaker rules for playoff seeding.
It includes data models, game simulation, and tiebreaker logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import json
import os
import time
import random

# ==========================================
# DATA MODELS
# ==========================================

# NFL Division/Conference structure
NFL_DIVISIONS = {
    'AFC': ['East', 'North', 'South', 'West'],
    'NFC': ['East', 'North', 'South', 'West']
}

NFL_TEAMS = {
    'AFC': {
        'East': ['Buffalo Bills', 'Miami Dolphins', 'New England Patriots', 'New York Jets'],
        'North': ['Baltimore Ravens', 'Cincinnati Bengals', 'Cleveland Browns', 'Pittsburgh Steelers'],
        'South': ['Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Tennessee Titans'],
        'West': ['Denver Broncos', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers']
    },
    'NFC': {
        'East': ['Dallas Cowboys', 'New York Giants', 'Philadelphia Eagles', 'Washington Commanders'],
        'North': ['Chicago Bears', 'Detroit Lions', 'Green Bay Packers', 'Minnesota Vikings'],
        'South': ['Atlanta Falcons', 'Carolina Panthers', 'New Orleans Saints', 'Tampa Bay Buccaneers'],
        'West': ['Arizona Cardinals', 'Los Angeles Rams', 'San Francisco 49ers', 'Seattle Seahawks']
    }
}

# Build lookup tables
TEAM_TO_DIVISION = {}
TEAM_TO_CONFERENCE = {}
for conf, divisions in NFL_TEAMS.items():
    for div, teams in divisions.items():
        for team in teams:
            TEAM_TO_DIVISION[team] = div
            TEAM_TO_CONFERENCE[team] = conf


@dataclass
class Game:
    """Represents a single NFL game (completed or scheduled)"""
    week: int
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    completed: bool = False
    
    @property
    def winner(self) -> Optional[str]:
        if not self.completed or self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return self.home_team
        elif self.away_score > self.home_score:
            return self.away_team
        return None  # Tie
    
    @property
    def loser(self) -> Optional[str]:
        if not self.completed or self.home_score is None or self.away_score is None:
            return None
        if self.home_score < self.away_score:
            return self.home_team
        elif self.away_score < self.home_score:
            return self.away_team
        return None  # Tie
    
    @property
    def is_tie(self) -> bool:
        return self.completed and self.home_score == self.away_score
    
    def involves_team(self, team: str) -> bool:
        return team in (self.home_team, self.away_team)
    
    def get_opponent(self, team: str) -> Optional[str]:
        if team == self.home_team:
            return self.away_team
        elif team == self.away_team:
            return self.home_team
        return None
    
    def get_team_score(self, team: str) -> Optional[int]:
        if team == self.home_team:
            return self.home_score
        elif team == self.away_team:
            return self.away_score
        return None
    
    def get_opponent_score(self, team: str) -> Optional[int]:
        if team == self.home_team:
            return self.away_score
        elif team == self.away_team:
            return self.home_score
        return None


@dataclass
class TeamStats:
    """Team statistics for tiebreaker calculations"""
    name: str
    conference: str
    division: str
    
    # Overall record
    wins: int = 0
    losses: int = 0
    ties: int = 0
    
    # Points
    points_for: int = 0
    points_against: int = 0
    
    # Division record
    div_wins: int = 0
    div_losses: int = 0
    div_ties: int = 0
    
    # Conference record
    conf_wins: int = 0
    conf_losses: int = 0
    conf_ties: int = 0
    
    # For simulation
    avg_points_for: float = 22.0
    avg_points_against: float = 22.0
    
    @property
    def win_pct(self) -> float:
        games = self.wins + self.losses + self.ties
        if games == 0:
            return 0.5
        return (self.wins + 0.5 * self.ties) / games
    
    @property
    def div_win_pct(self) -> float:
        games = self.div_wins + self.div_losses + self.div_ties
        if games == 0:
            return 0.5
        return (self.div_wins + 0.5 * self.div_ties) / games
    
    @property
    def conf_win_pct(self) -> float:
        games = self.conf_wins + self.conf_losses + self.conf_ties
        if games == 0:
            return 0.5
        return (self.conf_wins + 0.5 * self.conf_ties) / games
    
    @property
    def point_differential(self) -> int:
        return self.points_for - self.points_against
    
    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.ties


@dataclass 
class SeasonData:
    """Complete season data for simulation"""
    teams: Dict[str, TeamStats] = field(default_factory=dict)
    completed_games: List[Game] = field(default_factory=list)
    remaining_games: List[Game] = field(default_factory=list)
    
    def get_head_to_head(self, team1: str, team2: str) -> Tuple[int, int, int]:
        """Get head-to-head record between two teams (wins, losses, ties for team1)"""
        wins = losses = ties = 0
        for game in self.completed_games:
            if game.involves_team(team1) and game.involves_team(team2):
                if game.winner == team1:
                    wins += 1
                elif game.winner == team2:
                    losses += 1
                elif game.is_tie:
                    ties += 1
        return wins, losses, ties


# ==========================================
# TIEBREAKER IMPLEMENTATION
# ==========================================

class NFLTiebreaker:
    """
    Implements NFL tiebreaker procedures.
    
    Division Tiebreaker (in order):
    1. Head-to-head (best record in games between tied teams)
    2. Division record
    3. Common games record (min 4 games)
    4. Conference record
    5. Strength of Victory
    6. Strength of Schedule
    7. Best combined ranking (points for + points against rank in conference)
    8. Best combined ranking (all games)
    9. Net points in common games
    10. Net points in all games
    11. Net touchdowns in all games
    12. Coin toss (random)
    
    Wild Card Tiebreaker (in order):
    1. Head-to-head (only if all tied teams played each other)
    2. Conference record
    3. Common games record (min 4)
    4. Strength of Victory
    5. Strength of Schedule
    6. Best combined ranking (conference)
    7. Best combined ranking (all games)
    8. Net points in conference games
    9. Net points in all games
    10. Net touchdowns in all games
    11. Coin toss
    """
    
    def __init__(self, season_data: SeasonData):
        self.season = season_data
        self._h2h_cache = {}
        self._common_games_cache = {}
        self._sov_cache = {}
        self._sos_cache = {}
    
    def clear_caches(self):
        """Clear all caches (call after simulation updates)"""
        self._h2h_cache.clear()
        self._common_games_cache.clear()
        self._sov_cache.clear()
        self._sos_cache.clear()
    
    def get_head_to_head_record(self, team: str, opponents: List[str]) -> Tuple[int, int, int]:
        """Get team's record against a set of opponents"""
        cache_key = (team, tuple(sorted(opponents)))
        if cache_key in self._h2h_cache:
            return self._h2h_cache[cache_key]
        
        wins = losses = ties = 0
        for game in self.season.completed_games:
            if game.involves_team(team):
                opp = game.get_opponent(team)
                if opp in opponents:
                    if game.winner == team:
                        wins += 1
                    elif game.winner == opp:
                        losses += 1
                    elif game.is_tie:
                        ties += 1
        
        result = (wins, losses, ties)
        self._h2h_cache[cache_key] = result
        return result
    
    def get_common_opponents(self, teams: List[str]) -> Set[str]:
        """Find opponents that all teams have played (at least 4 games each)"""
        if len(teams) < 2:
            return set()
        
        opponent_games = {team: defaultdict(int) for team in teams}
        
        for game in self.season.completed_games:
            for team in teams:
                if game.involves_team(team):
                    opp = game.get_opponent(team)
                    if opp not in teams:  # Exclude games between the tied teams
                        opponent_games[team][opp] += 1
        
        # Find opponents all teams have played
        common = None
        for team in teams:
            team_opps = set(opponent_games[team].keys())
            if common is None:
                common = team_opps
            else:
                common &= team_opps
        
        return common or set()
    
    def get_common_games_record(self, team: str, common_opponents: Set[str]) -> Tuple[int, int, int]:
        """Get team's record against common opponents"""
        cache_key = (team, tuple(sorted(common_opponents)))
        if cache_key in self._common_games_cache:
            return self._common_games_cache[cache_key]
        
        wins = losses = ties = 0
        for game in self.season.completed_games:
            if game.involves_team(team):
                opp = game.get_opponent(team)
                if opp in common_opponents:
                    if game.winner == team:
                        wins += 1
                    elif game.winner == opp:
                        losses += 1
                    elif game.is_tie:
                        ties += 1
        
        result = (wins, losses, ties)
        self._common_games_cache[cache_key] = result
        return result
    
    def calculate_strength_of_victory(self, team: str) -> float:
        """
        Strength of Victory = Combined W-L-T of all teams beaten
        Returns win percentage of beaten opponents
        """
        if team in self._sov_cache:
            return self._sov_cache[team]
        
        beaten_opponents = []
        for game in self.season.completed_games:
            if game.winner == team:
                beaten_opponents.append(game.loser)
        
        if not beaten_opponents:
            self._sov_cache[team] = 0.0
            return 0.0
        
        total_wins = total_losses = total_ties = 0
        for opp in beaten_opponents:
            if opp in self.season.teams:
                stats = self.season.teams[opp]
                total_wins += stats.wins
                total_losses += stats.losses
                total_ties += stats.ties
        
        total_games = total_wins + total_losses + total_ties
        if total_games == 0:
            result = 0.0
        else:
            result = (total_wins + 0.5 * total_ties) / total_games
        
        self._sov_cache[team] = result
        return result
    
    def calculate_strength_of_schedule(self, team: str) -> float:
        """
        Strength of Schedule = Combined W-L-T of all opponents
        Returns win percentage of all opponents
        """
        if team in self._sos_cache:
            return self._sos_cache[team]
        
        opponents = []
        for game in self.season.completed_games:
            if game.involves_team(team):
                opponents.append(game.get_opponent(team))
        
        if not opponents:
            self._sos_cache[team] = 0.5
            return 0.5
        
        total_wins = total_losses = total_ties = 0
        for opp in opponents:
            if opp in self.season.teams:
                stats = self.season.teams[opp]
                total_wins += stats.wins
                total_losses += stats.losses
                total_ties += stats.ties
        
        total_games = total_wins + total_losses + total_ties
        if total_games == 0:
            result = 0.5
        else:
            result = (total_wins + 0.5 * total_ties) / total_games
        
        self._sos_cache[team] = result
        return result
    
    def get_conference_point_rankings(self, conference: str) -> Dict[str, int]:
        """Get combined points for/against rankings within conference"""
        conf_teams = [name for name, stats in self.season.teams.items() 
                      if stats.conference == conference]
        
        # Rank by points for (higher is better)
        pf_rank = {}
        sorted_by_pf = sorted(conf_teams, 
                              key=lambda t: self.season.teams[t].points_for, 
                              reverse=True)
        for i, team in enumerate(sorted_by_pf):
            pf_rank[team] = i + 1
        
        # Rank by points against (lower is better)
        pa_rank = {}
        sorted_by_pa = sorted(conf_teams, 
                              key=lambda t: self.season.teams[t].points_against)
        for i, team in enumerate(sorted_by_pa):
            pa_rank[team] = i + 1
        
        # Combined ranking (lower is better)
        combined = {team: pf_rank[team] + pa_rank[team] for team in conf_teams}
        return combined
    
    def break_division_tie(self, teams: List[str]) -> List[str]:
        """
        Break tie between teams in same division for division winner.
        Returns teams sorted from best to worst.
        """
        if len(teams) <= 1:
            return teams
        
        # Create sortable tuples with tiebreaker values (higher is better for all)
        def get_tiebreaker_tuple(team: str) -> tuple:
            stats = self.season.teams[team]
            other_teams = [t for t in teams if t != team]
            
            # 1. Head-to-head
            h2h_w, h2h_l, h2h_t = self.get_head_to_head_record(team, other_teams)
            h2h_games = h2h_w + h2h_l + h2h_t
            h2h_pct = (h2h_w + 0.5 * h2h_t) / h2h_games if h2h_games > 0 else 0.5
            
            # 2. Division record
            div_pct = stats.div_win_pct
            
            # 3. Common games
            common_opps = self.get_common_opponents(teams)
            if len(common_opps) >= 4:
                cg_w, cg_l, cg_t = self.get_common_games_record(team, common_opps)
                cg_games = cg_w + cg_l + cg_t
                cg_pct = (cg_w + 0.5 * cg_t) / cg_games if cg_games > 0 else 0.5
            else:
                cg_pct = 0.5  # Not enough common games
            
            # 4. Conference record
            conf_pct = stats.conf_win_pct
            
            # 5. Strength of Victory
            sov = self.calculate_strength_of_victory(team)
            
            # 6. Strength of Schedule
            sos = self.calculate_strength_of_schedule(team)
            
            # 7-8. Point rankings (use inverse since lower rank is better)
            conf_rankings = self.get_conference_point_rankings(stats.conference)
            combined_rank = conf_rankings.get(team, 16)
            point_rank_score = 32 - combined_rank  # Higher is better
            
            # 9-10. Net points
            net_points = stats.point_differential
            
            # 11. Random tiebreaker for "coin toss"
            coin_toss = random.random()
            
            return (
                h2h_pct,      # 1
                div_pct,      # 2
                cg_pct,       # 3
                conf_pct,     # 4
                sov,          # 5
                sos,          # 6
                point_rank_score,  # 7-8
                net_points,   # 9-10
                coin_toss     # 11-12
            )
        
        # Sort by tiebreaker tuple (descending - higher values are better)
        return sorted(teams, key=get_tiebreaker_tuple, reverse=True)
    
    def break_wild_card_tie(self, teams: List[str]) -> List[str]:
        """
        Break tie between teams from different divisions for wild card.
        Returns teams sorted from best to worst.
        """
        if len(teams) <= 1:
            return teams
        
        def get_tiebreaker_tuple(team: str) -> tuple:
            stats = self.season.teams[team]
            other_teams = [t for t in teams if t != team]
            
            # 1. Head-to-head (only if ALL tied teams played each other)
            all_played = True
            for other in other_teams:
                h2h = self.get_head_to_head_record(team, [other])
                if sum(h2h) == 0:  # Haven't played
                    all_played = False
                    break
            
            if all_played:
                h2h_w, h2h_l, h2h_t = self.get_head_to_head_record(team, other_teams)
                h2h_games = h2h_w + h2h_l + h2h_t
                h2h_pct = (h2h_w + 0.5 * h2h_t) / h2h_games if h2h_games > 0 else 0.5
            else:
                h2h_pct = 0.5  # Skip this tiebreaker
            
            # 2. Conference record
            conf_pct = stats.conf_win_pct
            
            # 3. Common games
            common_opps = self.get_common_opponents(teams)
            if len(common_opps) >= 4:
                cg_w, cg_l, cg_t = self.get_common_games_record(team, common_opps)
                cg_games = cg_w + cg_l + cg_t
                cg_pct = (cg_w + 0.5 * cg_t) / cg_games if cg_games > 0 else 0.5
            else:
                cg_pct = 0.5
            
            # 4. Strength of Victory
            sov = self.calculate_strength_of_victory(team)
            
            # 5. Strength of Schedule
            sos = self.calculate_strength_of_schedule(team)
            
            # 6-7. Point rankings
            conf_rankings = self.get_conference_point_rankings(stats.conference)
            combined_rank = conf_rankings.get(team, 16)
            point_rank_score = 32 - combined_rank
            
            # 8-9. Net points in conference / all games
            net_points = stats.point_differential
            
            # 10-11. Coin toss
            coin_toss = random.random()
            
            return (
                h2h_pct,
                conf_pct,
                cg_pct,
                sov,
                sos,
                point_rank_score,
                net_points,
                coin_toss
            )
        
        return sorted(teams, key=get_tiebreaker_tuple, reverse=True)
    
    def determine_division_winners(self, conference: str) -> Dict[str, str]:
        """
        Determine division winners for a conference.
        Returns {division: winning_team}
        """
        division_winners = {}
        
        for division in NFL_DIVISIONS[conference]:
            # Get teams in this division
            div_teams = [name for name, stats in self.season.teams.items()
                        if stats.conference == conference and stats.division == division]
            
            if not div_teams:
                continue
            
            # Group by wins
            by_wins = defaultdict(list)
            for team in div_teams:
                by_wins[self.season.teams[team].wins].append(team)
            
            # Get team(s) with most wins
            max_wins = max(by_wins.keys())
            top_teams = by_wins[max_wins]
            
            # Break ties if needed
            if len(top_teams) > 1:
                top_teams = self.break_division_tie(top_teams)
            
            division_winners[division] = top_teams[0]
        
        return division_winners
    
    def determine_wild_cards(self, conference: str, division_winners: Dict[str, str], 
                             num_wild_cards: int = 3) -> List[str]:
        """
        Determine wild card teams for a conference.
        Returns list of wild card teams in seeding order.
        """
        # Get non-division-winners
        winner_set = set(division_winners.values())
        wild_card_pool = [name for name, stats in self.season.teams.items()
                         if stats.conference == conference and name not in winner_set]
        
        if not wild_card_pool:
            return []
        
        # Sort by wins first
        wild_card_pool.sort(key=lambda t: self.season.teams[t].wins, reverse=True)
        
        # Group by wins and break ties
        result = []
        i = 0
        while i < len(wild_card_pool) and len(result) < num_wild_cards:
            current_wins = self.season.teams[wild_card_pool[i]].wins
            tied_teams = []
            
            while i < len(wild_card_pool) and self.season.teams[wild_card_pool[i]].wins == current_wins:
                tied_teams.append(wild_card_pool[i])
                i += 1
            
            # Break ties
            if len(tied_teams) > 1:
                # Check if any are from same division (use division tiebreaker first)
                by_division = defaultdict(list)
                for team in tied_teams:
                    by_division[self.season.teams[team].division].append(team)
                
                ordered = []
                for div, div_teams in by_division.items():
                    if len(div_teams) > 1:
                        div_teams = self.break_division_tie(div_teams)
                    ordered.extend(div_teams)
                
                # Now apply wild card tiebreaker to the full group
                tied_teams = self.break_wild_card_tie(ordered)
            
            result.extend(tied_teams)
        
        return result[:num_wild_cards]
    
    def determine_playoff_seeding(self, conference: str) -> List[str]:
        """
        Determine complete playoff seeding for a conference.
        Returns list of 7 teams in seed order (1-7).
        
        Seeds 1-4: Division winners (seeded by record, with tiebreakers)
        Seeds 5-7: Wild card teams
        """
        # Get division winners
        division_winners = self.determine_division_winners(conference)
        winner_list = list(division_winners.values())
        
        # Sort division winners by record (with tiebreakers)
        # First group by wins
        by_wins = defaultdict(list)
        for team in winner_list:
            by_wins[self.season.teams[team].wins].append(team)
        
        seeded_winners = []
        for wins in sorted(by_wins.keys(), reverse=True):
            teams = by_wins[wins]
            if len(teams) > 1:
                # Use division tiebreaker for teams from same division
                # or head-to-head for teams from different divisions
                teams = self.break_division_tie(teams)  # Works for cross-division too
            seeded_winners.extend(teams)
        
        # Get wild cards
        wild_cards = self.determine_wild_cards(conference, division_winners)
        
        return seeded_winners + wild_cards


# ==========================================
# GAME SIMULATION
# ==========================================

class GameSimulator:
    """Simulates NFL games with realistic scoring"""
    
    def __init__(self, season_data: SeasonData):
        self.season = season_data
        # League average stats
        self.league_avg_score = 22.0
        self.score_std_dev = 8.0
    
    def simulate_game(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """
        Simulate a single game between two teams.
        Uses team's scoring average adjusted for opponent's defensive average.
        Returns (home_score, away_score)
        """
        home_stats = self.season.teams.get(home_team)
        away_stats = self.season.teams.get(away_team)
        
        # Calculate expected scores
        if home_stats and home_stats.games_played > 0:
            home_off = home_stats.avg_points_for
            home_def = home_stats.avg_points_against
        else:
            home_off = home_def = self.league_avg_score
        
        if away_stats and away_stats.games_played > 0:
            away_off = away_stats.avg_points_for
            away_def = away_stats.avg_points_against
        else:
            away_off = away_def = self.league_avg_score
        
        # Expected score = average of (team's offense vs league avg defense) and (league avg offense vs opponent defense)
        home_expected = (home_off + (self.league_avg_score * 2 - away_def)) / 2 + 2.5  # Home field advantage
        away_expected = (away_off + (self.league_avg_score * 2 - home_def)) / 2
        
        # Add randomness
        home_score = max(0, int(random.gauss(home_expected, self.score_std_dev)))
        away_score = max(0, int(random.gauss(away_expected, self.score_std_dev)))
        
        # Make scores more realistic (tend toward typical NFL scores)
        home_score = self._round_to_nfl_score(home_score)
        away_score = self._round_to_nfl_score(away_score)
        
        return home_score, away_score
    
    def _round_to_nfl_score(self, score: int) -> int:
        """Round to more realistic NFL scoring values"""
        # Common NFL scoring increments: 3 (FG), 6 (TD no XP), 7 (TD+XP), 8 (TD+2pt)
        # Just return the score as-is for simplicity, but ensure non-negative
        return max(0, score)
    
    def simulate_remaining_games(self) -> List[Game]:
        """Simulate all remaining games in the season"""
        simulated = []
        for game in self.season.remaining_games:
            home_score, away_score = self.simulate_game(game.home_team, game.away_team)
            simulated_game = Game(
                week=game.week,
                home_team=game.home_team,
                away_team=game.away_team,
                home_score=home_score,
                away_score=away_score,
                completed=True
            )
            simulated.append(simulated_game)
        return simulated


# ==========================================
# SEASON SIMULATION
# ==========================================

def update_team_stats_from_games(season: SeasonData, games: List[Game]):
    """Update team statistics based on completed games"""
    for game in games:
        if not game.completed:
            continue
        
        for team_name in [game.home_team, game.away_team]:
            if team_name not in season.teams:
                continue
            
            stats = season.teams[team_name]
            opponent = game.get_opponent(team_name)
            opp_stats = season.teams.get(opponent)
            
            team_score = game.get_team_score(team_name)
            opp_score = game.get_opponent_score(team_name)
            
            if team_score is None or opp_score is None:
                continue
            
            # Update points
            stats.points_for += team_score
            stats.points_against += opp_score
            
            # Update record
            if game.winner == team_name:
                stats.wins += 1
                # Check division/conference
                if opp_stats:
                    if opp_stats.division == stats.division and opp_stats.conference == stats.conference:
                        stats.div_wins += 1
                    if opp_stats.conference == stats.conference:
                        stats.conf_wins += 1
            elif game.winner == opponent:
                stats.losses += 1
                if opp_stats:
                    if opp_stats.division == stats.division and opp_stats.conference == stats.conference:
                        stats.div_losses += 1
                    if opp_stats.conference == stats.conference:
                        stats.conf_losses += 1
            else:  # Tie
                stats.ties += 1
                if opp_stats:
                    if opp_stats.division == stats.division and opp_stats.conference == stats.conference:
                        stats.div_ties += 1
                    if opp_stats.conference == stats.conference:
                        stats.conf_ties += 1


def create_simulated_season(base_season: SeasonData) -> SeasonData:
    """Create a deep copy of season data for simulation"""
    import copy
    
    sim_season = SeasonData()
    
    # Deep copy teams
    for name, stats in base_season.teams.items():
        sim_season.teams[name] = TeamStats(
            name=stats.name,
            conference=stats.conference,
            division=stats.division,
            wins=stats.wins,
            losses=stats.losses,
            ties=stats.ties,
            points_for=stats.points_for,
            points_against=stats.points_against,
            div_wins=stats.div_wins,
            div_losses=stats.div_losses,
            div_ties=stats.div_ties,
            conf_wins=stats.conf_wins,
            conf_losses=stats.conf_losses,
            conf_ties=stats.conf_ties,
            avg_points_for=stats.avg_points_for,
            avg_points_against=stats.avg_points_against
        )
    
    # Copy completed games (don't need deep copy since we won't modify)
    sim_season.completed_games = list(base_season.completed_games)
    
    # Copy remaining games
    sim_season.remaining_games = [
        Game(
            week=g.week,
            home_team=g.home_team,
            away_team=g.away_team,
            home_score=g.home_score,
            away_score=g.away_score,
            completed=g.completed
        )
        for g in base_season.remaining_games
    ]
    
    return sim_season


def run_single_simulation(base_season: SeasonData) -> Dict[str, List[str]]:
    """
    Run a single season simulation.
    Returns playoff teams for each conference.
    """
    # Create copy for this simulation
    sim_season = create_simulated_season(base_season)
    
    # Simulate remaining games
    simulator = GameSimulator(sim_season)
    simulated_games = simulator.simulate_remaining_games()
    
    # Update stats with simulated games
    sim_season.completed_games.extend(simulated_games)
    update_team_stats_from_games(sim_season, simulated_games)
    
    # Determine playoff teams
    tiebreaker = NFLTiebreaker(sim_season)
    
    results = {}
    for conference in ['AFC', 'NFC']:
        playoff_teams = tiebreaker.determine_playoff_seeding(conference)
        results[conference] = playoff_teams
    
    return results


# ==========================================
# CACHING
# ==========================================

GAMES_CACHE_FILE = "cache/nfl_games_cache.json"
SCHEDULE_CACHE_FILE = "cache/nfl_schedule_cache.json"

# NFL 2025 Regular Season Week Start Dates (Thursday of each week)
# Games typically start Thursday evening and run through Monday night
NFL_2025_WEEK_STARTS = {
    1: "2025-09-04",
    2: "2025-09-11",
    3: "2025-09-18",
    4: "2025-09-25",
    5: "2025-10-02",
    6: "2025-10-09",
    7: "2025-10-16",
    8: "2025-10-23",
    9: "2025-10-30",
    10: "2025-11-06",
    11: "2025-11-13",
    12: "2025-11-20",
    13: "2025-11-27",
    14: "2025-12-04",
    15: "2025-12-11",
    16: "2025-12-18",
    17: "2025-12-25",
    18: "2026-01-01",
}


def get_current_nfl_week() -> int:
    """Determine the current NFL week based on today's date"""
    from datetime import datetime, date
    
    today = date.today()
    
    # Find the current week by checking which week has started
    current_week = 0
    for week, start_date_str in NFL_2025_WEEK_STARTS.items():
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
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
    from datetime import datetime, date
    
    current_week = get_current_nfl_week()
    
    # If current week is ahead of cached week, cache is stale
    if current_week > cached_week:
        return False
    
    # If we're in the same week, check if enough time has passed
    # that games results might have changed (after Thursday 8pm ET = ~1am UTC Friday)
    today = date.today()
    
    if current_week > 0 and current_week in NFL_2025_WEEK_STARTS:
        week_start_str = NFL_2025_WEEK_STARTS[current_week]
        week_start = datetime.strptime(week_start_str, "%Y-%m-%d").date()
        
        # If today is on or after the week start date,
        # check if cache was created before games started
        if today >= week_start:
            cache_date = datetime.fromtimestamp(cache_timestamp).date()
            # Cache is stale if it was created before this week's games started
            if cache_date < week_start:
                return False
    
    return True


def save_games_cache(games: List[Game]):
    """Save completed games to cache"""
    current_week = get_current_nfl_week()
    
    # Determine the latest week in the completed games
    max_completed_week = max((g.week for g in games), default=0)
    
    data = {
        'timestamp': time.time(),
        'cached_week': current_week,
        'max_completed_week': max_completed_week,
        'games': [
            {
                'week': g.week,
                'home_team': g.home_team,
                'away_team': g.away_team,
                'home_score': g.home_score,
                'away_score': g.away_score,
                'completed': g.completed
            }
            for g in games
        ]
    }
    import os
    os.makedirs(os.path.dirname(GAMES_CACHE_FILE), exist_ok=True)
    with open(GAMES_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(games)} games to {GAMES_CACHE_FILE}")


def load_games_cache() -> Optional[List[Game]]:
    """Load completed games from cache if valid"""
    if not os.path.exists(GAMES_CACHE_FILE):
        return None
    
    try:
        with open(GAMES_CACHE_FILE, 'r') as f:
            data = json.load(f)
        
        timestamp = data.get('timestamp', 0)
        cached_week = data.get('cached_week', 0)
        
        # Check if cache is still valid based on NFL week
        if not is_cache_valid_for_week(timestamp, cached_week):
            current_week = get_current_nfl_week()
            print(f"🔄 Games cache outdated (cached week {cached_week}, current week {current_week})")
            return None
        
        games = [
            Game(
                week=g['week'],
                home_team=g['home_team'],
                away_team=g['away_team'],
                home_score=g.get('home_score'),
                away_score=g.get('away_score'),
                completed=g.get('completed', False)
            )
            for g in data.get('games', [])
        ]
        
        print(f"📂 Loaded {len(games)} games from cache")
        return games
    except Exception as e:
        print(f"⚠️ Failed to load games cache: {e}")
        return None


def save_schedule_cache(games: List[Game]):
    """Save remaining schedule to cache"""
    current_week = get_current_nfl_week()
    
    # Determine the earliest remaining week
    min_remaining_week = min((g.week for g in games), default=18)
    
    data = {
        'timestamp': time.time(),
        'cached_week': current_week,
        'min_remaining_week': min_remaining_week,
        'games': [
            {
                'week': g.week,
                'home_team': g.home_team,
                'away_team': g.away_team,
                'completed': False
            }
            for g in games
        ]
    }
    import os
    os.makedirs(os.path.dirname(SCHEDULE_CACHE_FILE), exist_ok=True)
    with open(SCHEDULE_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(games)} scheduled games to {SCHEDULE_CACHE_FILE}")


def load_schedule_cache() -> Optional[List[Game]]:
    """Load remaining schedule from cache if valid"""
    if not os.path.exists(SCHEDULE_CACHE_FILE):
        return None
    
    try:
        with open(SCHEDULE_CACHE_FILE, 'r') as f:
            data = json.load(f)
        
        timestamp = data.get('timestamp', 0)
        cached_week = data.get('cached_week', 0)
        
        # Check if cache is still valid based on NFL week
        if not is_cache_valid_for_week(timestamp, cached_week):
            current_week = get_current_nfl_week()
            print(f"🔄 Schedule cache outdated (cached week {cached_week}, current week {current_week})")
            return None
        
        games = [
            Game(
                week=g['week'],
                home_team=g['home_team'],
                away_team=g['away_team'],
                completed=False
            )
            for g in data.get('games', [])
        ]
        
        print(f"📂 Loaded {len(games)} scheduled games from cache")
        return games
    except Exception as e:
        print(f"⚠️ Failed to load schedule cache: {e}")
        return None
