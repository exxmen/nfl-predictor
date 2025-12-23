"""
Intangibles Module - Non-EPA factors affecting NFL game outcomes.

This module implements adjustments for "intangible" factors that influence
game results beyond traditional EPA-based scoring:

1. Rest Days / Schedule Difficulty
2. Turnover Luck Regression
3. Travel / Time Zone Changes
4. Division Familiarity
5. Weather Impact (optional)

Research sources:
- Frontiers in Behavioral Economics (2024): "Bye-bye, bye advantage"
- Harvard Sports Analysis (2014): "How Random Are Turnovers"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import numpy as np
import pandas as pd

from .tiebreakers import Game, TEAM_TO_DIVISION


# Team timezone mapping for travel adjustments
TEAM_TIMEZONES = {
    # Eastern Time
    'BUF': 'ET', 'MIA': 'ET', 'NE': 'ET', 'NYJ': 'ET',
    'BAL': 'ET', 'PIT': 'ET', 'PHI': 'ET', 'WAS': 'ET',
    # Central Time
    'CHI': 'CT', 'DET': 'CT', 'GB': 'CT', 'MIN': 'CT',
    # Mountain Time
    'DEN': 'MT',
    # Pacific Time
    'ARI': 'PT', 'LAR': 'PT', 'SF': 'PT', 'SEA': 'PT',
    # Central (South)
    'HOU': 'CT', 'IND': 'CT', 'JAX': 'CT', 'TEN': 'CT',
    'KC': 'CT', 'LV': 'PT', 'LAC': 'PT', 'DAL': 'CT', 'NO': 'CT',
    'TB': 'ET', 'ATL': 'ET', 'CAR': 'ET', 'CIN': 'ET', 'CLE': 'ET',
}

# Stadium coordinates for future distance calculations
STADIUM_COORDS = {
    'ARI': (33.5276, -112.2626),  # State Farm Stadium
    'ATL': (33.7556, -84.4015),   # Mercedes-Benz Stadium
    'BAL': (39.2776, -76.6219),   # M&T Bank Stadium
    'BUF': (42.7739, -78.7869),   # Highmark Stadium
    'CAR': (35.2259, -80.8531),   # Bank of America Stadium
    'CHI': (41.8623, -87.6167),   # Soldier Field
    'CIN': (39.0955, -84.4165),   # Paycor Stadium
    'CLE': (41.5061, -81.6996),   # Cleveland Browns Stadium
    'DAL': (32.7476, -97.0947),   # AT&T Stadium
    'DEN': (39.7439, -105.0201),  # Empower Field at Mile High
    'DET': (42.3401, -83.0658),   # Ford Field
    'GB': (44.5013, -88.0622),    # Lambeau Field
    'HOU': (29.6873, -95.4105),   # NRG Stadium
    'IND': (39.7602, -86.1639),   # Lucas Oil Stadium
    'JAX': (30.3230, -81.6375),   # TIAA Bank Field
    'KC': (39.0489, -94.4839),    # Arrowhead Stadium
    'LAC': (33.5491, -117.7804),  # SoFi Stadium
    'LAR': (33.5491, -117.7804),  # SoFi Stadium (shared)
    'LV': (37.7512, -122.1969),   # Allegiant Stadium
    'MIA': (25.9580, -80.2385),   # Hard Rock Stadium
    'MIN': (44.9734, -93.2581),   # U.S. Bank Stadium
    'NE': (42.0909, -71.2643),    # Gillette Stadium
    'NO': (29.9509, -90.0815),    # Caesars Superdome
    'NYG': (40.8122, -74.0745),   # MetLife Stadium
    'NYJ': (40.8122, -74.0745),   # MetLife Stadium (shared)
    'PHI': (39.9012, -75.1674),   # Lincoln Financial Field
    'PIT': (40.4468, -80.0158),   # Acrisure Stadium
    'SF': (37.4031, -122.2947),   # Levi's Stadium
    'SEA': (47.5952, -122.3318),  # Lumen Field
    'TB': (27.9759, -82.5033),    # Raymond James Stadium
    'TEN': (36.1665, -86.7714),   # Nissan Stadium
    'WAS': (38.9070, -77.0266),   # Northwest Stadium
}


@dataclass
class IntangiblesConfig:
    """Configuration for which intangibles to apply and their weights."""
    # Enable/disable specific intangibles
    use_rest_days: bool = True
    use_turnover_luck: bool = True
    use_travel_adjustment: bool = True
    use_division_familiarity: bool = True
    use_weather: bool = False  # Requires API, disabled by default

    # Rest days adjustment (points per game)
    bye_week_advantage: float = 0.3      # Post-2011 CBA: ~0.3, not significant
    mini_bye_advantage: float = 0.5      # Post-TNF: ~0.5, not significant
    mnf_disadvantage: float = 0.1        # MNF short week: ~0.1, barely exists
    long_rest_advantage: float = 1.0     # 10+ days: estimated

    # Turnover luck regression
    turnover_regression_rate: float = 0.55  # 54.7% of TO margin is luck
    turnover_points_weight: float = 0.5      # Each TO = ~0.5 points adjustment

    # Travel adjustment
    timezone_change_penalty: float = 1.5    # Per 2+ time zone change
    east_coast_early_game_penalty: float = 1.0  # West team in 1pm ET

    # Division familiarity
    division_underdog_boost: float = 1.5    # Division dogs get small boost

    # Weather (if enabled)
    bad_weather_scoring_reduction: float = 3.0  # Points reduction in bad weather


class IntangiblesCalculator:
    """
    Calculates intangible adjustments for NFL games.

    Based on research from:
    - Frontiers in Behavioral Economics (2024) - Rest differential
    - Harvard Sports Analysis (2014) - Turnover luck
    """

    def __init__(self, config: Optional[IntangiblesConfig] = None):
        self.config = config or IntangiblesConfig()
        self._last_game_dates: Dict[str, datetime] = {}
        self._team_turnover_margin: Dict[str, float] = {}

    def set_last_game_dates(self, dates: Dict[str, datetime]):
        """Set the last game date for each team (for rest calculations)."""
        self._last_game_dates = dates

    def set_team_turnover_margin(self, margin: Dict[str, float]):
        """Set the current turnover margin for each team."""
        self._team_turnover_margin = margin

    def calculate_rest_advantage(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        bye_teams: Optional[set] = None,
        is_thursday_night: bool = False,
        is_monday_night: bool = False,
        home_rest: Optional[int] = None,
        away_rest: Optional[int] = None
    ) -> Tuple[float, str]:
        """
        Calculate rest advantage in points for both teams.

        Args:
            home_team, away_team: Team abbreviations
            game_date: Date of the game
            bye_teams: Set of teams on bye
            is_thursday_night: If this is a Thursday night game
            is_monday_night: If this is a Monday night game
            home_rest: Actual rest days for home team (from schedule)
            away_rest: Actual rest days for away team (from schedule)

        Returns:
            Tuple of (net_advantage, description)
            - Positive = home team advantage
            - Negative = away team advantage

        Rest categories (from Frontiers 2024):
        - Bye: +6 to +8 days (post-2011: +0.31 ppg, not significant)
        - Mini-bye: 10-11 days (post-TNF: +0.48 ppg, not significant)
        - MNF: 6 days or less (+0.14 ppg, barely exists)
        """
        if not self.config.use_rest_days:
            return 0.0, ""

        bye_teams = bye_teams or set()

        # Use actual rest days from schedule if available
        if home_rest is not None and away_rest is not None:
            # Use the provided rest days directly
            pass
        else:
            # Fallback to calculation from last game dates
            home_rest = self._get_rest_days(home_team, game_date, bye_teams)
            away_rest = self._get_rest_days(away_team, game_date, bye_teams)

        rest_diff = home_rest - away_rest

        # Determine rest advantage category
        description = f"Rest: {home_team} {home_rest}d vs {away_team} {away_rest}d"

        # Bye week advantage (6+ day difference, one team on bye)
        if rest_diff >= 6:
            return self.config.bye_week_advantage, f"{description} (+{self.config.bye_week_advantage:.1f} bye)"
        elif rest_diff <= -6:
            return -self.config.bye_week_advantage, f"{description} (-{self.config.bye_week_advantage:.1f} bye)"

        # Mini-bye (post-TNF 10-day rest vs normal 7-day)
        # One team has 9-11 days, other has 7, diff >= 2
        if (home_rest >= 9 and away_rest <= 8 and rest_diff >= 2):
            return self.config.mini_bye_advantage, f"{description} (+{self.config.mini_bye_advantage:.1f} mini-bye)"
        if (away_rest >= 9 and home_rest <= 8 and rest_diff <= -2):
            return -self.config.mini_bye_advantage, f"{description} (-{self.config.mini_bye_advantage:.1f} mini-bye)"

        # MNF/short week disadvantage
        # One team has 6 days or less (played MNF)
        if is_monday_night:
            if home_rest <= 6 and away_rest > 6:
                return -self.config.mnf_disadvantage, f"{description} (-{self.config.mnf_disadvantage:.1f} MNF)"
            elif away_rest <= 6 and home_rest > 6:
                return self.config.mnf_disadvantage, f"{description} (+{self.config.mnf_disadvantage:.1f} MNF)"

        # Long rest advantage (10+ days vs normal)
        if home_rest >= 10 and away_rest == 7:
            return self.config.long_rest_advantage, f"{description} (+{self.config.long_rest_advantage:.1f} long rest)"
        if away_rest >= 10 and home_rest == 7:
            return -self.config.long_rest_advantage, f"{description} (-{self.config.long_rest_advantage:.1f} long rest)"

        return 0.0, description

    def _get_rest_days(self, team: str, game_date: date, bye_teams: set) -> int:
        """Calculate days of rest for a team before a game."""
        # Check if team is on bye (no previous game)
        if team in bye_teams or team not in self._last_game_dates:
            return 99  # Indicates coming off bye

        last_game = self._last_game_dates[team]
        if isinstance(last_game, str):
            last_game = datetime.fromisoformat(last_game)
        if isinstance(last_game, datetime):
            last_game = last_game.date()

        return (game_date - last_game).days

    def calculate_turnover_regression(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, str]:
        """
        Calculate turnover luck regression adjustment.

        Based on Harvard Sports Analysis (2014):
        - 54.7% of turnover differential is luck
        - Teams with extreme margins regress to mean

        Returns:
            Tuple of (adjustment, description)
            - Positive = home team expected to improve
            - Negative = away team expected to improve
        """
        if not self.config.use_turnover_luck:
            return 0.0, ""

        if home_team not in self._team_turnover_margin or away_team not in self._team_turnover_margin:
            return 0.0, "Turnover: No data"

        home_to = self._team_turnover_margin[home_team]
        away_to = self._team_turnover_margin[away_team]

        # Expected regression (apply regression rate to turnover margin)
        # Positive TO margin = likely to regress negative
        # Negative TO margin = likely to regress positive
        home_regression = -home_to * self.config.turnover_regression_rate
        away_regression = -away_to * self.config.turnover_regression_rate

        # Convert turnovers to points (each TO ~0.2 wins = ~1 point)
        home_points = home_regression * self.config.turnover_points_weight
        away_points = away_regression * self.config.turnover_points_weight

        net = home_points - away_points

        desc = f"Turnover regression: {home_team} {home_points:+.1f} vs {away_team} {away_points:+.1f}"
        return net, desc

    def calculate_travel_adjustment(
        self,
        home_team: str,
        away_team: str,
        is_early_et_game: bool = False
    ) -> Tuple[float, str]:
        """
        Calculate travel/time zone adjustment.

        Research indicates West Coast teams traveling East face disadvantage:
        - 2-3 hour time zone changes affect performance
        - Early ET games (1pm) particularly hard for West teams

        Returns:
            Tuple of (adjustment, description)
            - Positive = home team advantage
            - Negative = away team advantage
        """
        if not self.config.use_travel_adjustment:
            return 0.0, ""

        home_tz = TEAM_TIMEZONES.get(home_team, 'CT')
        away_tz = TEAM_TIMEZONES.get(away_team, 'CT')

        # Time zone offset from ET (-2, -1, 0, +1)
        tz_offsets = {'ET': 0, 'CT': -1, 'MT': -2, 'PT': -3}
        home_offset = tz_offsets.get(home_tz, 0)
        away_offset = tz_offsets.get(away_tz, 0)

        adjustment = 0.0
        details = []

        # Away team traveling East (less rest, jet lag)
        tz_diff = away_offset - home_offset
        if abs(tz_diff) >= 2:
            travel_adj = -self.config.timezone_change_penalty if tz_diff < 0 else self.config.timezone_change_penalty
            adjustment += travel_adj
            details.append(f"Timezone change: {tz_diff} zones ({travel_adj:+.1f})")

        # West Coast team in 1pm ET game (circadian disadvantage)
        if is_early_et_game and away_tz in ['PT', 'MT'] and home_tz in ['ET', 'CT']:
            adjustment -= self.config.east_coast_early_game_penalty
            details.append(f"West team early game (-{self.config.east_coast_early_game_penalty:.1f})")

        desc = f"Travel: {away_team} @ {home_team}" + (", ".join(details) if details else "")
        return adjustment, desc

    def calculate_division_familiarity(
        self,
        home_team: str,
        away_team: str,
        home_spread: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calculate division familiarity adjustment.

        Underdogs perform slightly better vs division rivals due to:
        - Familiarity with opponent's schemes
        - "Any given Sunday" effect in rivalry games

        Returns:
            Tuple of (adjustment, description)
            - Positive = home team advantage
            - Negative = away team advantage
        """
        if not self.config.use_division_familiarity:
            return 0.0, ""

        home_div = TEAM_TO_DIVISION.get(home_team)
        away_div = TEAM_TO_DIVISION.get(away_team)

        if home_div != away_div:
            return 0.0, ""

        # Division game - check if there's a clear underdog
        if home_spread is not None:
            if home_spread < -3:  # Home team favored by 3+
                # Away team is underdog, give them a boost
                return -self.config.division_underdog_boost, f"Division rival dog (+{self.config.division_underdog_boost:.1f})"
            elif home_spread > 3:  # Away team favored
                # Home team is underdog, give them a boost
                return self.config.division_underdog_boost, f"Division rival dog (+{self.config.division_underdog_boost:.1f})"

        return 0.0, f"Division game"

    def calculate_weather_adjustment(
        self,
        home_team: str,
        away_team: str,
        weather_data: Optional[Dict] = None
    ) -> Tuple[float, str]:
        """
        Calculate weather impact adjustment (requires API data).

        Bad weather affects:
        - Total scoring reduction
        - Passing teams more than running teams

        Args:
            weather_data: Dict with temp, wind, rain, snow

        Returns:
            Tuple of (scaling_factor, description)
            - < 1.0 = reduce scoring for both teams
        """
        if not self.config.use_weather or not weather_data:
            return 1.0, ""

        temp = weather_data.get('temp', 70)
        wind = weather_data.get('wind', 0)
        precip = weather_data.get('precip', 0)  # inches

        reduction = 0.0
        details = []

        # Cold weather (< 40°F)
        if temp < 40:
            cold_factor = (40 - temp) * 0.05
            reduction += cold_factor
            details.append(f"Cold: {temp}°F")

        # Wind (> 15 mph)
        if wind > 15:
            wind_factor = (wind - 15) * 0.1
            reduction += wind_factor
            details.append(f"Wind: {wind}mph")

        # Rain/Snow
        if precip > 0.1:
            precip_factor = precip * 0.5
            reduction += precip_factor
            details.append(f"Precip: {precip}\"")

        # Dome teams in bad weather
        dome_teams = {'ARI', 'DET', 'HOU', 'IND', 'LV', 'MIN', 'NO', 'ATL', 'DAL'}
        if home_team in dome_teams or away_team in dome_teams:
            # Dome team less affected by weather
            reduction *= 0.5
            details.append("(dome team)")

        # Calculate scoring reduction (cap at 20%)
        reduction = min(reduction, 0.2)

        if reduction > 0:
            desc = "Weather: " + ", ".join(details) + f" ({reduction*100:.0f}% scoring reduction)"
            return 1.0 - reduction, desc

        return 1.0, ""

    def calculate_total_intangibles(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        bye_teams: Optional[set] = None,
        is_thursday_night: bool = False,
        is_monday_night: bool = False,
        is_early_et_game: bool = False,
        is_division: bool = False,
        home_spread: Optional[float] = None,
        weather_data: Optional[Dict] = None,
        home_rest: Optional[int] = None,
        away_rest: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Calculate all intangible adjustments for a game.

        Args:
            home_team, away_team: Team abbreviations
            game_date: Date of the game
            bye_teams: Set of teams on bye
            is_thursday_night: If this is a Thursday night game
            is_monday_night: If this is a Monday night game
            is_early_et_game: If this is an early ET game (1pm or 4:05pm)
            is_division: If this is a division game
            home_spread: Point spread (home team perspective)
            weather_data: Dict with temp, wind, precip
            home_rest: Actual rest days for home team
            away_rest: Actual rest days for away team

        Returns:
            Dict with point adjustments and weather scaling factor
        """
        adjustments = {
            'rest_advantage': 0.0,
            'turnover_regression': 0.0,
            'travel_adjustment': 0.0,
            'division_familiarity': 0.0,
            'weather_scaling': 1.0,
            'total_adjustment': 0.0,
            'descriptions': []
        }

        # Rest days (use actual rest from schedule if available)
        rest_adj, rest_desc = self.calculate_rest_advantage(
            home_team, away_team, game_date, bye_teams,
            is_thursday_night, is_monday_night, home_rest, away_rest
        )
        adjustments['rest_advantage'] = rest_adj
        adjustments['descriptions'].append(rest_desc)

        # Turnover regression
        to_adj, to_desc = self.calculate_turnover_regression(home_team, away_team)
        adjustments['turnover_regression'] = to_adj
        adjustments['descriptions'].append(to_desc)

        # Travel
        travel_adj, travel_desc = self.calculate_travel_adjustment(
            home_team, away_team, is_early_et_game
        )
        adjustments['travel_adjustment'] = travel_adj
        adjustments['descriptions'].append(travel_desc)

        # Division familiarity (use is_division flag if available)
        if is_division:
            # Only apply division underdog boost if it's a division game
            div_adj, div_desc = self.calculate_division_familiarity(
                home_team, away_team, home_spread
            )
            adjustments['division_familiarity'] = div_adj
            adjustments['descriptions'].append(div_desc)
        else:
            # Check division from team mapping
            div_adj, div_desc = self.calculate_division_familiarity(
                home_team, away_team, home_spread
            )
            adjustments['division_familiarity'] = div_adj
            adjustments['descriptions'].append(div_desc)

        # Weather (scaling factor, not point adjustment)
        weather_scale, weather_desc = self.calculate_weather_adjustment(
            home_team, away_team, weather_data
        )
        adjustments['weather_scaling'] = weather_scale
        if weather_scale < 1.0:
            adjustments['descriptions'].append(weather_desc)

        # Total point adjustment (positive = home advantage)
        adjustments['total_adjustment'] = (
            adjustments['rest_advantage'] +
            adjustments['turnover_regression'] +
            adjustments['travel_adjustment'] +
            adjustments['division_familiarity']
        )

        return adjustments


def print_intangibles_summary(adjustments: Dict[str, any]):
    """Print formatted intangibles summary."""
    print("\n📊 INTANGIBLES BREAKDOWN")
    print("-" * 50)

    for desc in adjustments['descriptions']:
        if desc:
            print(f"  • {desc}")

    print("-" * 50)
    print(f"  Total Point Adjustment: {adjustments['total_adjustment']:+.1f}")
    if adjustments['weather_scaling'] < 1.0:
        print(f"  Weather Scoring Factor: {adjustments['weather_scaling']:.2%}")


if __name__ == "__main__":
    # Test the intangibles calculator
    config = IntangiblesConfig(
        use_rest_days=True,
        use_turnover_luck=True,
        use_travel_adjustment=True,
        use_division_familiarity=True,
        use_weather=False
    )

    calc = IntangiblesCalculator(config)

    # Set some test data
    from datetime import timedelta
    today = date.today()
    calc.set_last_game_dates({
        'KC': today - timedelta(days=7),
        'BUF': today - timedelta(days=10),  # Mini-bye
        'SF': today - timedelta(days=7),
        'SEA': today - timedelta(days=6),   # MNF
    })
    calc.set_team_turnover_margin({
        'KC': +8,   # High positive, likely to regress negative
        'BUF': -5,  # Negative, likely to regress positive
        'SF': +2,
        'SEA': -1,
    })

    # Test scenarios
    print("=" * 50)
    print("INTANGIBLES TEST SCENARIOS")
    print("=" * 50)

    # Scenario 1: Mini-bye advantage
    adj = calc.calculate_total_intangibles(
        home_team='BUF',
        away_team='KC',
        game_date=today,
        bye_teams=set()
    )
    print_intangibles_summary(adj)

    # Scenario 2: West Coast travel
    adj = calc.calculate_total_intangibles(
        home_team='KC',
        away_team='SF',
        game_date=today,
        is_early_et_game=True
    )
    print_intangibles_summary(adj)

    # Scenario 3: Division rival
    adj = calc.calculate_total_intangibles(
        home_team='KC',
        away_team='BUF',
        game_date=today,
        home_spread=-2.5
    )
    print_intangibles_summary(adj)
