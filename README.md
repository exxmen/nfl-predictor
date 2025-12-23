# NFL Playoff Predictor

Monte Carlo simulation for NFL playoff probabilities using EPA analytics, momentum tracking, injury adjustments, and full tiebreaker support.

## 📊 Latest Predictions

**[View Live Predictions on Gist →](https://gist.github.com/exxmen/7c1a962fbe394a0cae6b5defe341faaa)**

Updated automatically via GitHub Actions every Tuesday and Friday during the NFL season.

## Features

- 📈 **EPA-Based Scoring Model**: Uses Expected Points Added (EPA) from play-by-play data with Poisson distribution for realistic score simulation
- 🔥 **Momentum/Recent Form**: Adjusts predictions based on team's last 4 games vs season average (hot streaks matter!)
- 🏥 **Injury Impact Analysis**: Scrapes ESPN injuries, matches to snap counts for starter detection, adjusts team strength based on player availability
- 🏈 **Real NFL Tiebreaker Rules**: Implements all 12 division and 11 wild card tiebreaker steps
- 📊 **Monte Carlo Simulation**: 100,000 simulations for accurate probability estimates
- ✅ **Validated Accuracy**: 73% win prediction accuracy, 0.19 Brier score (backtested on 2024 season)
- 🚀 **Fast HTTP Scraping**: Gets standings and game data from Pro-Football-Reference (no browser needed)
- 📅 **Smart Caching**: Week-based cache invalidation (refreshes when new NFL week starts)
- ⚙️ **GitHub Actions**: Automated runs update a public Gist with latest predictions
- 📈 **Intangibles**: Rest days, travel/timezone, turnover luck regression, division familiarity

## Quick Start

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/exxmen/nfl-predictor.git
cd nfl-predictor
uv run python main.py
```
## Intangibles

The simulator accounts for non-statistical factors that affect game outcomes:

| Factor | Effect | Source |
|--------|--------|--------|
| Bye Week | +0.5 ppg | Frontiers Behavioral Economics 2024 |
| Mini-bye (post-TNF) | +0.75 ppg | Frontiers Behavioral Economics 2024 |
| West→East Travel | +1.0 ppg (home advantage) | Various studies |
| Early ET Game | +0.75 ppg (vs West Coast team) | Various studies |
| Turnover Luck | 54.7% regression rate | Harvard Sports Analysis 2014 |
| Division Underdog | +0.75 ppg | Conventional wisdom |

Intangibles are enabled by default. To disable:

```bash
nfl-predict --no-intangibles
```

## Usage

### Interactive Mode
```bash
# Default: 10,000 simulations
uv run python main.py

# More simulations for higher accuracy
uv run python main.py -n 50000

# Simple mode (no tiebreakers, faster)
uv run python main.py --simple
```

### Scheduled Mode
```bash
# Test run (1,000 simulations)
uv run python scheduled_run.py --simulations 1000

# Production (100K simulations, saves to results/)
uv run python scheduled_run.py --simulations 100000
```

## How It Works

1. **Scrapes current standings** from Pro-Football-Reference
2. **Fetches completed games** and remaining schedule
3. **Loads injury data** from ESPN (current season) or nfl_data_py (historical)
4. **Calculates momentum** by comparing last 4 games EPA to season average
5. **Calculates injury impacts** using snap count data to identify starters
6. **Simulates remaining games** using EPA + momentum + injury adjustments
7. **Applies full NFL tiebreakers** to determine playoff seeds
8. **Aggregates results** across 100,000 simulations

## GitHub Actions

The workflow runs automatically:
- **Tuesday 10:00 UTC** - After Monday Night Football
- **Friday 10:00 UTC** - After Thursday Night Football

Results are published to the [Gist](https://gist.github.com/exxmen/7c1a962fbe394a0cae6b5defe341faaa) and saved as workflow artifacts.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point, interactive mode |
| `scheduled_run.py` | Automated script, saves results to files |
| `advanced_simulation.py` | Monte Carlo simulation engine with EPA-based Poisson scoring |
| `injury_loader.py` | ESPN injury scraper with nfl_data_py fallback |
| `player_impact.py` | Position-based injury impact calculation |
| `epa_loader.py` | Fetches and caches EPA data from nfl_data_py |
| `backtest.py` | Model validation against historical seasons |
| `nfl_tiebreakers.py` | NFL tiebreaker rules implementation |
| `pfr_scraper.py` | Pro-Football-Reference HTTP scraper |

## Backtest Results

Model validated against 2024 NFL season with injury data:

| Season | Week | Win Accuracy | Brier Score | Playoff Accuracy |
|--------|------|--------------|-------------|------------------|
| 2024   | 14   | 74.0%        | 0.1883      | 100.0%           |

*Brier score measures prediction calibration (lower = better, <0.22 = good)*

## Output Example

```
🏥 Loading injury data...
📋 Calculated injury impacts for 32 teams
   Teams with significant injury impacts:
     CLE: 120.0% impact
     TB: 115.0% impact
     HOU: 103.7% impact
     ...
     NE: 4.3% impact

🏈 AFC PLAYOFF PICTURE
--------------------------------------------------

Division Leaders:
  East: New England Patriots    Div:  87.7%  Playoff:  99.9%  Wins: 13.0
  North: Pittsburgh Steelers    Div:  76.4%  Playoff:  76.7%  Wins: 9.0
  South: Jacksonville Jaguars   Div:  49.6%  Playoff:  91.5%  Wins: 11.0
  West: Denver Broncos          Div:  90.7%  Playoff: 100.0%  Wins: 13.3

Wild Card Race:
  1. Buffalo Bills              WC:  74.7%  Playoff:  87.0%  Wins: 11.0
  2. Los Angeles Chargers       WC:  72.0%  Playoff:  81.3%  Wins: 10.7
  3. Houston Texans             WC:  43.8%  Playoff:  78.3%  Wins: 10.3

Outside Looking In:
  1. Indianapolis Colts         WC:  43.0%  Playoff:  58.8%  Wins: 10.0
  2. Baltimore Ravens           WC:   0.0%  Playoff:  20.8%  Wins: 7.7
  3. Cincinnati Bengals         WC:   0.0%  Playoff:   2.8%  Wins: 6.3
```

## Data Sources

- **Standings & Schedules**: [Pro-Football-Reference.com](https://www.pro-football-reference.com/)
- **EPA Play-by-Play Data**: [nfl_data_py](https://github.com/nflverse/nfl_data_py) (via [nflverse](https://nflverse.nflverse.com/))
- **Injury Reports**: [ESPN NFL Injuries](https://www.espn.com/nfl/injuries) (current season)

## License

[MIT](LICENSE)
