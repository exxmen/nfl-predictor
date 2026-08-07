# NFL Playoff Predictor

Monte Carlo simulation for NFL playoff probabilities using EPA analytics, momentum tracking, injury adjustments, and full tiebreaker support.

## 📊 Latest Predictions

**[View Live Predictions on Gist →](https://gist.github.com/exxmen/7c1a962fbe394a0cae6b5defe341faaa)**

Updated automatically via GitHub Actions every Tuesday and Friday during the NFL season.

## Features

- 📈 **EPA-Based Scoring Model**: Uses Expected Points Added (EPA) from play-by-play data with Poisson distribution for realistic score simulation
- 💰 **Market Anchoring**: Blends predictions toward consensus closing spreads (free via nfl_data_py schedule feed) — no paid API required
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
uv sync
uv run nfl-predict
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
uv run nfl-predict

# More simulations for higher accuracy
uv run nfl-predict -n 50000

# Simple mode (no tiebreakers, faster)
uv run nfl-predict --simple
```

### Scheduled Mode
```bash
# Test run (1,000 simulations)
uv run nfl-scheduled --simulations 1000

# Production (100K simulations, saves to results/)
uv run nfl-scheduled --simulations 100000
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

## Market Anchoring

When a consensus closing spread is available for a remaining game (pulled from the `spread_line` column of the free nfl_data_py schedule feed), the simulator blends its EPA-based expected score toward the market-implied score:

```
final = (1 - w) * model + w * market,  where w = 0.30 by default
```

This is the single highest-value accuracy lever: the closing line is the most-informed prior available. No API key or paid tier required. If a game has no line yet, it's simply skipped (market anchoring stays off for that game). The blend weight `MARKET_WEIGHT` in `market.py` can be tuned via backtest.

To disable market anchoring entirely, pass `market_weight=0.0` to `run_advanced_simulation`.

## GitHub Actions

The workflow runs automatically:
- **Tuesday 10:00 UTC** - After Monday Night Football
- **Friday 10:00 UTC** - After Thursday Night Football

Results are published to the [Gist](https://gist.github.com/exxmen/7c1a962fbe394a0cae6b5defe341faaa) and saved as workflow artifacts.

## Files

| File | Description |
|------|-------------|
| `nfl_predictor/cli.py` | Main entry point, interactive mode (`nfl-predict`) |
| `nfl_predictor/scheduler.py` | Automated scheduled runner, saves results (`nfl-scheduled`) |
| `nfl_predictor/simulation.py` | Monte Carlo engine with EPA-based Poisson scoring |
| `nfl_predictor/epa.py` | Fetches and caches EPA data from nfl_data_py |
| `nfl_predictor/market.py` | Consensus spread data (free via nfl_data_py) + market anchoring |
| `nfl_predictor/injuries.py` | ESPN injury scraper with nfl_data_py fallback |
| `nfl_predictor/player_impact.py` | Position-based injury impact calculation |
| `nfl_predictor/intangibles.py` | Rest, travel, weather, turnover-luck adjustments |
| `nfl_predictor/backtest.py` | Model validation: Brier, log-loss, calibration (ECE), market benchmark |
| `nfl_predictor/tiebreakers.py` | NFL tiebreaker rules implementation |
| `nfl_predictor/scraper.py` | Pro-Football-Reference HTTP scraper |

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
