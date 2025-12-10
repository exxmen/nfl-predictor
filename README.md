# NFL Playoff Predictor

Monte Carlo simulation for NFL playoff probabilities with full tiebreaker support.

## 📊 Latest Predictions

**[View Live Predictions on Gist →](https://gist.github.com/exxmen/7c1a962fbe394a0cae6b5defe341faaa)**

Updated automatically via GitHub Actions every Tuesday and Friday during the NFL season.

## Features

- 🏈 **Real NFL Tiebreaker Rules**: Implements all 12 division and 11 wild card tiebreaker steps
- 📊 **Monte Carlo Simulation**: 100,000 simulations for accurate probability estimates
- 🚀 **Fast HTTP Scraping**: Gets standings and game data from Pro-Football-Reference (no browser needed)
- 📅 **Smart Caching**: Week-based cache invalidation (refreshes when new NFL week starts)
- ⚙️ **GitHub Actions**: Automated runs update a public Gist with latest predictions

## Quick Start

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/exxmen/nfl-predictor.git
cd nfl-predictor
uv run python main.py
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
3. **Simulates remaining games** using team scoring averages
4. **Applies full NFL tiebreakers** to determine playoff seeds
5. **Aggregates results** across 100,000 simulations

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
| `advanced_simulation.py` | Monte Carlo simulation engine |
| `nfl_tiebreakers.py` | NFL tiebreaker rules implementation |
| `pfr_scraper.py` | Pro-Football-Reference HTTP scraper |

## Output Example

```
🏈 AFC PLAYOFF PICTURE
--------------------------------------------------

Division Leaders:
  East: New England Patriots    Div:  79.3%  Playoff:  99.9%  Wins: 12.8
  North: Pittsburgh Steelers    Div:  71.7%  Playoff:  72.2%  Wins: 9.0
  South: Jacksonville Jaguars   Div:  56.7%  Playoff:  92.3%  Wins: 11.0
  West: Denver Broncos          Div:  87.1%  Playoff:  99.9%  Wins: 12.9

Wild Card Race:
  1. Buffalo Bills              WC:  74.1%  Playoff:  94.8%  Wins: 11.4
  2. Los Angeles Chargers       WC:  69.4%  Playoff:  82.3%  Wins: 10.7
  3. Indianapolis Colts         WC:  45.8%  Playoff:  76.0%  Wins: 10.4

Outside Looking In:
  1. Houston Texans             WC:  38.2%  Playoff:  51.4%  Wins: 9.5
  2. Baltimore Ravens           WC:   0.0%  Playoff:  22.1%  Wins: 7.8
  3. Cincinnati Bengals         WC:   0.0%  Playoff:   6.3%  Wins: 6.8
```

## Data Source

All NFL standings and game data is sourced from [Pro-Football-Reference.com](https://www.pro-football-reference.com/). 

## License

[MIT](LICENSE)
