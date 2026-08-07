"""Tests for the free market (spread) integration and backtest metrics."""

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.market import market_implied_scores, DEFAULT_TOTAL
from nfl_predictor.backtest import NFLBacktester


def test_market_implied_scores_sign_convention():
    # Positive spread = home favored (verified vs nflverse data) -> home scores more
    home, away = market_implied_scores(3.0, total=43.0)
    assert home == pytest.approx(23.0)
    assert away == pytest.approx(20.0)
    assert home > away

    # Negative spread = away favored -> away scores more
    home2, away2 = market_implied_scores(-3.0, total=43.0)
    assert home2 == pytest.approx(20.0)
    assert away2 == pytest.approx(23.0)
    assert home2 < away2


def test_market_prob_from_spread():
    # Home favored (positive spread) -> p > 0.5
    p_home_fav = NFLBacktester.market_prob_from_spread(3.0)
    assert p_home_fav > 0.5
    # Away favored (negative spread) -> p < 0.5
    p_away_fav = NFLBacktester.market_prob_from_spread(-3.0)
    assert p_away_fav < 0.5
    # Pick 'em -> p ~ 0.5
    assert abs(NFLBacktester.market_prob_from_spread(0.0) - 0.5) < 1e-3
    # Missing spread -> None
    assert NFLBacktester.market_prob_from_spread(None) is None


def _make_predictions():
    return [
        {"home_win_prob": 0.9, "home_won": True, "home_spread": 6.0},   # mkt: home fav, won -> correct
        {"home_win_prob": 0.8, "home_won": True, "home_spread": 4.0},   # mkt: home fav, won -> correct
        {"home_win_prob": 0.3, "home_won": False, "home_spread": -5.0}, # mkt: away fav, lost -> correct
        {"home_win_prob": 0.4, "home_won": False, "home_spread": -3.0}, # mkt: away fav, lost -> correct
        {"home_win_prob": 0.7, "home_won": False, "home_spread": None}, # no line
    ]


def test_calculate_brier_and_log_loss():
    bt = NFLBacktester()
    preds = _make_predictions()
    brier = bt.calculate_brier_score(preds)
    ll = bt.calculate_log_loss(preds)

    assert brier > 0
    assert ll > 0
    # Near-perfect predictions give near-zero (log-loss clips p to [0.001,0.999])
    perfect = [{"home_win_prob": 1.0, "home_won": True},
               {"home_win_prob": 0.0, "home_won": False}]
    assert bt.calculate_brier_score(perfect) < 1e-6
    assert bt.calculate_log_loss(perfect) < 0.01
    # Coin-flip baseline log-loss ≈ 0.693
    coinflip = [{"home_win_prob": 0.5, "home_won": True},
                {"home_win_prob": 0.5, "home_won": False}]
    assert abs(bt.calculate_log_loss(coinflip) - 0.693) < 0.01


def test_calculate_ece_perfect_calibration_is_zero():
    bt = NFLBacktester()
    # Perfectly calibrated: all predict 0.5, exactly half true half false
    # -> conf == acc in the single occupied bin -> ECE = 0
    preds = [{"home_win_prob": 0.5, "home_won": True} for _ in range(50)]
    preds += [{"home_win_prob": 0.5, "home_won": False} for _ in range(50)]
    assert bt.calculate_ece(preds, n_bins=10) < 1e-6


def test_calculate_ece_detects_miscalibration():
    bt = NFLBacktester()
    # Overconfident: predicts 0.9 but always loses -> ECE should be large
    preds = [{"home_win_prob": 0.9, "home_won": False} for _ in range(100)]
    ece = bt.calculate_ece(preds, n_bins=10)
    assert ece > 0.5  # |acc - conf| = |0 - 0.9| = 0.9 in the single bin


def test_market_benchmark_skips_missing_spreads():
    bt = NFLBacktester()
    mkt = bt.calculate_market_benchmark(_make_predictions())
    # 4 of 5 preds have a spread
    assert mkt["n_with_spread"] == 4
    assert mkt["market_brier"] is not None
    assert mkt["market_win_accuracy"] is not None
    # All 4 games with spreads were called correctly
    assert mkt["market_win_accuracy"] == pytest.approx(1.0)


def test_market_benchmark_all_missing():
    bt = NFLBacktester()
    preds = [{"home_win_prob": 0.6, "home_won": True, "home_spread": None}]
    mkt = bt.calculate_market_benchmark(preds)
    assert mkt["market_brier"] is None
    assert mkt["market_win_accuracy"] is None
    assert mkt["n_with_spread"] == 0


def test_market_benchmark_pickem_counts_half_credit():
    bt = NFLBacktester()
    # spread=0 -> pick'em (p=0.5): credited 0.5, not a silent miss
    preds = [{"home_win_prob": 0.5, "home_won": True, "home_spread": 0.0},
             {"home_win_prob": 0.8, "home_won": True, "home_spread": 4.0}]
    mkt = bt.calculate_market_benchmark(preds)
    assert mkt["n_with_spread"] == 2
    # 0.5 (pick'em) + 1.0 (correct favorite) = 1.5 / 2 = 0.75
    assert mkt["market_win_accuracy"] == pytest.approx(0.75)


def test_df_to_spreads_treats_nan_as_missing():
    """DataFrame float columns turn missing spreads into NaN; must map to None
    so attach_spreads_to_games doesn't anchor on a NaN spread."""
    import pandas as pd
    import numpy as np
    from nfl_predictor.market import _df_to_spreads, attach_spreads_to_games
    from nfl_predictor.tiebreakers import Game

    df = pd.DataFrame([
        {"home_full": "Kansas City Chiefs", "away_full": "Buffalo Bills", "spread": np.nan},
        {"home_full": "Green Bay Packers", "away_full": "Detroit Lions", "spread": 3.5},
    ])
    spreads = _df_to_spreads(df)
    assert spreads[("Kansas City Chiefs", "Buffalo Bills")] is None
    assert spreads[("Green Bay Packers", "Detroit Lions")] == 3.5

    # attach_spreads_to_games must NOT attach the NaN line
    games = [Game(2, "Kansas City Chiefs", "Buffalo Bills", completed=False),
             Game(2, "Green Bay Packers", "Detroit Lions", completed=False)]
    attached = attach_spreads_to_games(games, spreads)
    assert attached == 1
    assert games[0].home_spread is None
    assert games[1].home_spread == 3.5
