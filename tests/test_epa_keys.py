"""Tests for the team-name normalization used across the simulation pipeline.

Covers the core bug fix: nfl_data_py keys EPA by abbreviation (KC) while the
simulation passes full names (Kansas City Chiefs). The simulator must resolve
BOTH key forms and warn loudly on any real miss instead of silently falling
back to league averages.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.team_names import to_full_name, to_abbreviation
from nfl_predictor.simulation import EPAGameSimulator


def test_to_full_name_resolves_both_forms():
    assert to_full_name("KC") == "Kansas City Chiefs"
    # Full name passes through unchanged
    assert to_full_name("Kansas City Chiefs") == "Kansas City Chiefs"
    # Unknown keys pass through unchanged (handled downstream as a miss)
    assert to_full_name("Nope FC") == "Nope FC"


def test_to_abbreviation_resolves_both_forms():
    assert to_abbreviation("Kansas City Chiefs") == "KC"
    assert to_abbreviation("KC") == "KC"


def _make_epa_df():
    return pd.DataFrame({
        "team": ["KC", "PHI", "BUF"],
        "off_epa": [0.10, 0.08, 0.05],
        "def_epa": [0.06, 0.07, 0.09],
        "ppg": [27.0, 26.0, 24.0],
        "ppg_allowed": [20.0, 19.0, 21.0],
        "off_momentum": [0.5, -0.2, 0.1],
        "def_momentum": [0.3, 0.4, -0.1],
    })


def test_simulator_indexes_by_both_abbrev_and_full_name():
    sim = EPAGameSimulator(epa_df=_make_epa_df())
    # Abbreviation lookup (backtest path)
    assert sim.get_team_stats("KC")["ppg"] == pytest.approx(27.0)
    # Full-name lookup (scheduler/simulation path)
    assert sim.get_team_stats("Kansas City Chiefs")["ppg"] == pytest.approx(27.0)
    # Both forms resolve to the SAME record object
    assert sim.get_team_stats("KC") is sim.get_team_stats("Kansas City Chiefs")


def test_lookup_miss_warns_and_falls_back():
    sim = EPAGameSimulator(epa_df=_make_epa_df())
    # Unknown team -> league average fallback, no crash
    stats = sim.get_team_stats("Not A Real Team")
    assert stats["ppg"] == pytest.approx(sim.league_avg_ppg)
    assert stats["off_epa"] == pytest.approx(sim.league_avg_off_epa)
    # The miss was recorded so we could detect silent degradation
    assert "Not A Real Team" in sim._lookup_miss_warned


def test_get_lambdas_are_deterministic():
    sim = EPAGameSimulator(epa_df=_make_epa_df())
    l1 = sim.get_lambdas("Kansas City Chiefs", "Buffalo Bills", {})
    l2 = sim.get_lambdas("Kansas City Chiefs", "Buffalo Bills", {})
    assert l1 == l2
    assert l1[0] >= 7.0 and l1[1] >= 7.0  # Poisson needs positive lambda


def test_market_anchor_pulls_toward_spread():
    from nfl_predictor.market import market_implied_scores, DEFAULT_TOTAL

    sim = EPAGameSimulator(epa_df=_make_epa_df(), market_weight=0.5)
    home, away = "Kansas City Chiefs", "Buffalo Bills"
    no_mkt_h, no_mkt_a = sim.get_lambdas(home, away, {})

    # A +7 home-favorite line implies specific market scores
    mi_h, mi_a = market_implied_scores(7.0, total=DEFAULT_TOTAL)

    # The blend must move the model lambda TOWARD the market value,
    # regardless of direction (model may over- or under-score vs market).
    mkt_h, mkt_a = sim.get_lambdas(home, away, {"home_spread": 7.0})
    assert abs(mkt_h - mi_h) < abs(no_mkt_h - mi_h)   # pulled toward market home
    assert abs(mkt_a - mi_a) < abs(no_mkt_a - mi_a)   # pulled toward market away

    # And it stays bounded between the two (convex combination, w=0.5)
    assert min(no_mkt_h, mi_h) - 1e-6 <= mkt_h <= max(no_mkt_h, mi_h) + 1e-6
