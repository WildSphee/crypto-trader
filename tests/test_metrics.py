import math

import numpy as np
import pytest

from evaluations.metrics import (
    SweepConfig,
    bars_per_year_from_interval,
    evaluate_threshold,
    one_bar_pnl,
    sweep_thresholds,
)


def test_bars_per_year_from_interval_basic():
    assert bars_per_year_from_interval("1h") == pytest.approx(365 * 24)
    assert bars_per_year_from_interval("5m") == pytest.approx(365 * 24 * 12)


def test_one_bar_pnl_long_only_with_costs():
    signal = np.array([1, 0, 1])
    fwd = np.array([0.01, -0.02, 0.005])
    cost_rt = 0.001  # 10 bps enter+exit
    net = one_bar_pnl(signal, fwd, cost_rt)
    assert net.tolist() == pytest.approx([0.009, 0.0, 0.004])


def test_evaluate_threshold_struct_and_metrics():
    p_up = np.array([0.2, 0.55, 0.8, 0.9, 0.51])
    fwd = np.array([-0.01, 0.005, 0.02, -0.003, 0.004])
    res = evaluate_threshold(
        thr=0.5,
        p_up=p_up,
        fwd_returns=fwd,
        cost_roundtrip=0.001,
        interval_code="1h",
    )
    for k in [
        "threshold",
        "trades",
        "hit_rate",
        "avg_net_ret_per_bar",
        "avg_net_ret_per_trade",
        "total_net_return",
        "sharpe_like",
    ]:
        assert k in res
    # trades fired where p_up>=0.5
    assert res["trades"] == 4
    # sanity: avg per-bar should be finite
    assert math.isfinite(res["avg_net_ret_per_bar"])


def test_sweep_thresholds_picks_better_threshold():
    # Returns are positive mostly when probability is high
    p_up = np.linspace(0.4, 0.95, 100)
    fwd = np.where(p_up > 0.7, 0.01, -0.005)
    best = sweep_thresholds(
        p_up=p_up,
        fwd_returns=fwd,
        cost_roundtrip=0.0,
        interval_code="1h",
        sweep=SweepConfig(0.5, 0.95, 0.01),
        best_metric="total_net_return",
    )
    assert best["threshold"] >= 0.68  # should bias towards the higher-prob region
