import numpy as np

from evaluations.evaluator import choose_best_threshold_for_window
from evaluations.metrics import SweepConfig


def test_choose_best_threshold_for_window_end_to_end():
    # Synthetic window where high probs correlate with positive returns
    p_up = np.array([0.45, 0.52, 0.6, 0.75, 0.9])
    fwd = np.array([-0.004, 0.002, 0.003, 0.006, 0.012])
    res = choose_best_threshold_for_window(
        p_up_window=p_up,
        fwd_ret_window=fwd,
        interval_code="1h",
        fees_bps=5.0,
        slippage_bps=5.0,
        sweep=SweepConfig(0.5, 0.9, 0.05),
        best_metric="sharpe_like",
    )
    assert "threshold" in res and "sharpe_like" in res
    assert res["threshold"] >= 0.5
