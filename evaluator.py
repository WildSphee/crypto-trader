"""
Minimal evaluator you can import in test_model.py and bot.py to avoid duplication.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from metrics import SweepConfig, sweep_thresholds


def choose_best_threshold_for_window(
    p_up_window: np.ndarray,
    fwd_ret_window: np.ndarray,
    interval_code: str,
    fees_bps: float,
    slippage_bps: float,
    sweep: SweepConfig,
    best_metric: str = "sharpe_like",
) -> Dict:
    cost_rt = 2.0 * ((fees_bps + slippage_bps) / 10_000.0)
    return sweep_thresholds(
        p_up=p_up_window,
        fwd_returns=fwd_ret_window,
        cost_roundtrip=cost_rt,
        interval_code=interval_code,
        sweep=sweep,
        best_metric=best_metric,
    )