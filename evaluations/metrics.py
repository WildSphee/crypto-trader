from dataclasses import dataclass
from typing import Dict
import numpy as np


MINUTES_MAP = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 24 * 60,
}


def bars_per_year_from_interval(interval_code: str) -> float:
    m = MINUTES_MAP[interval_code]
    return (365.0 * 24.0 * 60.0) / m


def one_bar_pnl(
    signal: np.ndarray, fwd_returns: np.ndarray, cost_per_roundtrip: float
) -> np.ndarray:
    """Long-only net P&L for each bar given 0/1 signals and next-bar returns.
    cost_per_roundtrip is cost for an enter+exit pair, applied on entry bars.
    """
    return signal * (fwd_returns - cost_per_roundtrip)


@dataclass
class SweepConfig:
    start: float = 0.50
    stop: float = 0.90
    step: float = 0.005

    def grid(self) -> np.ndarray:
        return np.round(np.arange(self.start, self.stop + 1e-12, self.step), 6)


def _stats_from_net(net: np.ndarray, signal: np.ndarray, interval_code: str) -> Dict:
    trades = int(signal.sum())
    hit_rate = float((net[signal == 1] > 0).mean()) if trades > 0 else float("nan")
    avg_net_ret_per_bar = float(np.nanmean(net)) if net.size else 0.0
    avg_net_ret_per_trade = (
        float(np.nanmean(net[signal == 1])) if trades > 0 else float("nan")
    )

    eq = np.cumprod(1.0 + np.nan_to_num(net, nan=0.0))
    total_ret = float(eq[-1] - 1.0) if len(eq) else 0.0

    mu = np.nanmean(net) if net.size else 0.0
    sd = np.nanstd(net, ddof=1) if net.size > 1 else 0.0
    bpy = bars_per_year_from_interval(interval_code)
    sharpe = float((mu / sd) * np.sqrt(bpy)) if sd > 0 else float("nan")

    return {
        "trades": trades,
        "hit_rate": hit_rate,
        "avg_net_ret_per_bar": avg_net_ret_per_bar,
        "avg_net_ret_per_trade": avg_net_ret_per_trade,
        "total_net_return": total_ret,
        "sharpe_like": sharpe,
    }


def evaluate_threshold(
    thr: float,
    p_up: np.ndarray,
    fwd_returns: np.ndarray,
    cost_roundtrip: float,
    interval_code: str,
) -> Dict:
    sig = (p_up >= thr).astype(int)
    net = one_bar_pnl(sig, fwd_returns, cost_roundtrip)
    stats = _stats_from_net(net, sig, interval_code)
    stats.update({"threshold": float(thr)})
    return stats


def sweep_thresholds(
    p_up: np.ndarray,
    fwd_returns: np.ndarray,
    cost_roundtrip: float,
    interval_code: str,
    sweep: SweepConfig,
    best_metric: str = "sharpe_like",
) -> Dict:
    """Return the dictionary of stats for the best threshold according to best_metric."""
    candidates = [
        evaluate_threshold(th, p_up, fwd_returns, cost_roundtrip, interval_code)
        for th in sweep.grid()
    ]
    if not candidates:
        return {"threshold": 0.5, "trades": 0}

    def keyfun(r):
        val = r.get(best_metric, float("nan"))
        return -np.inf if (val is None or np.isnan(val)) else val

    return max(candidates, key=keyfun)
