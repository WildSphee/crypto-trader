import math
from typing import List, Tuple

import numpy as np


def _phi(z: float) -> float:
    # standard normal CDF using math.erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _skew_kurtosis(x: np.ndarray) -> Tuple[float, float]:
    # sample skewness and kurtosis (not excess), NaN-safe
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    c = x - m
    m2 = float(np.mean(c**2))
    if m2 == 0.0:
        return float("nan"), float("nan")
    m3 = float(np.mean(c**3))
    m4 = float(np.mean(c**4))
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2)  # this is γ4 (not excess)
    return skew, kurt


def _sharpe(x: np.ndarray) -> float:
    # scale-invariant; works in bps or fractions
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan")
    s = float(np.std(x, ddof=0))
    if s == 0.0:
        return float("nan")
    return float(np.mean(x)) / s


def deflated_sharpe_ratio(
    net_returns_bps: np.ndarray,
    trial_srs: List[float],
) -> float:
    """
    Deflated Sharpe Ratio per Bailey & López de Prado.
    - net_returns_bps: series of per-trade net returns (bps) at the *chosen* threshold.
    - trial_srs: Sharpe ratios across the set of *trials* (e.g., thresholds) used to pick the best one.
    We estimate SR_0 from N= len(trial_srs) and V[SR_n] = var(trial_srs).
    """
    x = np.asarray(net_returns_bps, dtype=float)
    x = x[~np.isnan(x)]
    T = x.size
    if T < 2:
        return float("nan")

    # Observed SR at chosen threshold
    SR_hat = _sharpe(x)
    if not np.isfinite(SR_hat):
        return float("nan")

    # Moments on the *trade* return series used for SR_hat
    skew, kurt = _skew_kurtosis(x)
    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return float("nan")

    # Multiple-testing ingredients from the sweep
    trial_srs = [s for s in trial_srs if np.isfinite(s)]
    N = max(1, len(trial_srs))
    var_sr = float(np.var(trial_srs)) if N > 1 else 0.0
    sigma_sr = math.sqrt(var_sr)

    # Expected max SR under H0 (no skill) per Wikipedia/BLP approximation
    # SR_0 = sqrt(V[SR_n]) * [ (1-γ) Φ^{-1}(1-1/N) + γ Φ^{-1}(1-1/(Ne)) ]
    # We approximate Φ^{-1} via math.erf inverse using a rational approximation.
    # To keep things *minimal* and robust, if N==1 or var_sr==0, fall back to SR_0=0 (no deflation).
    if N > 1 and sigma_sr > 0.0:
        gamma_EM = 0.5772156649015329  # Euler–Mascheroni

        # simple probit via Acklam’s approximation (compact, no dependency)
        def _ppf(p: float) -> float:
            # bounds guard
            p = min(max(p, 1e-12), 1 - 1e-12)
            # Coeffs from Peter J. Acklam's inverse normal CDF
            a = [
                -3.969683028665376e01,
                2.209460984245205e02,
                -2.759285104469687e02,
                1.383577518672690e02,
                -3.066479806614716e01,
                2.506628277459239e00,
            ]
            b = [
                -5.447609879822406e01,
                1.615858368580409e02,
                -1.556989798598866e02,
                6.680131188771972e01,
                -1.328068155288572e01,
            ]
            c = [
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e00,
                -2.549732539343734e00,
                4.374664141464968e00,
                2.938163982698783e00,
            ]
            d = [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e00,
                3.754408661907416e00,
            ]
            plow = 0.02425
            phigh = 1 - plow
            if p < plow:
                q = math.sqrt(-2 * math.log(p))
                return (
                    ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
                ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            if phigh < p:
                q = math.sqrt(-2 * math.log(1 - p))
                return -(
                    ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
                ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )

        term1 = _ppf(1.0 - 1.0 / N)
        term2 = _ppf(1.0 - 1.0 / (N * math.e))
        SR0 = sigma_sr * ((1.0 - gamma_EM) * term1 + gamma_EM * term2)
    else:
        SR0 = 0.0  # no deflation possible ⇒ falls back to PSR against 0

    # Standard deviation around H0 (from BLP / Wikipedia)
    denom = 1.0 - skew * SR0 + ((kurt - 1.0) / 4.0) * (SR0**2)
    if denom <= 0.0 or T <= 1:
        return float("nan")
    sigma_h0 = math.sqrt(denom / (T - 1))

    z = (SR_hat - SR0) / (sigma_h0 if sigma_h0 > 0.0 else float("inf"))
    return _phi(z)
