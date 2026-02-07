from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..data.validators import validate_returns
from ..types import BacktestResult
from .var import var as compute_var


def var_forecast_series(
    returns: pd.Series,
    *,
    level: float = 0.95,
    window: int = 252,
    horizon_days: int = 1,
    method: str = "historical",
    dist: str = "normal",
) -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    w = int(window)
    if w <= 2:
        raise ValueError("window must be > 2")
    h = int(horizon_days)
    if h < 1:
        raise ValueError("horizon_days must be >= 1")
    l = float(level)
    if not (0.5 < l < 1.0):
        raise ValueError("level must be in (0.5, 1)")

    x = r.astype(float)
    out = pd.Series(index=x.index, dtype=float, name=f"VaR_{l:g}_{method}")
    for i in range(w, len(x)):
        hist = x.iloc[i - w : i]
        res = compute_var(hist, level=l, horizon_days=h, method=method, window=None, dist=dist)
        out.iloc[i] = float(res.var)

    return out


def var_violations(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    r = returns.dropna().astype(float)
    v = var_series.dropna().astype(float)

    if len(r) == 0 or len(v) == 0:
        raise ValueError("Empty returns or VaR series")

    idx = r.index.intersection(v.index)
    if len(idx) == 0:
        v2 = v.shift(1)
        idx = r.index.intersection(v2.index)
        if len(idx) == 0:
            raise ValueError("No overlapping data between returns and VaR series")
        v = v2

    r = r.loc[idx]
    v = v.loc[idx]

    threshold = v.where(v < 0.0, -v)
    return (r < threshold).astype(int)




def kupiec_uc_test(violations: pd.Series, expected_rate: float) -> Tuple[float, float]:
    v = violations.dropna().astype(int)
    if len(v) == 0:
        return np.nan, np.nan
    n = int(len(v))
    x = int(v.sum())
    p = float(expected_rate)

    if not (0.0 < p < 1.0):
        raise ValueError("expected_rate must be in (0, 1)")

    phat = x / n
    if phat == 0.0:
        log_l1 = (n - x) * np.log(1.0 - p)
        log_l0 = (n - x) * np.log(1.0 - 0.0)
    elif phat == 1.0:
        log_l1 = x * np.log(p)
        log_l0 = x * np.log(1.0)
    else:
        log_l1 = x * np.log(p) + (n - x) * np.log(1.0 - p)
        log_l0 = x * np.log(phat) + (n - x) * np.log(1.0 - phat)

    stat = -2.0 * (log_l1 - log_l0)
    pvalue = 1.0 - float(stats.chi2.cdf(stat, df=1))
    return float(stat), float(pvalue)


def backtest_var(
    returns: pd.Series,
    *,
    level: float = 0.95,
    window: int = 252,
    horizon_days: int = 1,
    method: str = "historical",
    dist: str = "normal",
    test: str = "kupiec_uc",
) -> Tuple[BacktestResult, pd.Series, pd.Series]:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    if window is not None and len(r.dropna()) <= int(window) + 5:
        return (
            {"error": "insufficient_data", "nobs": int(len(r.dropna())), "window": int(window)},
            pd.Series(dtype=float),
            pd.Series(dtype=int),
        )

    l = float(level)
    alpha = 1.0 - l
    forecast = var_forecast_series(
        r,
        level=l,
        window=int(window),
        horizon_days=int(horizon_days),
        method=str(method),
        dist=str(dist),
    )
    violations = var_violations(r, forecast)

    nobs = int(violations.shape[0])
    count = int(violations.sum())
    rate = float(count / nobs) if nobs > 0 else np.nan

    t = str(test).lower().strip()
    if t in {"kupiec_uc", "kupiec", "uc"}:
        stat, pvalue = kupiec_uc_test(violations, expected_rate=float(alpha))
        result = BacktestResult(
            test="kupiec_uc",
            level=float(l),
            horizon_days=int(horizon_days),
            window=int(window),
            nobs=int(nobs),
            violations=int(count),
            violation_rate=float(rate),
            expected_rate=float(alpha),
            statistic=float(stat),
            pvalue=float(pvalue),
        )
        return result, forecast, violations

    raise ValueError(f"Unsupported test: {test}")
