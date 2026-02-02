from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..data.validators import validate_returns
from ..utils.math import quantile
from ..types import VaRResult


def _finite_array(returns: pd.Series) -> np.ndarray:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    x = r.to_numpy(dtype=float, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite returns")
    return x


def historical_var(returns: pd.Series, level: float = 0.95, horizon_days: int = 1, window: Optional[int] = None) -> VaRResult:
    x = _finite_array(returns)
    if window is not None:
        w = int(window)
        if w <= 1:
            raise ValueError("window must be > 1")
        if x.size < w:
            raise ValueError("Not enough observations for window")
        x = x[-w:]
    l = float(level)
    if not (0.5 < l < 1.0):
        raise ValueError("level must be in (0.5, 1)")
    q = float(np.quantile(x, 1.0 - l))
    h = int(horizon_days)
    if h < 1:
        raise ValueError("horizon_days must be >= 1")
    v = q * np.sqrt(float(h))
    return VaRResult(level=l, horizon_days=h, method="historical", var=float(v), window=window)


def parametric_var(
    returns: pd.Series,
    level: float = 0.95,
    horizon_days: int = 1,
    window: Optional[int] = None,
    *,
    dist: str = "normal",
) -> VaRResult:
    x = _finite_array(returns)
    if window is not None:
        w = int(window)
        if w <= 1:
            raise ValueError("window must be > 1")
        if x.size < w:
            raise ValueError("Not enough observations for window")
        x = x[-w:]
    l = float(level)
    if not (0.5 < l < 1.0):
        raise ValueError("level must be in (0.5, 1)")
    h = int(horizon_days)
    if h < 1:
        raise ValueError("horizon_days must be >= 1")

    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    if not np.isfinite(sig) or sig <= 0.0:
        raise ValueError("Non-positive volatility")

    d = str(dist).lower().strip()
    if d in {"normal", "gaussian"}:
        z = float(stats.norm.ppf(1.0 - l))
        q = mu + sig * z
    elif d in {"t", "student", "student_t"}:
        nu = 5.0
        z = float(stats.t.ppf(1.0 - l, df=nu))
        q = mu + sig * z
    else:
        raise ValueError(f"Unsupported dist: {dist}")

    v = q * np.sqrt(float(h))
    return VaRResult(level=l, horizon_days=h, method=f"parametric_{d}", var=float(v), window=window)


def var(
    returns: pd.Series,
    level: float = 0.95,
    horizon_days: int = 1,
    *,
    method: str = "historical",
    window: Optional[int] = None,
    dist: str = "normal",
) -> VaRResult:
    m = str(method).lower().strip()
    if m in {"historical", "hist"}:
        return historical_var(returns, level=level, horizon_days=horizon_days, window=window)
    if m in {"parametric", "gaussian", "normal", "t"}:
        d = dist
        if m == "t":
            d = "t"
        if m in {"gaussian", "normal"}:
            d = "normal"
        return parametric_var(returns, level=level, horizon_days=horizon_days, window=window, dist=d)
    raise ValueError(f"Unsupported method: {method}")


def var_curve(
    returns: pd.Series,
    levels: Sequence[float] = (0.95, 0.99),
    horizon_days: int = 1,
    *,
    method: str = "historical",
    window: Optional[int] = None,
    dist: str = "normal",
) -> pd.Series:
    vals: Dict[float, float] = {}
    for l in levels:
        res = var(returns, level=float(l), horizon_days=horizon_days, method=method, window=window, dist=dist)
        vals[float(l)] = float(res.var)
    s = pd.Series(vals).sort_index()
    s.index.name = "level"
    s.name = "VaR"
    return s
