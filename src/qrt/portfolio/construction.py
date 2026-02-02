from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union, Mapping

import numpy as np
import pandas as pd

from ..data.validators import validate_timeseries
from ..utils.math import normalize_weights


@dataclass(frozen=True)
class PortfolioSpec:
    weights: Dict[str, float]

    def normalized(self) -> "PortfolioSpec":
        w = normalize_weights(self.weights)
        return PortfolioSpec(weights=dict(w))


def equal_weight(tickers: Sequence[str]) -> Dict[str, float]:
    names = [str(t).upper() for t in tickers if str(t).strip()]
    if len(names) == 0:
        raise ValueError("No tickers provided")
    w = 1.0 / float(len(names))
    return {t: float(w) for t in names}


def from_holdings(
    holdings: Union[Mapping[str, float], pd.Series, pd.DataFrame],
    *,
    value_col: str = "value",
) -> Dict[str, float]:
    if isinstance(holdings, pd.Series):
        raw = {str(k): float(v) for k, v in holdings.dropna().items()}
        return dict(normalize_weights(raw))
    if isinstance(holdings, pd.DataFrame):
        if value_col not in holdings.columns:
            raise ValueError(f"Missing {value_col} column")
        idx = holdings.index.astype(str)
        vals = holdings[value_col].astype(float).to_numpy()
        raw = {str(idx[i]): float(vals[i]) for i in range(len(idx)) if np.isfinite(vals[i])}
        return dict(normalize_weights(raw))
    if isinstance(holdings, dict):
        raw = {str(k): float(v) for k, v in holdings.items() if np.isfinite(float(v))}
        return dict(normalize_weights(raw))
    raise TypeError("Unsupported holdings type")


def clip_weights(
    weights: Dict[str, float],
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    renormalize: bool = True,
) -> Dict[str, float]:
    lo = float(min_weight)
    hi = float(max_weight)
    if lo < 0 or hi <= 0 or lo > hi:
        raise ValueError("Invalid min/max weights")
    w = {str(k): float(v) for k, v in weights.items()}
    clipped = {k: float(np.clip(v, lo, hi)) for k, v in w.items()}
    if not renormalize:
        return clipped
    return dict(normalize_weights(clipped))


def ensure_tickers_present(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    cols = set(str(c) for c in df.columns)
    out = {str(k): float(v) for k, v in weights.items() if str(k) in cols}
    if len(out) == 0:
        raise ValueError("No weights match the returns columns")
    return dict(normalize_weights(out))


def add_asset(
    weights: Dict[str, float],
    ticker: str,
    weight: float,
    *,
    renormalize_existing: bool = True,
) -> Dict[str, float]:
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("Empty ticker")
    w_new = float(weight)
    if w_new < 0:
        raise ValueError("weight must be >= 0")
    if w_new > 1:
        raise ValueError("weight must be <= 1")

    base = {str(k).upper(): float(v) for k, v in weights.items()}
    base.pop(t, None)

    if not renormalize_existing:
        base[t] = w_new
        return dict(normalize_weights(base))

    remaining = 1.0 - w_new
    if remaining < 0:
        raise ValueError("weight must be <= 1")
    if len(base) == 0:
        return {t: 1.0}

    base_norm = normalize_weights(base)
    scaled = {k: float(v) * remaining for k, v in base_norm.items()}
    scaled[t] = w_new
    return dict(normalize_weights(scaled))
