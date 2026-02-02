from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union, Mapping


import numpy as np
import pandas as pd


Number = Union[int, float, np.number]
ArrayLike = Union[Sequence[Number], np.ndarray, pd.Series]


def as_float(x: Union[str, Number]) -> float:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("Empty numeric string")
        return float(s)
    raise TypeError(f"Unsupported type: {type(x)}")


def as_int(x: Union[str, int, float, np.number]) -> int:
    if isinstance(x, bool):
        raise TypeError("Bool is not a valid int")
    if isinstance(x, int):
        return int(x)
    if isinstance(x, (float, np.number)):
        if int(x) != float(x):
            raise ValueError(f"Expected integer-like value, got {x}")
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("Empty integer string")
        return int(s)
    raise TypeError(f"Unsupported type: {type(x)}")


def safe_divide(a: Number, b: Number, default: float = np.nan) -> float:
    denom = float(b)
    if denom == 0.0 or np.isnan(denom):
        return float(default)
    return float(a) / denom


def clip(x: ArrayLike, low: Optional[float] = None, high: Optional[float] = None) -> ArrayLike:
    if isinstance(x, pd.Series):
        return x.clip(lower=low, upper=high)
    arr = np.asarray(x, dtype=float)
    return np.clip(arr, low if low is not None else -np.inf, high if high is not None else np.inf)


def annualise_return(total_return: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0:
        return np.nan
    base = 1.0 + float(total_return)
    if base <= 0:
        return np.nan
    return base ** (float(periods_per_year) / float(periods)) - 1.0


def annualise_vol(vol: float, periods_per_year: int) -> float:
    return float(vol) * np.sqrt(float(periods_per_year))


def to_log_returns(r: ArrayLike) -> ArrayLike:
    if isinstance(r, pd.Series):
        return np.log1p(r)
    arr = np.asarray(r, dtype=float)
    return np.log1p(arr)


def to_simple_returns(r: ArrayLike) -> ArrayLike:
    if isinstance(r, pd.Series):
        return np.expm1(r)
    arr = np.asarray(r, dtype=float)
    return np.expm1(arr)


def compound_returns(r: ArrayLike) -> float:
    if isinstance(r, pd.Series):
        x = r.dropna().to_numpy(dtype=float, copy=False)
    else:
        x = np.asarray(r, dtype=float)
        x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.prod(1.0 + x) - 1.0)


def cumulative_curve(r: ArrayLike, start: float = 1.0) -> np.ndarray:
    if isinstance(r, pd.Series):
        x = r.fillna(0.0).to_numpy(dtype=float, copy=False)
    else:
        x = np.asarray(r, dtype=float)
        x = np.nan_to_num(x, nan=0.0)
    curve = np.empty_like(x, dtype=float)
    acc = float(start)
    for i in range(x.size):
        acc *= 1.0 + float(x[i])
        curve[i] = acc
    return curve


def drawdown_curve(curve: ArrayLike) -> np.ndarray:
    if isinstance(curve, pd.Series):
        x = curve.to_numpy(dtype=float, copy=False)
    else:
        x = np.asarray(curve, dtype=float)
    if x.size == 0:
        return np.asarray([], dtype=float)
    peaks = np.maximum.accumulate(x)
    dd = (x / peaks) - 1.0
    return dd


def max_drawdown(curve: ArrayLike) -> float:
    dd = drawdown_curve(curve)
    if dd.size == 0:
        return np.nan
    return float(np.min(dd))


def normalize_weights(weights: Union[Mapping[str, Number], Sequence[Number], np.ndarray, pd.Series]) -> Union[dict, np.ndarray, pd.Series]:
    if isinstance(weights, dict):
        total = float(sum(float(v) for v in weights.values()))
        if total == 0.0:
            raise ValueError("Weights sum to zero")
        return {str(k): float(v) / total for k, v in weights.items()}
    if isinstance(weights, pd.Series):
        total = float(weights.sum())
        if total == 0.0 or np.isnan(total):
            raise ValueError("Weights sum to zero")
        return weights / total
    arr = np.asarray(weights, dtype=float)
    total = float(np.sum(arr))
    if total == 0.0 or np.isnan(total):
        raise ValueError("Weights sum to zero")
    return arr / total


def ensure_1d(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def nanmean(x: ArrayLike) -> float:
    arr = ensure_1d(x)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def nanstd(x: ArrayLike, ddof: int = 1) -> float:
    arr = ensure_1d(x)
    if arr.size == 0:
        return np.nan
    return float(np.nanstd(arr, ddof=int(ddof)))


def quantile(x: ArrayLike, q: float) -> float:
    arr = ensure_1d(x)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, float(q)))


def cov_matrix(returns: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    df = returns.copy()
    df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("Empty returns after dropping NaNs")
    return df.cov(ddof=int(ddof))


def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    df = returns.copy()
    df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("Empty returns after dropping NaNs")
    return df.corr()
