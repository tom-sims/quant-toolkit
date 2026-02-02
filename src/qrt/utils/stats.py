from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


def _to_1d(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _finite(x: ArrayLike) -> np.ndarray:
    arr = _to_1d(x)
    return arr[np.isfinite(arr)]


def mean(x: ArrayLike) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def std(x: ArrayLike, ddof: int = 1) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    return float(np.std(arr, ddof=int(ddof)))


def sem(x: ArrayLike, ddof: int = 1) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    s = np.std(arr, ddof=int(ddof))
    return float(s / np.sqrt(arr.size))


def skew(x: ArrayLike, bias: bool = False) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    return float(stats.skew(arr, bias=bias))


def kurtosis(x: ArrayLike, fisher: bool = True, bias: bool = False) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    return float(stats.kurtosis(arr, fisher=fisher, bias=bias))


def quantile(x: ArrayLike, q: float) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, float(q)))


def winsorize_series(x: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError("winsorize_series expects a pandas Series")
    lo, hi = float(limits[0]), float(limits[1])
    if lo < 0 or hi < 0 or lo >= 1 or hi >= 1:
        raise ValueError("winsorize limits must be in [0, 1)")
    a = x.astype(float)
    q_lo = a.quantile(lo)
    q_hi = a.quantile(1.0 - hi)
    return a.clip(lower=q_lo, upper=q_hi)


def zscore(x: ArrayLike) -> np.ndarray:
    arr = _to_1d(x).astype(float)
    mu = np.nanmean(arr)
    sig = np.nanstd(arr, ddof=1)
    if not np.isfinite(sig) or sig == 0.0:
        return np.full_like(arr, np.nan, dtype=float)
    return (arr - mu) / sig


@dataclass(frozen=True)
class OLSResult:
    dep: str
    indep: Tuple[str, ...]
    params: Dict[str, float]
    tstats: Dict[str, float]
    pvalues: Dict[str, float]
    stderr: Dict[str, float]
    r2: float
    adj_r2: float
    nobs: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dep": self.dep,
            "indep": list(self.indep),
            "params": self.params,
            "tstats": self.tstats,
            "pvalues": self.pvalues,
            "stderr": self.stderr,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "nobs": self.nobs,
        }


def ols(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    add_const: bool = True,
    dep_name: Optional[str] = None,
) -> OLSResult:
    import statsmodels.api as sm

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x must be a pandas DataFrame")

    y2 = y.astype(float).copy()
    x2 = x.astype(float).copy()
    df = pd.concat([y2.rename("y"), x2], axis=1).dropna(how="any")
    if len(df) < 5:
        raise ValueError("Not enough observations for OLS")

    y3 = df["y"]
    x3 = df.drop(columns=["y"])
    if add_const:
        x3 = sm.add_constant(x3, has_constant="add")

    model = sm.OLS(y3, x3).fit()

    params = {str(k): float(v) for k, v in model.params.items()}
    tstats = {str(k): float(v) for k, v in model.tvalues.items()}
    pvalues = {str(k): float(v) for k, v in model.pvalues.items()}
    stderr = {str(k): float(v) for k, v in model.bse.items()}

    dep = dep_name or (y.name if y.name else "y")
    indep = tuple(str(c) for c in x3.columns)

    return OLSResult(
        dep=dep,
        indep=indep,
        params=params,
        tstats=tstats,
        pvalues=pvalues,
        stderr=stderr,
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        nobs=int(model.nobs),
    )


def rolling_ols(
    y: pd.Series,
    x: pd.DataFrame,
    window: int,
    *,
    add_const: bool = True,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x must be a pandas DataFrame")
    w = int(window)
    if w <= 2:
        raise ValueError("window must be > 2")
    mp = int(min_periods) if min_periods is not None else w

    df = pd.concat([y.rename("y"), x], axis=1).dropna(how="any")
    if len(df) < mp:
        raise ValueError("Not enough observations for rolling OLS")

    cols = list(x.columns)
    out_cols = (["const"] + cols) if add_const else cols
    out = pd.DataFrame(index=df.index, columns=out_cols, dtype=float)

    import statsmodels.api as sm

    for i in range(mp - 1, len(df)):
        j0 = i - w + 1
        if j0 < 0:
            continue
        chunk = df.iloc[j0 : i + 1]
        yv = chunk["y"].astype(float)
        xv = chunk[cols].astype(float)
        if add_const:
            xv = sm.add_constant(xv, has_constant="add")
        res = sm.OLS(yv, xv).fit()
        out.iloc[i, :] = res.params.reindex(out_cols).to_numpy(dtype=float)

    return out
