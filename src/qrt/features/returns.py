from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..data.validators import align_series, validate_prices, validate_returns, validate_timeseries


@dataclass(frozen=True)
class ReturnSpec:
    kind: str = "log"
    dropna: bool = True


def returns_from_prices(prices: pd.Series, kind: str = "log") -> pd.Series:
    p, _ = validate_prices(prices, name=prices.name or "prices")
    k = str(kind).strip().lower()
    if k not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    if k == "log":
        r = np.log(p).diff()
    else:
        r = p.pct_change()
    r = r.rename(p.name or "returns")
    return r


def prices_from_returns(returns: pd.Series, start: float = 1.0, kind: str = "log") -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    k = str(kind).strip().lower()
    if k not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    x = r.fillna(0.0)
    if k == "log":
        curve = float(start) * np.exp(np.cumsum(x.to_numpy(dtype=float)))
    else:
        curve = float(start) * np.cumprod(1.0 + x.to_numpy(dtype=float))
    out = pd.Series(curve, index=r.index, name=r.name or "curve")
    return out


def excess_returns(
    asset_returns: pd.Series,
    risk_free_returns: pd.Series,
    *,
    name: Optional[str] = None,
) -> pd.Series:
    df = align_series([asset_returns, risk_free_returns], names=["asset", "rf"], how="inner", dropna=True)
    ex = (df["asset"] - df["rf"]).rename(name or (asset_returns.name or "excess"))
    return ex


def align_asset_benchmark_rf(
    asset_prices: pd.Series,
    benchmark_prices: Optional[pd.Series] = None,
    risk_free: Optional[pd.Series] = None,
    *,
    kind: str = "log",
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    a = returns_from_prices(asset_prices, kind=kind)

    b = None
    if benchmark_prices is not None:
        b = returns_from_prices(benchmark_prices, kind=kind)

    rf = None
    if risk_free is not None:
        rf, _ = validate_timeseries(risk_free, "risk_free", allow_nan=True, require_positive=False)
        rf = rf.rename("rf")

    series = [a.rename("asset")]
    names = ["asset"]
    if b is not None:
        series.append(b.rename("benchmark"))
        names.append("benchmark")
    if rf is not None:
        series.append(rf.rename("rf"))
        names.append("rf")

    df = align_series(series, names=names, how="inner", dropna=True)

    asset_r = df["asset"].rename(asset_prices.name or "asset")
    bench_r = df["benchmark"].rename(benchmark_prices.name or "benchmark") if "benchmark" in df.columns else None
    rf_r = df["rf"].rename(risk_free.name or "rf") if "rf" in df.columns else None

    return asset_r, bench_r, rf_r


def resample_returns(returns: pd.Series, rule: str, kind: str = "log") -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    k = str(kind).strip().lower()
    if k not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    if rule is None or str(rule).strip() == "":
        return r
    if k == "log":
        out = r.resample(rule).sum().rename(r.name)
    else:
        out = (1.0 + r).resample(rule).prod().sub(1.0).rename(r.name)
    return out


def align_factor_data(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf_col: str = "RF",
) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.Series]]:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    f, _ = validate_timeseries(factors, "factors", allow_nan=True, require_positive=False)
    cols = list(f.columns)

    rf = None
    if rf_col in f.columns:
        rf = f[rf_col].rename("rf")
        f = f.drop(columns=[rf_col])

    df = pd.concat([r.rename("asset"), f], axis=1).dropna(how="any")
    if len(df) == 0:
        raise ValueError("No overlapping dates between returns and factors")
    out_r = df["asset"].rename(r.name or "asset")
    out_f = df.drop(columns=["asset"])

    if rf is not None:
        rf2 = rf.reindex(df.index)
        if rf2.isna().any():
            rf2 = rf2.fillna(method="ffill").fillna(method="bfill")
        rf2 = rf2.rename("rf")
        return out_r, out_f, rf2

    return out_r, out_f, None
