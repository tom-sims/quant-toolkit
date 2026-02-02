from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .validators import align_series, validate_timeseries, validate_prices, validate_returns


@dataclass(frozen=True)
class AlignmentInfo:
    start: pd.Timestamp
    end: pd.Timestamp
    rows: int
    columns: Tuple[str, ...]


def _as_series(x: pd.Series, name: str) -> pd.Series:
    s = x.copy()
    s.name = name
    return s


def common_index(
    series: Sequence[pd.Series],
    *,
    how: str = "inner",
    dropna: bool = True,
) -> pd.DatetimeIndex:
    df = align_series(series, how=how, dropna=dropna)
    return df.index


def align_prices(
    prices: Dict[str, pd.Series],
    *,
    how: str = "inner",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, AlignmentInfo]:
    if not prices:
        raise ValueError("prices is empty")
    series = []
    names = []
    for k, v in prices.items():
        s, _ = validate_prices(v, name=f"{k}_prices")
        series.append(s.rename(str(k)))
        names.append(str(k))
    df = pd.concat(series, axis=1, join=how)
    df, _ = validate_timeseries(df, "aligned_prices", allow_nan=True, require_positive=False)
    if dropna:
        df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("Aligned prices are empty")
    info = AlignmentInfo(
        start=pd.Timestamp(df.index.min()),
        end=pd.Timestamp(df.index.max()),
        rows=int(len(df)),
        columns=tuple(df.columns.astype(str)),
    )
    return df, info


def align_returns(
    returns: Dict[str, pd.Series],
    *,
    how: str = "inner",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, AlignmentInfo]:
    if not returns:
        raise ValueError("returns is empty")
    series = []
    names = []
    for k, v in returns.items():
        s, _ = validate_returns(v, name=f"{k}_returns")
        series.append(s.rename(str(k)))
        names.append(str(k))
    df = pd.concat(series, axis=1, join=how)
    df, _ = validate_timeseries(df, "aligned_returns", allow_nan=True, require_positive=False)
    if dropna:
        df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("Aligned returns are empty")
    info = AlignmentInfo(
        start=pd.Timestamp(df.index.min()),
        end=pd.Timestamp(df.index.max()),
        rows=int(len(df)),
        columns=tuple(df.columns.astype(str)),
    )
    return df, info


def align_asset_benchmark_rf(
    asset_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_returns: Optional[pd.Series] = None,
    *,
    how: str = "inner",
    dropna: bool = True,
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], AlignmentInfo]:
    a, _ = validate_returns(asset_returns, name=asset_returns.name or "asset")
    series = [_as_series(a, "asset")]
    names = ["asset"]

    if benchmark_returns is not None:
        b, _ = validate_returns(benchmark_returns, name=benchmark_returns.name or "benchmark")
        series.append(_as_series(b, "benchmark"))
        names.append("benchmark")

    if risk_free_returns is not None:
        rf, _ = validate_timeseries(risk_free_returns.rename("rf"), "rf", allow_nan=True, require_positive=False)
        series.append(_as_series(rf, "rf"))
        names.append("rf")

    df = pd.concat(series, axis=1, join=how)
    df, _ = validate_timeseries(df, "aligned_asset_benchmark_rf", allow_nan=True, require_positive=False)
    if dropna:
        df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("No overlapping dates after alignment")

    info = AlignmentInfo(
        start=pd.Timestamp(df.index.min()),
        end=pd.Timestamp(df.index.max()),
        rows=int(len(df)),
        columns=tuple(df.columns.astype(str)),
    )

    out_asset = df["asset"].rename(asset_returns.name or "asset")
    out_bench = df["benchmark"].rename(benchmark_returns.name or "benchmark") if "benchmark" in df.columns else None
    out_rf = df["rf"].rename(risk_free_returns.name or "rf") if "rf" in df.columns else None

    return out_asset, out_bench, out_rf, info


def align_returns_with_factors(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf_col: str = "RF",
    how: str = "inner",
    dropna: bool = True,
) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.Series], AlignmentInfo]:
    r, _ = validate_returns(asset_returns, name=asset_returns.name or "asset")
    f, _ = validate_timeseries(factors, "factors", allow_nan=True, require_positive=False)

    rf = None
    f2 = f.copy()
    if rf_col in f2.columns:
        rf = f2[rf_col].rename("rf")
        f2 = f2.drop(columns=[rf_col])

    df = pd.concat([r.rename("asset"), f2], axis=1, join=how)
    if dropna:
        df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("No overlapping dates between returns and factors")

    out_r = df["asset"].rename(asset_returns.name or "asset")
    out_f = df.drop(columns=["asset"])

    out_rf = None
    if rf is not None:
        out_rf = rf.reindex(df.index)
        if out_rf.isna().any():
            out_rf = out_rf.ffill().bfill()
        out_rf = out_rf.rename("rf")

    info = AlignmentInfo(
        start=pd.Timestamp(df.index.min()),
        end=pd.Timestamp(df.index.max()),
        rows=int(len(df)),
        columns=tuple(df.columns.astype(str)),
    )

    return out_r, out_f, out_rf, info


def enforce_business_days(df: pd.DataFrame) -> pd.DataFrame:
    x, _ = validate_timeseries(df, "dataframe", allow_nan=True, require_positive=False)
    idx = x.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")
    bidx = pd.bdate_range(start=idx.min(), end=idx.max())
    out = x.reindex(bidx)
    return out
