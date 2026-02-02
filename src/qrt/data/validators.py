from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import get_logger


logger = get_logger(__name__)

PandasObj = Union[pd.Series, pd.DataFrame]


@dataclass(frozen=True)
class ValidationResult:
    name: str
    rows: int
    cols: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


def _to_datetime_index(index: Any) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.DatetimeIndex(index)


def _standardize_index(obj: PandasObj) -> PandasObj:
    idx = _to_datetime_index(obj.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if isinstance(obj, pd.Series):
        out = obj.copy()
        out.index = idx
        return out
    out = obj.copy()
    out.index = idx
    return out


def _ensure_sorted_unique_index(obj: PandasObj, name: str) -> PandasObj:
    obj = _standardize_index(obj)
    if obj.index.has_duplicates:
        dup = obj.index[obj.index.duplicated()].unique()
        raise ValueError(f"{name}: duplicate timestamps found: {dup[:10].tolist()}")
    if not obj.index.is_monotonic_increasing:
        obj = obj.sort_index()
    return obj


def _require_non_empty(obj: PandasObj, name: str) -> None:
    if obj is None:
        raise ValueError(f"{name}: is None")
    if len(obj) == 0:
        raise ValueError(f"{name}: empty")
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 0:
        raise ValueError(f"{name}: no columns")


def _require_numeric(obj: PandasObj, name: str) -> None:
    if isinstance(obj, pd.Series):
        if not pd.api.types.is_numeric_dtype(obj.dtype):
            raise TypeError(f"{name}: expected numeric series, got {obj.dtype}")
        return
    for c in obj.columns:
        if not pd.api.types.is_numeric_dtype(obj[c].dtype):
            raise TypeError(f"{name}: column {c} is not numeric: {obj[c].dtype}")


def _require_finite(obj: PandasObj, name: str) -> None:
    if isinstance(obj, pd.Series):
        arr = obj.to_numpy(dtype=float, copy=False)
        if np.isfinite(arr).sum() == 0:
            raise ValueError(f"{name}: all values are NaN/inf")
        return
    arr = obj.to_numpy(dtype=float, copy=False)
    if np.isfinite(arr).sum() == 0:
        raise ValueError(f"{name}: all values are NaN/inf")


def _drop_all_nan_rows(obj: PandasObj) -> PandasObj:
    if isinstance(obj, pd.Series):
        return obj.dropna()
    return obj.dropna(how="all")


def _as_series(x: PandasObj, name: str, column: Optional[str] = None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x
        if s.name is None:
            s = s.rename(name)
        return s
    if column is not None:
        if column not in x.columns:
            raise ValueError(f"{name}: missing column {column}")
        s = x[column].rename(column)
        return s
    if x.shape[1] == 1:
        col = str(x.columns[0])
        return x.iloc[:, 0].rename(col)
    raise ValueError(f"{name}: expected Series or single-column DataFrame, got {list(x.columns)[:10]}")


def validate_timeseries(
    x: PandasObj,
    name: str,
    *,
    allow_nan: bool = True,
    require_positive: bool = False,
    column: Optional[str] = None,
) -> Tuple[PandasObj, ValidationResult]:
    _require_non_empty(x, name)
    x = _ensure_sorted_unique_index(x, name)
    if isinstance(x.index, pd.DatetimeIndex) is False:
        raise TypeError(f"{name}: index must be datetime-like")
    x = _drop_all_nan_rows(x)
    _require_non_empty(x, name)
    _require_numeric(x, name)
    _require_finite(x, name)
    if not allow_nan:
        if isinstance(x, pd.Series):
            if x.isna().any():
                raise ValueError(f"{name}: contains NaNs")
        else:
            if x.isna().any().any():
                raise ValueError(f"{name}: contains NaNs")
    if require_positive:
        if isinstance(x, pd.Series):
            bad = (x <= 0).sum()
            if bad:
                raise ValueError(f"{name}: contains {int(bad)} non-positive values")
        else:
            bad = (x <= 0).to_numpy().sum()
            if bad:
                raise ValueError(f"{name}: contains {int(bad)} non-positive values")
    start = pd.Timestamp(x.index.min()) if len(x) else None
    end = pd.Timestamp(x.index.max()) if len(x) else None
    cols = 1 if isinstance(x, pd.Series) else int(x.shape[1])
    res = ValidationResult(name=name, rows=int(len(x)), cols=cols, start=start, end=end)
    return x, res


def validate_prices(
    prices: PandasObj,
    name: str = "prices",
    *,
    column: Optional[str] = None,
) -> Tuple[pd.Series, ValidationResult]:
    s = _as_series(prices, name, column=column)
    s, res = validate_timeseries(s, name, allow_nan=True, require_positive=True)
    return s, res


def validate_returns(
    returns: PandasObj,
    name: str = "returns",
    *,
    column: Optional[str] = None,
) -> Tuple[pd.Series, ValidationResult]:
    s = _as_series(returns, name, column=column)
    s, res = validate_timeseries(s, name, allow_nan=True, require_positive=False)
    return s, res


def align_series(
    series: Sequence[pd.Series],
    names: Optional[Sequence[str]] = None,
    *,
    how: str = "inner",
    dropna: bool = True,
) -> pd.DataFrame:
    if len(series) == 0:
        raise ValueError("align_series: empty input")
    cleaned: list[pd.Series] = []
    for i, s in enumerate(series):
        nm = names[i] if names and i < len(names) else (s.name or f"s{i}")
        s2, _ = validate_timeseries(s, nm, allow_nan=True, require_positive=False)
        s2 = _as_series(s2, nm)
        cleaned.append(s2)

    df = pd.concat(cleaned, axis=1, join=how)
    df = _ensure_sorted_unique_index(df, "aligned")
    if dropna:
        df = df.dropna(how="any")
    if len(df) == 0:
        raise ValueError("align_series: alignment produced empty data")
    return df


def require_min_rows(x: PandasObj, min_rows: int, name: str) -> None:
    if len(x) < int(min_rows):
        raise ValueError(f"{name}: expected at least {int(min_rows)} rows, got {int(len(x))}")


def require_same_index(a: PandasObj, b: PandasObj, a_name: str = "a", b_name: str = "b") -> None:
    a2 = _ensure_sorted_unique_index(a, a_name)
    b2 = _ensure_sorted_unique_index(b, b_name)
    if not a2.index.equals(b2.index):
        raise ValueError(f"{a_name} and {b_name}: indices do not match")
