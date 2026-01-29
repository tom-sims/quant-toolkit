import numpy as np
import pandas as pd


def validate_timeseries(df, require_datetime_index=True, allow_nans=False, min_rows=5):
    if df is None:
        raise ValueError("Input is None.")
    if len(df) < min_rows:
        raise ValueError("Not enough data. Need at least " + str(min_rows) + " rows, got " + str(len(df)) + ".")

    if require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Timeseries must have a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Timeseries index must be sorted ascending by date.")

    if not allow_nans:
        if isinstance(df, pd.Series):
            has_nan = df.isna().any()
        else:
            has_nan = df.isna().any().any()
        if has_nan:
            raise ValueError("Timeseries contains NaNs. Clean or forward-fill before modelling.")


def align_on_intersection(*series):
    """
    Align multiple series/dataframes on the intersection of their datetime indices.
    """
    if len(series) < 2:
        raise ValueError("Provide at least two series/dataframes to align.")

    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)

    aligned = []
    for s in series:
        aligned.append(s.loc[idx])

    return aligned


def compute_returns(prices, method="simple", dropna=True):
    """
    Convert prices to returns.
    method: "simple" or "log"
    """
    validate_timeseries(prices, allow_nans=False)

    if method not in ("simple", "log"):
        raise ValueError("method must be 'simple' or 'log'.")

    if method == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices).diff()

    return rets.dropna() if dropna else rets


def validate_weights(weights, tol=1e-6):
    w = np.asarray(weights, dtype=float)

    if np.any(~np.isfinite(w)):
        raise ValueError("Weights contain NaN/inf.")
    s = float(w.sum())
    if abs(s - 1.0) > tol:
        raise ValueError("Weights must sum to 1. Got " + format(s, ".6f") + ".")
