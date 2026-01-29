# === REALISED VOLATILITY INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns, e.g. prices.pct_change().dropna()
#
# window : rolling window length in periods (e.g. 20, 63, 252)
#
# method : "std" or "sqrt_sum_sq"
#          - "std"         : rolling standard deviation of returns
#          - "sqrt_sum_sq" : sqrt(sum(r^2)) over window (common for realized vol definitions)
#
# ddof : degrees of freedom for std method (default 1)
#
# periods_per_year : optional
#          annualises volatility by sqrt(periods_per_year)
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# rv = realized_volatility(aapl, window=20, method="std", periods_per_year=252)
#
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# rv_df = realized_volatility(rets, window=63, method="sqrt_sum_sq", periods_per_year=252)
#
# Returns:
# Series input -> Series
# DataFrame input -> DataFrame

import numpy as np
import pandas as pd

def _to_df(x):
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or "returns")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("returns must be a pandas Series or DataFrame.")


def _clean_returns(returns):
    df = _to_df(returns).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def realized_variance(returns, window=20, method="sqrt_sum_sq", ddof=1):
    df = _clean_returns(returns)
    w = int(window)

    m = str(method).strip().lower()
    if m not in ("std", "sqrt_sum_sq"):
        raise ValueError("method must be 'std' or 'sqrt_sum_sq'.")

    if m == "std":
        var = df.rolling(w).var(ddof=int(ddof))
    else:
        var = (df ** 2).rolling(w).sum()

    if isinstance(returns, pd.Series):
        return var.iloc[:, 0].rename("realized_variance")
    return var


def realized_volatility(returns, window=20, method="std", ddof=1, periods_per_year=None):
    var = realized_variance(returns, window=window, method=method, ddof=ddof)
    vol = np.sqrt(var)

    if periods_per_year is not None:
        vol = vol * np.sqrt(float(periods_per_year))

    if isinstance(returns, pd.Series):
        return vol.rename("realized_volatility")
    return vol
