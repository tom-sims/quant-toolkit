# === EWMA VOLATILITY INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns (daily typical), e.g.
#           returns = prices.pct_change().dropna()
#
# lam : decay factor (lambda), typical RiskMetrics value is 0.94 for daily data
#       higher lam = slower moving volatility
#
# periods_per_year : optional
#       annualises volatility by sqrt(periods_per_year)
#       if you pass None, no auto-import here (keep it explicit), but you can wire it to config later
#
# init_var : optional initial variance
#       if None, uses sample variance of returns
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# vol = ewma_volatility(aapl, lam=0.94, periods_per_year=252)
#
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# vol_df = ewma_volatility(rets, lam=0.94, periods_per_year=252)
#
# Returns:
# Series input -> Series of annualised EWMA volatility
# DataFrame input -> DataFrame of annualised EWMA volatility (per column)

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


def ewma_variance(returns, lam=0.94, init_var=None):
    df = _clean_returns(returns)

    lam = float(lam)
    if lam <= 0 or lam >= 1:
        raise ValueError("lam must be between 0 and 1 (e.g. 0.94).")

    if init_var is None:
        var0 = df.var(ddof=1)
    else:
        if np.isscalar(init_var):
            var0 = pd.Series(float(init_var), index=df.columns)
        else:
            v = np.asarray(init_var, dtype=float)
            if len(v) != df.shape[1]:
                raise ValueError("init_var length must match number of columns in returns.")
            var0 = pd.Series(v, index=df.columns)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    prev = var0.copy()
    first = True

    for t in df.index:
        x = df.loc[t]
        if first:
            out.loc[t] = prev.values
            first = False
        else:
            prev = lam * prev + (1.0 - lam) * (x ** 2)
            out.loc[t] = prev.values

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("ewma_variance")
    return out


def ewma_volatility(returns, lam=0.94, periods_per_year=None, init_var=None):
    var = ewma_variance(returns, lam=lam, init_var=init_var)
    vol = np.sqrt(var)

    if periods_per_year is not None:
        vol = vol * np.sqrt(float(periods_per_year))

    if isinstance(returns, pd.Series):
        return vol.rename("ewma_volatility")
    return vol
