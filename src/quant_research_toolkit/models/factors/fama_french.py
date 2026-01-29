# === FF5 INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets, columns=tickers)
#           values must be simple daily returns, e.g. prices.pct_change().dropna()
#
# weights : only required if returns is a DataFrame (portfolio)
#           list of weights same length as columns, will be normalized automatically
#
# factors / rf : optional
#           pass these in if you already downloaded them elsewhere
#           factors must contain columns: ["Mkt-RF","SMB","HML","RMW","CMA"]
#           rf must be a Series named "RF"
#
# How to call:
#
# # single asset
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# model = FF5(aapl)
# print(model.params)
# print(model.rsquared_adj)
#
# # portfolio
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# model = FF5(rets, weights=[0.5, 0.5])
# print(model.params)
# print(model.rsquared_adj)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from ...io.cache import DiskCache

_FF5_URL_DAILY = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def _clean_ff_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_fama_french_5_factors_daily(use_cache=True, cache_dir=None):
    cache = None
    if use_cache:
        if cache_dir is None:
            cache_dir = Path.home() / ".quant_research_toolkit" / "cache"
        cache = DiskCache(cache_dir)

        cached = cache.get("ff5_daily_factors")
        cached_rf = cache.get("ff5_daily_rf")
        if cached is not None and cached_rf is not None:
            factors = cached
            rf = cached_rf.iloc[:, 0].rename("RF")
            return factors, rf

    ff = pd.read_csv(_FF5_URL_DAILY, compression="zip", skiprows=3)
    ff = _clean_ff_columns(ff)

    first_col = ff.columns[0]
    ff = ff[~ff[first_col].astype(str).str.contains("Copyright", na=False)]

    ff["Date"] = pd.to_datetime(ff[first_col], format="%Y%m%d", errors="coerce")
    ff = ff.dropna(subset=["Date"]).set_index("Date").sort_index()

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    rf_col = "RF"

    for c in factor_cols + [rf_col]:
        if c not in ff.columns:
            matches = [col for col in ff.columns if str(col).strip() == c]
            if matches:
                ff = ff.rename(columns={matches[0]: c})

    factors = ff[factor_cols].apply(pd.to_numeric, errors="coerce") / 100.0
    rf = pd.to_numeric(ff[rf_col], errors="coerce") / 100.0
    rf = rf.rename("RF")

    factors = factors.dropna()
    rf = rf.dropna()

    if use_cache and cache is not None:
        cache.set("ff5_daily_factors", factors)
        cache.set("ff5_daily_rf", rf.to_frame("RF"))

    return factors, rf


def _portfolio_from_returns(returns_df, weights):
    if weights is None:
        raise ValueError("weights is required when returns is a DataFrame (portfolio).")

    w = np.array(weights, dtype=float)
    if len(w) != returns_df.shape[1]:
        raise ValueError("weights length must match number of columns in returns DataFrame.")
    if w.sum() == 0:
        raise ValueError("weights sum to 0.")

    w = w / w.sum()
    port = returns_df.values @ w
    return pd.Series(port.ravel(), index=returns_df.index, name="Portfolio")


def _fit_ff5(excess_returns, aligned_factors):
    X = sm.add_constant(aligned_factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    model = sm.OLS(excess_returns, X).fit()
    return model


def FF5(returns, weights=None, factors=None, rf=None, use_cache=True, cache_dir=None):
    if factors is None or rf is None:
        factors, rf = get_fama_french_5_factors_daily(use_cache=use_cache, cache_dir=cache_dir)

    # clean inputs
    if isinstance(returns, pd.Series):
        name = returns.name or "Asset"
        aligned = pd.concat([returns.rename(name), rf, factors], axis=1, join="inner").dropna()
        excess = aligned[name] - aligned["RF"]
        return _fit_ff5(excess, aligned)

    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1 and weights is None:
            # treat 1-col DF as single asset if no weights provided
            s = returns.iloc[:, 0].rename(returns.columns[0] if returns.columns[0] is not None else "Asset")
            aligned = pd.concat([s, rf, factors], axis=1, join="inner").dropna()
            excess = aligned[s.name] - aligned["RF"]
            return _fit_ff5(excess, aligned)

        port = _portfolio_from_returns(returns.dropna(), weights)
        aligned = pd.concat([port, rf, factors], axis=1, join="inner").dropna()
        excess = aligned["Portfolio"] - aligned["RF"]
        return _fit_ff5(excess, aligned)

    raise TypeError("returns must be a pandas Series or DataFrame.")
