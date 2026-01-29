# === CAPM INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets, columns=tickers)
#           values must be simple periodic returns (daily typical), e.g.
#           returns = prices.pct_change().dropna()
#
# benchmark_returns : Series (market benchmark, e.g. SPY returns)
#           must be same frequency as returns, will be aligned on intersecting dates
#
# weights : only required if returns is a DataFrame (portfolio)
#           list of weights same length as columns, will be normalized automatically
#
# risk_free : optional (annual rate)
#           if None, will try to read from config.py (RISK_FREE_RATE / RISK_FREE)
#           e.g. risk_free=0.05 means 5% per year
#
# periods_per_year : optional
#           if None, will try to read from config.py (PERIODS_PER_YEAR / TRADING_DAYS)
#           common values: daily=252, weekly=52, monthly=12
#
# How to call:
#
# # single asset
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# spy  = prices["SPY"].pct_change().dropna().rename("SPY")
# model = CAPM(aapl, spy, risk_free=0.05)
#
# # portfolio
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# spy  = prices["SPY"].pct_change().dropna().rename("SPY")
# model = CAPM(rets, spy, weights=[0.5, 0.5], risk_free=0.05)
#
# then:
# print(model.params)
# print(model.rsquared_adj)
#
# model outputs:
# model.params["const"]  -> alpha per period (e.g. daily alpha)
# model.params["MKT"]    -> beta to benchmark excess returns
# model.pvalues[...]     -> p-values
# model.rsquared         -> R-squared
# model.rsquared_adj     -> Adjusted R-squared

import pandas as pd
import numpy as np
import statsmodels.api as sm

try:
    from ... import config as _config
except Exception:
    _config = None

def _get_periods_per_year(periods_per_year):
    if periods_per_year is not None:
        return int(periods_per_year)

    if _config is not None:
        for name in ("PERIODS_PER_YEAR", "TRADING_DAYS", "PERIODS_PER_YEAR_DEFAULT", "DEFAULT_PERIODS_PER_YEAR"):
            if hasattr(_config, name):
                try:
                    return int(getattr(_config, name))
                except Exception:
                    pass

    return 252


def _get_risk_free(risk_free):
    if risk_free is not None:
        return float(risk_free)

    if _config is not None:
        for name in ("RISK_FREE_RATE", "RISK_FREE", "DEFAULT_RISK_FREE_RATE"):
            if hasattr(_config, name):
                try:
                    return float(getattr(_config, name))
                except Exception:
                    pass

    return 0.0


def _to_series(x, name=None):
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a Series or single-column DataFrame.")
        s = x.iloc[:, 0].copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    raise TypeError("Input must be a pandas Series or DataFrame.")


def _clean_index(x):
    x = x.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x


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


def _align_asset_and_benchmark(asset_returns, benchmark_returns):
    a = _clean_index(asset_returns)
    b = _clean_index(benchmark_returns)

    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    return a, b


def _fit_capm(excess_asset, excess_benchmark):
    X = sm.add_constant(excess_benchmark.rename("MKT"))
    model = sm.OLS(excess_asset, X).fit()
    return model


def CAPM(returns, benchmark_returns, weights=None, risk_free=None, periods_per_year=None):
    ppy = _get_periods_per_year(periods_per_year)
    rf_annual = _get_risk_free(risk_free)
    rf_per_period = rf_annual / float(ppy)

    bench = _to_series(benchmark_returns, name="Benchmark")

    if isinstance(returns, pd.Series):
        asset = _to_series(returns, name=returns.name or "Asset")
        asset, bench = _align_asset_and_benchmark(asset, bench)

        excess_asset = asset - rf_per_period
        excess_bench = bench - rf_per_period

        return _fit_capm(excess_asset, excess_bench)

    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1 and weights is None:
            asset = returns.iloc[:, 0].rename(returns.columns[0] if returns.columns[0] is not None else "Asset")
            asset, bench = _align_asset_and_benchmark(asset, bench)

            excess_asset = asset - rf_per_period
            excess_bench = bench - rf_per_period

            return _fit_capm(excess_asset, excess_bench)

        port = _portfolio_from_returns(_clean_index(returns), weights)
        port, bench = _align_asset_and_benchmark(port, bench)

        excess_port = port - rf_per_period
        excess_bench = bench - rf_per_period

        return _fit_capm(excess_port, excess_bench)

    raise TypeError("returns must be a pandas Series or DataFrame.")
