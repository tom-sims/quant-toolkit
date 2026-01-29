# === RISK-ADJUSTED METRICS INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be periodic SIMPLE returns (daily typical), e.g.
#           returns = prices.pct_change().dropna()
#
# benchmark_returns : Series (or single-column DataFrame)
#           same frequency as returns, used for beta/alpha/IR/TE/Treynor
#           will be aligned on intersecting dates automatically
#
# risk_free : optional (annual rate, not per-period)
#           if None, this file pulls a default from config.py (e.g. RISK_FREE_RATE)
#           if you pass risk_free=0.05, that means 5% per year
#
# periods_per_year : optional
#           if None, pulls default from config.py (daily=252, monthly=12, etc.)
#
# How to call:
#
# # single asset + benchmark
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# spy  = prices["SPY"].pct_change().dropna().rename("SPY")
#
# print(sharpe_ratio(aapl, risk_free=0.05))
# print(sortino_ratio(aapl, risk_free=0.05))
# print(beta(aapl, spy))
# print(information_ratio(aapl, spy))
#
# # multiple assets vs one benchmark
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# print(beta(rets, spy))               # returns a Series of betas
# print(alpha(rets, spy, risk_free=0.05))
#
# What each metric does:
#
# sharpe_ratio(returns, risk_free)
#   -> (annualised excess return) / (annualised volatility)
#
# sortino_ratio(returns, risk_free, mar=0.0)
#   -> (annualised excess return) / (annualised downside deviation below MAR)
#
# calmar_ratio(returns)
#   -> (CAGR) / (abs(max drawdown))
#
# tracking_error(returns, benchmark_returns)
#   -> annualised std dev of active returns (returns - benchmark)
#
# information_ratio(returns, benchmark_returns)
#   -> (annualised active return) / (tracking error)
#
# beta(returns, benchmark_returns)
#   -> sensitivity to benchmark: cov(r, b) / var(b)
#
# alpha(returns, benchmark_returns, risk_free)
#   -> CAPM-style alpha (annual):
#      (ann_return - rf) - beta * (ann_benchmark - rf)
#
# treynor_ratio(returns, benchmark_returns, risk_free)
#   -> (ann_return - rf) / beta
#
# r_squared(returns, benchmark_returns)
#   -> correlation^2 between returns and benchmark (how “explained” by benchmark)
#
# omega_ratio(returns, threshold=0.0)
#   -> sum(gains above threshold) / sum(losses below threshold)
#
# How to interpret numbers (quick rules):
#
# Sharpe (historical, not forecast)
# <0.5 weak | 0.5–1.0 ok | 1.0–1.5 good | >1.5 very good
#
# Sortino
# usually > Sharpe if upside volatility dominates; focuses on downside risk
#
# Calmar
# >1 is decent, >2 strong, >3 very strong (depends heavily on sample length)
#
# Beta
# ~1 market-like | <0.7 defensive | >1.3 aggressive
#
# Information Ratio
# >0.3 ok | >0.5 good | >0.8 strong (hard to sustain)
#
# Tracking Error
# higher = more benchmark deviation (active risk)
#
# Frequency notes:
# risk_free is treated as an ANNUAL rate and converted to per-period by rf/periods_per_year
# annualised vol uses sqrt(periods_per_year)
# annualised returns use compounding with periods_per_year


import numpy as np
import pandas as pd

try:
    from .. import config as _config
except Exception:
    _config = None

from .performance import (
    _clean_returns,
    _get_periods_per_year,
    annualized_return,
    annualized_volatility,
    downside_deviation,
    max_drawdown,
)


def _get_risk_free(risk_free):
    if risk_free is not None:
        return risk_free

    if _config is not None:
        for name in ("RISK_FREE_RATE", "RISK_FREE", "DEFAULT_RISK_FREE_RATE"):
            if hasattr(_config, name):
                try:
                    return float(getattr(_config, name))
                except Exception:
                    pass

    return 0.0


def _align_two(a, b):
    a = _clean_returns(a)
    b = _clean_returns(b)

    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    return a, b


def _as_series_or_scalar(values, original):
    if isinstance(original, pd.Series):
        return float(values.iloc[0])
    return values


def _annualize_rf(rf, periods_per_year):
    rf = float(rf)
    return rf / float(periods_per_year)


def _cov(a, b):
    return ((a - a.mean()) * (b - b.mean())).mean()


def sharpe_ratio(returns, risk_free=None, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)
    rf = _get_risk_free(risk_free)
    rf_per_period = _annualize_rf(rf, ppy)

    excess = df - rf_per_period
    ann_excess = annualized_return(excess, periods_per_year=ppy)
    ann_vol = annualized_volatility(df, periods_per_year=ppy)

    sr = ann_excess / ann_vol.replace(0, np.nan)
    return _as_series_or_scalar(sr, returns)


def sortino_ratio(returns, risk_free=None, mar=0.0, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)
    rf = _get_risk_free(risk_free)
    rf_per_period = _annualize_rf(rf, ppy)

    excess = df - rf_per_period
    ann_excess = annualized_return(excess, periods_per_year=ppy)
    dd = downside_deviation(df, mar=mar, periods_per_year=ppy)

    so = ann_excess / dd.replace(0, np.nan)
    return _as_series_or_scalar(so, returns)


def calmar_ratio(returns, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    ann_ret = annualized_return(df, periods_per_year=ppy)
    mdd = max_drawdown(df)

    denom = (-mdd).replace(0, np.nan)
    cr = ann_ret / denom
    return _as_series_or_scalar(cr, returns)


def tracking_error(returns, benchmark_returns, periods_per_year=None, ddof=1):
    r, b = _align_two(returns, benchmark_returns)
    ppy = _get_periods_per_year(periods_per_year)

    if isinstance(r, pd.DataFrame) and isinstance(b, pd.DataFrame) and b.shape[1] != 1:
        raise ValueError("benchmark_returns should be a Series or a single-column DataFrame.")

    b_series = b.iloc[:, 0]
    active = r.sub(b_series, axis=0)

    te = active.std(ddof=ddof) * np.sqrt(ppy)
    return _as_series_or_scalar(te, returns)


def information_ratio(returns, benchmark_returns, periods_per_year=None):
    r, b = _align_two(returns, benchmark_returns)
    ppy = _get_periods_per_year(periods_per_year)

    if isinstance(r, pd.DataFrame) and isinstance(b, pd.DataFrame) and b.shape[1] != 1:
        raise ValueError("benchmark_returns should be a Series or a single-column DataFrame.")

    b_series = b.iloc[:, 0]
    active = r.sub(b_series, axis=0)

    ann_active = annualized_return(active, periods_per_year=ppy)
    te = tracking_error(r, b_series, periods_per_year=ppy)

    ir = ann_active / te.replace(0, np.nan)
    return _as_series_or_scalar(ir, returns)


def beta(returns, benchmark_returns):
    r, b = _align_two(returns, benchmark_returns)

    if isinstance(r, pd.DataFrame) and isinstance(b, pd.DataFrame) and b.shape[1] != 1:
        raise ValueError("benchmark_returns should be a Series or a single-column DataFrame.")

    b_series = b.iloc[:, 0]
    var_b = ((b_series - b_series.mean()) ** 2).mean()
    if var_b == 0:
        if isinstance(returns, pd.Series):
            return np.nan
        return pd.Series(np.nan, index=r.columns)

    betas = []
    for col in r.columns:
        cov_rb = _cov(r[col], b_series)
        betas.append(cov_rb / var_b)

    betas = pd.Series(betas, index=r.columns)
    return _as_series_or_scalar(betas, returns)


def alpha(returns, benchmark_returns, risk_free=None, periods_per_year=None):
    r, b = _align_two(returns, benchmark_returns)
    ppy = _get_periods_per_year(periods_per_year)
    rf = _get_risk_free(risk_free)
    rf_per_period = _annualize_rf(rf, ppy)

    if isinstance(r, pd.DataFrame) and isinstance(b, pd.DataFrame) and b.shape[1] != 1:
        raise ValueError("benchmark_returns should be a Series or a single-column DataFrame.")

    b_series = b.iloc[:, 0]

    ann_r = annualized_return(r, periods_per_year=ppy)
    ann_b = annualized_return(b_series, periods_per_year=ppy)

    betas = beta(r, b_series)

    a = (ann_r - rf) - (betas * (ann_b - rf))
    return _as_series_or_scalar(a, returns)


def treynor_ratio(returns, benchmark_returns, risk_free=None, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)
    rf = _get_risk_free(risk_free)

    ann_r = annualized_return(df, periods_per_year=ppy)
    betas = beta(df, benchmark_returns)

    tr = (ann_r - rf) / betas.replace(0, np.nan)
    return _as_series_or_scalar(tr, returns)


def r_squared(returns, benchmark_returns):
    r, b = _align_two(returns, benchmark_returns)

    if isinstance(r, pd.DataFrame) and isinstance(b, pd.DataFrame) and b.shape[1] != 1:
        raise ValueError("benchmark_returns should be a Series or a single-column DataFrame.")

    b_series = b.iloc[:, 0]
    var_b = ((b_series - b_series.mean()) ** 2).mean()
    if var_b == 0:
        if isinstance(returns, pd.Series):
            return np.nan
        return pd.Series(np.nan, index=r.columns)

    out = []
    for col in r.columns:
        cov_rb = _cov(r[col], b_series)
        var_r = ((r[col] - r[col].mean()) ** 2).mean()
        if var_r == 0:
            out.append(np.nan)
            continue
        corr = cov_rb / np.sqrt(var_r * var_b)
        out.append(float(corr) ** 2)

    out = pd.Series(out, index=r.columns)
    return _as_series_or_scalar(out, returns)


def omega_ratio(returns, threshold=0.0):
    df = _clean_returns(returns)
    thr = float(threshold)

    gains = (df - thr).clip(lower=0).sum()
    losses = (thr - df).clip(lower=0).sum()

    om = gains / losses.replace(0, np.nan)
    return _as_series_or_scalar(om, returns)
