# === PERFORMANCE METRICS INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets / portfolio components)
#           values must be periodic SIMPLE returns (daily is typical), e.g.
#           returns = prices.pct_change().dropna()
#
# periods_per_year : optional
#           if None, this file pulls your default from config.py (e.g. PERIODS_PER_YEAR / TRADING_DAYS)
#           common values: daily=252, weekly=52, monthly=12
#
# Note on NaNs:
#           functions clean returns internally (sort index, drop NaNs/infs)
#
# How to call:
#
# # single asset
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# print(cagr(aapl))
# print(annualized_volatility(aapl))
# print(max_drawdown(aapl))
#
# # multiple assets
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# print(performance_summary(rets))
#
# What each metric does:
#
# equity_curve(returns)
#   -> growth of 1 unit of capital over time (compounds returns)
#
# cumulative_return(returns)
#   -> total compounded return over the full sample
#
# annualized_return(returns) / cagr(returns)
#   -> geometric annualised return from compounding
#
# annualized_volatility(returns)
#   -> standard deviation of returns, scaled by sqrt(periods_per_year)
#
# downside_deviation(returns, mar=0.0)
#   -> volatility of negative deviations below MAR (minimum acceptable return)
#   -> used by Sortino ratio
#
# drawdown_series(returns)
#   -> underwater curve: equity / running_peak - 1
#
# max_drawdown(returns)
#   -> worst peak-to-trough loss in the sample (most negative drawdown)
#
# max_drawdown_duration(returns)
#   -> longest consecutive time (in periods) spent below the previous peak
#
# hit_rate(returns)
#   -> fraction of periods with positive returns
#
# best_period(returns) / worst_period(returns)
#   -> best/worst single-period return
#
# avg_gain_loss(returns)
#   -> average positive return and average negative return
#
# skewness(returns) / kurtosis(returns)
#   -> distribution shape (kurtosis default = excess kurtosis)
#
# rolling_volatility(returns, window)
#   -> rolling annualised volatility over a window (in periods)
#
# performance_summary(returns)
#   -> quick report table: CAGR, annualised vol, max drawdown, hit rate
#
# How to interpret numbers (quick rules):
#
# CAGR (annualised return)
#   higher is better, but always compare against max drawdown + volatility
#
# Annualised vol
#   lower = smoother, higher = more risk (vol is NOT downside-only)
#
# Max drawdown
#   -0.20 = worst peak-to-trough loss of 20%
#   -0.50 = strategy halved at worst point (big risk)
#
# Hit rate
#   >0.55 is strong for many strategies; ~0.50 is coin-flip-like
#   a low hit rate can still be fine if avg_gain >> avg_loss
#
# Frequency notes:
# annualise return uses compounding with periods_per_year
# annualise vol uses sqrt(periods_per_year)


import numpy as np
import pandas as pd

try:
    from .. import config as _config
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


def _to_df(x):
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or "returns")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("Input must be a pandas Series or DataFrame.")


def _clean_returns(returns):
    df = _to_df(returns).copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="any")
    return df


def _as_series_or_scalar(values, original):
    if isinstance(original, pd.Series):
        return float(values.iloc[0])
    return values


def equity_curve(returns, start_value=1.0):
    df = _clean_returns(returns)
    curve = (1.0 + df).cumprod() * float(start_value)

    if isinstance(returns, pd.Series):
        return curve.iloc[:, 0].rename(returns.name or "equity_curve")
    return curve


def cumulative_return(returns):
    df = _clean_returns(returns)
    total = (1.0 + df).prod() - 1.0
    return _as_series_or_scalar(total, returns)


def annualized_return(returns, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    n = len(df)
    if n == 0:
        raise ValueError("No data after cleaning returns.")

    growth = (1.0 + df).prod()
    ann = growth ** (ppy / n) - 1.0
    return _as_series_or_scalar(ann, returns)


def cagr(returns, periods_per_year=None):
    return annualized_return(returns, periods_per_year=periods_per_year)


def annualized_volatility(returns, periods_per_year=None, ddof=1):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    vol = df.std(ddof=ddof) * np.sqrt(ppy)
    return _as_series_or_scalar(vol, returns)


def downside_deviation(returns, mar=0.0, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    mar = float(mar)
    downside = np.minimum(df - mar, 0.0)
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(ppy)
    return _as_series_or_scalar(dd, returns)


def drawdown_series(returns, start_value=1.0):
    curve = equity_curve(returns, start_value=start_value)

    if isinstance(curve, pd.Series):
        dd = curve / curve.cummax() - 1.0
        return dd.rename("drawdown")

    dd = curve / curve.cummax() - 1.0
    return dd


def max_drawdown(returns, start_value=1.0):
    dd = drawdown_series(returns, start_value=start_value)
    if isinstance(dd, pd.Series):
        return float(dd.min())
    return dd.min()


def max_drawdown_duration(returns, start_value=1.0):
    dd = drawdown_series(returns, start_value=start_value)

    def _duration_1d(x):
        underwater = x < 0
        if underwater.sum() == 0:
            return 0

        max_run = 0
        run = 0
        for v in underwater.values:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return int(max_run)

    if isinstance(dd, pd.Series):
        return _duration_1d(dd)

    return dd.apply(_duration_1d)


def hit_rate(returns):
    df = _clean_returns(returns)
    hr = (df > 0).mean()
    return _as_series_or_scalar(hr, returns)


def best_period(returns):
    df = _clean_returns(returns)
    best = df.max()
    return _as_series_or_scalar(best, returns)


def worst_period(returns):
    df = _clean_returns(returns)
    worst = df.min()
    return _as_series_or_scalar(worst, returns)


def avg_gain_loss(returns):
    df = _clean_returns(returns)

    pos = df.where(df > 0)
    neg = df.where(df < 0)

    avg_gain = pos.mean()
    avg_loss = neg.mean()

    if isinstance(returns, pd.Series):
        return {
            "avg_gain": float(avg_gain.iloc[0]) if not np.isnan(avg_gain.iloc[0]) else np.nan,
            "avg_loss": float(avg_loss.iloc[0]) if not np.isnan(avg_loss.iloc[0]) else np.nan,
        }

    return pd.DataFrame({"avg_gain": avg_gain, "avg_loss": avg_loss})


def skewness(returns):
    df = _clean_returns(returns)
    s = df.skew()
    return _as_series_or_scalar(s, returns)


def kurtosis(returns, fisher=True):
    df = _clean_returns(returns)
    k = df.kurtosis(fisher=fisher)
    return _as_series_or_scalar(k, returns)


def rolling_volatility(returns, window=20, periods_per_year=None, ddof=1):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    rv = df.rolling(int(window)).std(ddof=ddof) * np.sqrt(ppy)

    if isinstance(returns, pd.Series):
        return rv.iloc[:, 0].rename("rolling_volatility")
    return rv


def performance_summary(returns, periods_per_year=None):
    df = _clean_returns(returns)
    ppy = _get_periods_per_year(periods_per_year)

    ann_ret = annualized_return(df, periods_per_year=ppy)
    ann_vol = annualized_volatility(df, periods_per_year=ppy)
    mdd = max_drawdown(df)
    hr = hit_rate(df)

    if isinstance(returns, pd.Series):
        return {
            "cagr": float(ann_ret),
            "ann_vol": float(ann_vol),
            "max_drawdown": float(mdd),
            "hit_rate": float(hr),
        }

    return pd.DataFrame({
        "cagr": ann_ret,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "hit_rate": hr,
    })
