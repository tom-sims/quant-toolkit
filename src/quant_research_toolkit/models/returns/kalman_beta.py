# === KALMAN BETA INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns (daily typical), e.g.
#           returns = prices.pct_change().dropna()
#
# benchmark_returns : Series (market/benchmark returns, e.g. SPY)
#           same frequency as returns, will be aligned on intersecting dates
#
# rf : optional Series named "RF" OR a float annual risk-free rate
#           if Series: subtracts rf per period from both asset and benchmark
#           if float: treated as annual rate and converted to per-period using periods_per_year
#
# include_alpha : if True, estimates time-varying alpha AND beta
#                 if False, estimates beta only (regression through origin)
#
# delta : controls how quickly beta can move (process noise)
#         typical small values: 1e-5 to 1e-3
#         higher delta -> more responsive beta, noisier estimates
#
# ve : observation noise variance (measurement variance)
#      bigger ve -> smoother beta, smaller ve -> more reactive beta
#
# periods_per_year : only used if rf is a float annual rate and you want conversion to per-period
#           if None, tries config.py PERIODS_PER_YEAR / TRADING_DAYS, else defaults 252
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# spy  = prices["SPY"].pct_change().dropna().rename("SPY")
#
# out = kalman_beta(aapl, spy, include_alpha=True, delta=1e-4, ve=1e-3)
# betas = out["state"][["beta"]]
#
# If returns is a DataFrame:
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# out = kalman_beta(rets, spy)
# out["state"]   -> dict of DataFrames per asset
#
# Outputs:
# out["state"] : DataFrame (or dict of DataFrames) with time series of alpha/beta
# out["yhat"]  : fitted values (Series) [Series input only]
# out["resid"] : residuals (Series) [Series input only]

import numpy as np
import pandas as pd

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


def _clean_series(x, name=None):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected Series or single-column DataFrame.")
        x = x.iloc[:, 0]

    if not isinstance(x, pd.Series):
        raise TypeError("Expected a pandas Series or single-column DataFrame.")

    s = x.copy()
    if s.name is None and name is not None:
        s.name = name

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _align(a, b, rf):
    a = _clean_series(a, name=a.name or "Asset")
    b = _clean_series(b, name=b.name or "Benchmark")

    items = [a, b]

    rf_series = None
    if rf is not None and isinstance(rf, (pd.Series, pd.DataFrame)):
        rf_series = _clean_series(rf, name="RF")
        items.append(rf_series)

    idx = items[0].index
    for it in items[1:]:
        idx = idx.intersection(it.index)

    a = a.loc[idx]
    b = b.loc[idx]
    if rf_series is not None:
        rf_series = rf_series.loc[idx]

    return a, b, rf_series


def _kalman_filter(y, x, include_alpha, delta, ve, init_cov):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    n = len(y)
    if include_alpha:
        k = 2
    else:
        k = 1

    beta = np.zeros(k, dtype=float)
    P = np.eye(k, dtype=float) * float(init_cov)

    state = np.zeros((n, k), dtype=float)
    yhat = np.zeros(n, dtype=float)
    resid = np.zeros(n, dtype=float)

    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    ve = float(ve)
    if ve <= 0:
        raise ValueError("ve must be > 0.")
    init_cov = float(init_cov)
    if init_cov <= 0:
        raise ValueError("init_cov must be > 0.")

    for t in range(n):
        if include_alpha:
            H = np.array([1.0, x[t]], dtype=float)
        else:
            H = np.array([x[t]], dtype=float)

        Q = (delta / (1.0 - delta)) * P
        R = P + Q

        yhat_t = float(H @ beta)
        e = y[t] - yhat_t

        S = float(H @ R @ H.T + ve)
        if S <= 0:
            S = ve

        K = (R @ H.T) / S
        beta = beta + K * e
        P = R - np.outer(K, H) @ R

        state[t, :] = beta
        yhat[t] = yhat_t
        resid[t] = e

    return state, yhat, resid


def kalman_beta(
    returns,
    benchmark_returns,
    rf=None,
    include_alpha=True,
    delta=1e-4,
    ve=1e-3,
    init_cov=1e5,
    periods_per_year=None,
):
    if isinstance(returns, pd.DataFrame):
        bench = _clean_series(benchmark_returns, name="Benchmark")
        out_state = {}

        for col in returns.columns:
            asset = returns[col].rename(str(col))
            a, b, rf_series = _align(asset, bench, rf)

            if rf_series is not None:
                y = a - rf_series
                x = b - rf_series
            else:
                if rf is not None and np.isscalar(rf):
                    ppy = _get_periods_per_year(periods_per_year)
                    rf_per_period = float(rf) / float(ppy)
                    y = a - rf_per_period
                    x = b - rf_per_period
                else:
                    y = a
                    x = b

            state, _, _ = _kalman_filter(
                y.values,
                x.values,
                include_alpha=bool(include_alpha),
                delta=delta,
                ve=ve,
                init_cov=init_cov,
            )

            if include_alpha:
                df_state = pd.DataFrame(state, index=y.index, columns=["alpha", "beta"])
            else:
                df_state = pd.DataFrame(state, index=y.index, columns=["beta"])

            out_state[str(col)] = df_state

        return {"state": out_state}

    asset, bench, rf_series = _align(returns, benchmark_returns, rf)

    if rf_series is not None:
        y = asset - rf_series
        x = bench - rf_series
    else:
        if rf is not None and np.isscalar(rf):
            ppy = _get_periods_per_year(periods_per_year)
            rf_per_period = float(rf) / float(ppy)
            y = asset - rf_per_period
            x = bench - rf_per_period
        else:
            y = asset
            x = bench

    state, yhat, resid = _kalman_filter(
        y.values,
        x.values,
        include_alpha=bool(include_alpha),
        delta=delta,
        ve=ve,
        init_cov=init_cov,
    )

    if include_alpha:
        df_state = pd.DataFrame(state, index=y.index, columns=["alpha", "beta"])
    else:
        df_state = pd.DataFrame(state, index=y.index, columns=["beta"])

    yhat_s = pd.Series(yhat, index=y.index, name="yhat")
    resid_s = pd.Series(resid, index=y.index, name="resid")

    return {
        "state": df_state,
        "yhat": yhat_s,
        "resid": resid_s,
        "asset_excess": y,
        "benchmark_excess": x,
    }
