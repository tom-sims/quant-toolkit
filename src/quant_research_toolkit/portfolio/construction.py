# === PORTFOLIO CONSTRUCTION INPUTS ===
# prices : optional DataFrame of prices (columns=assets)
# returns : optional DataFrame/Series of returns (columns=assets). If prices is provided, returns can be None.
#
# weights : array-like of weights (length = n_assets). If None, functions build weights.
#
# rebalance : "M" (monthly), "W" (weekly), "D" (daily), or None
#            used in backtest-style helpers
#
# How to call:
#
# prices = yf.download(["AAPL","MSFT"], start="2020-01-01")["Adj Close"]
# rets = prices.pct_change().dropna()
#
# w_eq = equal_weight(rets)
# w_invvol = inverse_vol_weight(rets, window=63)
# w_top = top_n_weight(rets, scores=momentum_score(rets, lookback=126), n=1)
#
# port_rets = portfolio_returns(rets, w_eq)
# curve = equity_curve_from_returns(port_rets)
#
# Returns:
# weights -> numpy array (or pandas Series if you set as_series=True)
# portfolio_returns -> Series (portfolio return time series)

import numpy as np
import pandas as pd

def _to_df(x):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or "asset")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("Expected a pandas Series or DataFrame.")


def _clean_index(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def returns_from_prices(prices):
    df = _to_df(prices)
    df = _clean_index(df)
    return df.pct_change().dropna(how="any")


def _as_weights(w, columns=None, as_series=False):
    w = np.asarray(w, dtype=float).ravel()
    if w.sum() == 0:
        raise ValueError("weights sum to 0.")
    w = w / w.sum()

    if as_series and columns is not None:
        return pd.Series(w, index=list(columns), name="weight")
    return w


def equal_weight(returns, as_series=False):
    df = _clean_index(_to_df(returns))
    n = df.shape[1]
    w = np.ones(n, dtype=float) / float(n)
    return _as_weights(w, columns=df.columns, as_series=as_series)


def top_n_weight(returns, scores, n=5, long_only=True, as_series=False):
    df = _clean_index(_to_df(returns))

    if isinstance(scores, pd.Series):
        s = scores.reindex(df.columns)
    else:
        s = pd.Series(scores, index=df.columns)

    s = s.dropna()
    if len(s) == 0:
        raise ValueError("scores has no valid values.")

    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1.")

    top = s.sort_values(ascending=False).head(n).index
    w = pd.Series(0.0, index=df.columns)

    if long_only:
        w.loc[top] = 1.0 / float(len(top))
    else:
        # simple long/short: top n long, bottom n short
        bot = s.sort_values(ascending=True).head(n).index
        w.loc[top] = 0.5 / float(len(top))
        w.loc[bot] = -0.5 / float(len(bot))

    if as_series:
        return w.rename("weight")
    return w.values


def inverse_vol_weight(returns, window=63, vol_floor=1e-8, as_series=False):
    df = _clean_index(_to_df(returns))
    w = int(window)

    vol = df.rolling(w).std(ddof=1).iloc[-1]
    vol = vol.replace([np.inf, -np.inf], np.nan)

    inv = 1.0 / np.maximum(vol.values.astype(float), float(vol_floor))
    return _as_weights(inv, columns=df.columns, as_series=as_series)


def momentum_score(returns, lookback=126):
    df = _clean_index(_to_df(returns))
    lb = int(lookback)
    if lb < 2:
        raise ValueError("lookback must be >= 2.")

    # simple cumulative return over lookback
    s = (1.0 + df.tail(lb)).prod() - 1.0
    return s.rename("momentum")


def volatility_target_weights(returns, target_vol=0.10, window=63, periods_per_year=252, as_series=False):
    df = _clean_index(_to_df(returns))
    w = int(window)

    vol = df.rolling(w).std(ddof=1).iloc[-1] * np.sqrt(float(periods_per_year))
    vol = vol.replace([np.inf, -np.inf], np.nan)

    inv = 1.0 / np.maximum(vol.values.astype(float), 1e-8)
    base = _as_weights(inv, columns=df.columns, as_series=False)

    # scale to hit target vol (roughly, assuming uncorrelated assets)
    approx_port_vol = np.sqrt(np.sum((base * vol.values) ** 2))
    if approx_port_vol <= 0:
        scale = 1.0
    else:
        scale = float(target_vol) / float(approx_port_vol)

    w_scaled = base * scale
    return _as_weights(w_scaled, columns=df.columns, as_series=as_series)


def normalize_weights(weights, columns=None, as_series=False):
    return _as_weights(weights, columns=columns, as_series=as_series)


def portfolio_returns(returns, weights, rebalance=None):
    df = _clean_index(_to_df(returns))

    if isinstance(weights, pd.Series):
        w0 = weights.reindex(df.columns).values
    else:
        w0 = np.asarray(weights, dtype=float).ravel()

    if len(w0) != df.shape[1]:
        raise ValueError("weights length must match number of assets (columns in returns).")

    if rebalance is None:
        w = _as_weights(w0, columns=df.columns, as_series=False)
        port = df.values @ w
        return pd.Series(port, index=df.index, name="portfolio_returns")

    rebalance = str(rebalance).strip().upper()
    if rebalance not in ("D", "W", "M"):
        raise ValueError("rebalance must be one of: None, 'D', 'W', 'M'.")

    # Rebalance schedule: use period starts
    if rebalance == "D":
        groups = df.index
        # effectively constant weights each day
        w = _as_weights(w0, columns=df.columns, as_series=False)
        port = df.values @ w
        return pd.Series(port, index=df.index, name="portfolio_returns")

    if rebalance == "W":
        rule = "W-FRI"
    else:
        rule = "M"

    # Hold weights constant between rebalance points
    w = _as_weights(w0, columns=df.columns, as_series=False)
    port = []

    last_reb = None
    for dt, row in df.iterrows():
        if last_reb is None:
            last_reb = dt
        # check if we crossed into a new resample period
        if dt in df.resample(rule).asfreq().index and dt != last_reb:
            last_reb = dt
            w = _as_weights(w0, columns=df.columns, as_series=False)

        port.append(float(np.dot(row.values, w)))

    return pd.Series(port, index=df.index, name="portfolio_returns")


def equity_curve_from_returns(returns, start_value=1.0):
    s = _to_df(returns)
    s = _clean_index(s)
    if s.shape[1] != 1:
        raise ValueError("Expected a Series or single-column DataFrame.")
    r = s.iloc[:, 0]
    curve = (1.0 + r).cumprod() * float(start_value)
    return curve.rename("equity_curve")
