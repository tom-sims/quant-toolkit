# === ROLLING FACTOR REGRESSION INPUTS ===
# returns : Series (single asset) or DataFrame (portfolio components)
#           values must be simple periodic returns (daily typical)
#
# factors : DataFrame of factor returns (same frequency)
#           must include the columns you want to regress on
#           e.g. FF5 factors: ["Mkt-RF","SMB","HML","RMW","CMA"]
#
# rf : optional Series named "RF"
#           if provided, the regression is run on excess returns (returns - RF)
#           if not provided, regression is run on raw returns
#
# weights : only required if returns is a DataFrame (portfolio)
#           list of weights same length as columns, normalized automatically
#
# window : rolling window length in periods (e.g. 252 for ~1 year of daily data)
#
# add_const : include intercept (alpha) if True
#
# min_obs : minimum observations inside a window to run regression
#
# How to call:
#
# from quant_research_toolkit.models.factors import FF5, get_fama_french_5_factors_daily
# from quant_research_toolkit.models.factors.rolling_factors import rolling_factor_regression
#
# factors, rf = get_fama_french_5_factors_daily()
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
#
# out = rolling_factor_regression(aapl, factors, rf=rf, window=252)
# betas = out["betas"]
# alpha = out["alpha"]
# r2    = out["r2"]

import pandas as pd
import numpy as np
import statsmodels.api as sm

try:
    from ... import config as _config
except Exception:
    _config = None

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


def _clean_df(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


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


def _align_all(asset, factors, rf):
    f = _clean_df(factors)
    a = _clean_index(asset)

    items = [a, f]
    if rf is not None:
        r = _to_series(rf, name="RF")
        r = _clean_index(r)
        items.append(r)

    idx = items[0].index
    for it in items[1:]:
        idx = idx.intersection(it.index)

    a = a.loc[idx]
    f = f.loc[idx]

    if rf is not None:
        r = items[2].loc[idx]
        return a, f, r

    return a, f, None


def rolling_factor_regression(
    returns,
    factors,
    rf=None,
    weights=None,
    window=252,
    add_const=True,
    min_obs=None,
):
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1 and weights is None:
            asset = returns.iloc[:, 0].rename(returns.columns[0] if returns.columns[0] is not None else "Asset")
        else:
            asset = _portfolio_from_returns(_clean_df(returns), weights)
    elif isinstance(returns, pd.Series):
        asset = returns.rename(returns.name or "Asset")
    else:
        raise TypeError("returns must be a pandas Series or DataFrame.")

    asset, f, rfree = _align_all(asset, factors, rf)

    y = asset
    if rfree is not None:
        y = y - rfree

    if min_obs is None:
        min_obs = max(30, int(window * 0.5))

    cols = list(f.columns)
    idx = y.index

    betas = pd.DataFrame(index=idx, columns=cols, dtype=float)
    alpha = pd.Series(index=idx, dtype=float, name="alpha") if add_const else None
    r2 = pd.Series(index=idx, dtype=float, name="r2")
    nobs = pd.Series(index=idx, dtype=float, name="nobs")

    w = int(window)

    for end in range(w - 1, len(idx)):
        start = end - w + 1
        loc = idx[start:end + 1]

        y_win = y.loc[loc]
        X_win = f.loc[loc]

        if y_win.isna().any() or X_win.isna().any().any():
            continue

        if len(y_win) < int(min_obs):
            continue

        if add_const:
            X = sm.add_constant(X_win)
        else:
            X = X_win

        try:
            model = sm.OLS(y_win, X).fit()
        except Exception:
            continue

        if add_const:
            alpha.loc[idx[end]] = float(model.params.get("const", np.nan))

        for c in cols:
            betas.loc[idx[end], c] = float(model.params.get(c, np.nan))

        r2.loc[idx[end]] = float(model.rsquared) if hasattr(model, "rsquared") else np.nan
        nobs.loc[idx[end]] = float(model.nobs) if hasattr(model, "nobs") else np.nan

    out = {
        "betas": betas,
        "r2": r2,
        "nobs": nobs,
    }
    if add_const:
        out["alpha"] = alpha

    return out
