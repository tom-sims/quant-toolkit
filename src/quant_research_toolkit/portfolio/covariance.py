# === COVARIANCE INPUTS ===
# returns : DataFrame (required)
#           values must be simple periodic returns, e.g. prices.pct_change().dropna()
#
# periods_per_year : optional
#           if provided, annualises cov by * periods_per_year
#
# How to call:
#
# rets = prices[["AAPL","MSFT","GOOG"]].pct_change().dropna()
#
# cov_s = sample_covariance(rets, periods_per_year=252)
# cov_e = ewma_covariance(rets, lam=0.94, periods_per_year=252)
# cov_lw = ledoit_wolf_covariance(rets, periods_per_year=252)
# corr = covariance_to_correlation(cov_s)

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf, OAS
except Exception:
    LedoitWolf = None
    OAS = None

def _to_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or "asset")
    raise TypeError("returns must be a pandas DataFrame (or Series).")


def _clean_returns(returns):
    df = _to_df(returns).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if df.shape[0] < 2:
        raise ValueError("Not enough data to compute covariance.")
    return df


def _annualize_cov(cov, periods_per_year):
    if periods_per_year is None:
        return cov
    return cov * float(periods_per_year)


def sample_covariance(returns, periods_per_year=None, ddof=1):
    df = _clean_returns(returns)
    cov = df.cov(ddof=int(ddof))
    return _annualize_cov(cov, periods_per_year)


def correlation_matrix(returns):
    df = _clean_returns(returns)
    return df.corr()


def covariance_to_correlation(cov):
    if not isinstance(cov, (pd.DataFrame, np.ndarray)):
        raise TypeError("cov must be a pandas DataFrame or numpy array.")

    if isinstance(cov, pd.DataFrame):
        C = cov.values.astype(float)
        cols = cov.columns
        idx = cov.index
    else:
        C = np.asarray(cov, dtype=float)
        cols = None
        idx = None

    d = np.sqrt(np.maximum(np.diag(C), 0.0))
    d[d == 0] = np.nan
    corr = C / np.outer(d, d)

    if cols is not None:
        return pd.DataFrame(corr, index=idx, columns=cols)
    return corr


def correlation_to_covariance(corr, vol):
    if isinstance(corr, pd.DataFrame):
        R = corr.values.astype(float)
        cols = corr.columns
        idx = corr.index
    else:
        R = np.asarray(corr, dtype=float)
        cols = None
        idx = None

    v = np.asarray(vol, dtype=float).ravel()
    if R.shape[0] != R.shape[1] or R.shape[0] != len(v):
        raise ValueError("Dimensions do not match: corr must be NxN and vol must be length N.")

    cov = R * np.outer(v, v)

    if cols is not None:
        return pd.DataFrame(cov, index=idx, columns=cols)
    return cov


def ewma_covariance(returns, lam=0.94, periods_per_year=None, demean=True, init_cov=None):
    df = _clean_returns(returns)

    lam = float(lam)
    if lam <= 0 or lam >= 1:
        raise ValueError("lam must be between 0 and 1 (e.g. 0.94).")

    X = df.values.astype(float)
    if demean:
        X = X - X.mean(axis=0, keepdims=True)

    n = X.shape[1]

    if init_cov is None:
        S = np.cov(X, rowvar=False, ddof=1)
    else:
        if isinstance(init_cov, pd.DataFrame):
            S = init_cov.reindex(index=df.columns, columns=df.columns).values.astype(float)
        else:
            S = np.asarray(init_cov, dtype=float)
        if S.shape != (n, n):
            raise ValueError("init_cov must be NxN with N = number of columns in returns.")

    for t in range(X.shape[0]):
        x = X[t].reshape(-1, 1)
        S = lam * S + (1.0 - lam) * (x @ x.T)

    cov = pd.DataFrame(S, index=df.columns, columns=df.columns)
    cov = _annualize_cov(cov, periods_per_year)
    return cov


def _identity_shrinkage_cov(df, delta=None, periods_per_year=None, ddof=1):
    X = df.values.astype(float)
    X = X - X.mean(axis=0, keepdims=True)

    S = np.cov(X, rowvar=False, ddof=int(ddof))
    n = S.shape[0]

    mu = np.trace(S) / float(n)
    F = mu * np.eye(n)

    if delta is None:
        # simple heuristic if user didn't give a shrink intensity
        # (not full Ledoit-Wolf, but stable and ok for a starter toolkit)
        # more samples -> less shrink
        T = X.shape[0]
        delta = min(0.9, max(0.0, n / float(max(T, 1)) * 0.5))

    delta = float(delta)
    if delta < 0 or delta > 1:
        raise ValueError("delta must be between 0 and 1.")

    C = delta * F + (1.0 - delta) * S
    cov = pd.DataFrame(C, index=df.columns, columns=df.columns)
    cov = _annualize_cov(cov, periods_per_year)
    return cov


def ledoit_wolf_covariance(returns, periods_per_year=None):
    df = _clean_returns(returns)

    if LedoitWolf is None:
        return _identity_shrinkage_cov(df, delta=None, periods_per_year=periods_per_year)

    X = df.values.astype(float)
    lw = LedoitWolf().fit(X)
    cov = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
    cov = _annualize_cov(cov, periods_per_year)
    return cov


def oas_covariance(returns, periods_per_year=None):
    df = _clean_returns(returns)

    if OAS is None:
        return _identity_shrinkage_cov(df, delta=0.5, periods_per_year=periods_per_year)

    X = df.values.astype(float)
    oas = OAS().fit(X)
    cov = pd.DataFrame(oas.covariance_, index=df.columns, columns=df.columns)
    cov = _annualize_cov(cov, periods_per_year)
    return cov


def ensure_psd(cov, eps=1e-10):
    if isinstance(cov, pd.DataFrame):
        C = cov.values.astype(float)
        cols = cov.columns
        idx = cov.index
    else:
        C = np.asarray(cov, dtype=float)
        cols = None
        idx = None

    C = 0.5 * (C + C.T)

    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, float(eps))
    C_psd = (vecs * vals) @ vecs.T

    if cols is not None:
        return pd.DataFrame(C_psd, index=idx, columns=cols)
    return C_psd
