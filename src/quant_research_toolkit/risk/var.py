import numpy as np
import pandas as pd


# === VaR INPUTS ===
# returns : Series (single asset / portfolio returns) or DataFrame (multiple assets)
#           values must be simple periodic returns (negative = loss)
#
# alpha : tail probability (e.g. 0.05 for 95% VaR)
#
# method : "historical", "gaussian", "cornish_fisher", "ewma", "monte_carlo"
#
# window : rolling window length for rolling VaR (e.g. 252). If None -> use full sample
#
# periods_per_year : optional (only used in a few helper places if you later extend)
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
#
# v1 = var_historical(aapl, alpha=0.05, window=252)         # rolling historical VaR
# v2 = var_gaussian(aapl, alpha=0.05, window=252)           # rolling parametric VaR
# v3 = var_cornish_fisher(aapl, alpha=0.05, window=252)     # rolling CF VaR
# v4 = var_ewma(aapl, alpha=0.05, lam=0.94)                 # EWMA volatility VaR
# v5 = var_monte_carlo(aapl, alpha=0.05, window=252, n_paths=100000)  # MC VaR (normal sim)
#
# Convention:
# Returned VaR series is POSITIVE loss magnitude.
# Breach rule: return_t < -VaR_t  (loss bigger than VaR)


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


def _z_alpha(alpha):
    a = float(alpha)
    if a <= 0 or a >= 1:
        raise ValueError("alpha must be between 0 and 1 (e.g. 0.05).")
    try:
        from scipy.stats import norm
        return float(norm.ppf(a))
    except Exception:
        # fallback approximation (Acklam-ish rough)
        # good enough for common alphas like 0.01/0.05
        p = a
        if p < 0.5:
            t = np.sqrt(-2.0 * np.log(p))
            return float(-(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                           (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3)))
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        return float((t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                      (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3)))


def _apply_window(series, window):
    if window is None:
        return series, None
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2.")
    return series, w


def var_historical(returns, alpha=0.05, window=252):
    df = _clean_returns(returns)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    a = float(alpha)

    for col in df.columns:
        s = df[col]
        if window is None:
            q = np.quantile(s.values, a)
            out[col] = np.nan
            out.loc[s.index[-1], col] = -float(q)
        else:
            w = int(window)
            out[col] = (-s.rolling(w).quantile(a)).astype(float)

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("VaR_historical")
    return out


def var_gaussian(returns, alpha=0.05, window=252, demean=True, ddof=1):
    df = _clean_returns(returns)

    z = _z_alpha(alpha)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col]
        if window is None:
            mu = float(s.mean()) if demean else 0.0
            sig = float(s.std(ddof=int(ddof)))
            out[col] = np.nan
            out.loc[s.index[-1], col] = -(mu + z * sig)
        else:
            w = int(window)
            mu = s.rolling(w).mean() if demean else 0.0
            sig = s.rolling(w).std(ddof=int(ddof))
            out[col] = -(mu + z * sig)

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("VaR_gaussian")
    return out


def var_cornish_fisher(returns, alpha=0.05, window=252, ddof=1):
    df = _clean_returns(returns)

    z = _z_alpha(alpha)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col]

        if window is None:
            x = s.values.astype(float)
            mu = float(np.mean(x))
            sig = float(np.std(x, ddof=int(ddof)))
            sk = float(pd.Series(x).skew())
            ku = float(pd.Series(x).kurtosis())  # excess kurtosis
            z_cf = (
                z
                + (1.0 / 6.0) * (z * z - 1.0) * sk
                + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * ku
                - (1.0 / 36.0) * (2.0 * z ** 3 - 5.0 * z) * (sk ** 2)
            )
            out[col] = np.nan
            out.loc[s.index[-1], col] = -(mu + z_cf * sig)
        else:
            w = int(window)
            mu = s.rolling(w).mean()
            sig = s.rolling(w).std(ddof=int(ddof))
            sk = s.rolling(w).skew()
            ku = s.rolling(w).kurt()  # excess kurtosis

            z_cf = (
                z
                + (1.0 / 6.0) * (z * z - 1.0) * sk
                + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * ku
                - (1.0 / 36.0) * (2.0 * z ** 3 - 5.0 * z) * (sk ** 2)
            )
            out[col] = -(mu + z_cf * sig)

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("VaR_cornish_fisher")
    return out


def var_ewma(returns, alpha=0.05, lam=0.94, window=None, demean=True):
    df = _clean_returns(returns)

    lam = float(lam)
    if lam <= 0 or lam >= 1:
        raise ValueError("lam must be between 0 and 1 (e.g. 0.94).")

    z = _z_alpha(alpha)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col].copy()

        if demean:
            s = s - s.mean()

        x = s.values.astype(float)
        if len(x) < 2:
            continue

        # init variance
        var = np.var(x, ddof=1)
        vars_ = np.full(len(x), np.nan, dtype=float)

        for t in range(len(x)):
            if t == 0:
                vars_[t] = var
            else:
                var = lam * var + (1.0 - lam) * (x[t - 1] ** 2)
                vars_[t] = var

        sig = np.sqrt(vars_)
        # mean is zero because we demeaned (or assume zero mean)
        out[col] = -(0.0 + z * sig)

        if window is not None:
            w = int(window)
            out[col] = out[col].where(out[col].notna()).rolling(w).last()

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("VaR_ewma")
    return out


def var_monte_carlo(returns, alpha=0.05, window=252, n_paths=100000, seed=None, demean=True, ddof=1):
    df = _clean_returns(returns)

    a = float(alpha)
    n_paths = int(n_paths)
    if n_paths < 1000:
        raise ValueError("n_paths should be at least 1000 (e.g. 100000).")

    if seed is not None:
        np.random.seed(int(seed))

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col]

        if window is None:
            x = s.values.astype(float)
            mu = float(np.mean(x)) if demean else 0.0
            sig = float(np.std(x, ddof=int(ddof)))
            sims = np.random.normal(mu, sig, size=n_paths)
            q = np.quantile(sims, a)
            out[col] = np.nan
            out.loc[s.index[-1], col] = -float(q)
        else:
            w = int(window)
            vals = s.values.astype(float)
            idx = s.index

            for t in range(w, len(vals) + 1):
                x = vals[t - w:t]
                mu = float(np.mean(x)) if demean else 0.0
                sig = float(np.std(x, ddof=int(ddof)))
                sims = np.random.normal(mu, sig, size=n_paths)
                q = np.quantile(sims, a)
                out.loc[idx[t - 1], col] = -float(q)

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("VaR_monte_carlo")
    return out


def var_breach_series(realized_returns, var_series):
    r = _clean_returns(realized_returns)
    v = _clean_returns(var_series)

    if r.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError("var_breach_series expects Series inputs (or single-column DataFrames).")

    r = r.iloc[:, 0]
    v = v.iloc[:, 0]

    idx = r.index.intersection(v.index)
    r = r.loc[idx]
    v = v.loc[idx]

    b = (r < -v).astype(int)
    b.name = "breach"
    return b
