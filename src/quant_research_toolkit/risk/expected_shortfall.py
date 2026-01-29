# === EXPECTED SHORTFALL / CVaR INPUTS ===
# returns : Series (single asset / portfolio returns) or DataFrame (multiple assets)
#           values must be simple periodic returns (negative = loss)
#
# alpha : tail probability (e.g. 0.05 for 95% ES)
#
# method : "historical", "gaussian", "cornish_fisher", "monte_carlo"
#
# window : rolling window length for rolling ES (e.g. 252). If None -> use full sample
#
# Convention:
# Returned ES series is POSITIVE loss magnitude.
# Breach rule (VaR): return_t < -VaR_t
# ES is the average loss given you're in the tail.
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
#
# es1 = es_historical(aapl, alpha=0.05, window=252)
# es2 = es_gaussian(aapl, alpha=0.05, window=252)
# es3 = es_cornish_fisher(aapl, alpha=0.05, window=252)
# es4 = es_monte_carlo(aapl, alpha=0.05, window=252, n_paths=100000)

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


def _z_alpha(alpha):
    a = float(alpha)
    if a <= 0 or a >= 1:
        raise ValueError("alpha must be between 0 and 1 (e.g. 0.05).")
    try:
        from scipy.stats import norm
        return float(norm.ppf(a))
    except Exception:
        # fallback approximation (same as in var.py)
        p = a
        if p < 0.5:
            t = np.sqrt(-2.0 * np.log(p))
            return float(-(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                           (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3)))
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        return float((t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                      (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3)))


def _phi(z):
    try:
        from scipy.stats import norm
        return float(norm.pdf(z))
    except Exception:
        return float((1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z))


def es_historical(returns, alpha=0.05, window=252):
    df = _clean_returns(returns)
    a = float(alpha)

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col].astype(float)

        if window is None:
            q = np.quantile(s.values, a)
            tail = s.values[s.values <= q]
            es = -float(np.mean(tail)) if len(tail) > 0 else np.nan
            out[col] = np.nan
            out.loc[s.index[-1], col] = es
        else:
            w = int(window)
            vals = s.values
            idx = s.index

            for t in range(w, len(vals) + 1):
                x = vals[t - w:t]
                q = np.quantile(x, a)
                tail = x[x <= q]
                es = -float(np.mean(tail)) if len(tail) > 0 else np.nan
                out.loc[idx[t - 1], col] = es

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("ES_historical")
    return out


def es_gaussian(returns, alpha=0.05, window=252, demean=True, ddof=1):
    df = _clean_returns(returns)

    z = _z_alpha(alpha)  # negative for alpha < 0.5
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # For Normal(mu, sigma), ES at level alpha:
    # ES_alpha = -( mu - sigma * phi(z_alpha) / alpha )   (since tail is left tail)
    # If mu=0, ES = sigma * phi(z)/alpha
    phi_z = _phi(z)
    a = float(alpha)

    for col in df.columns:
        s = df[col].astype(float)

        if window is None:
            mu = float(s.mean()) if demean else 0.0
            sig = float(s.std(ddof=int(ddof)))
            es = -(mu - sig * phi_z / a)
            out[col] = np.nan
            out.loc[s.index[-1], col] = float(es)
        else:
            w = int(window)
            mu = s.rolling(w).mean() if demean else 0.0
            sig = s.rolling(w).std(ddof=int(ddof))
            out[col] = -(mu - sig * phi_z / a)

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("ES_gaussian")
    return out


def es_cornish_fisher(returns, alpha=0.05, window=252, ddof=1):
    df = _clean_returns(returns)

    a = float(alpha)
    z = _z_alpha(a)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # Practical CF ES approach:
    # 1) Compute CF adjusted quantile z_cf
    # 2) Use historical tail mean conditional on being below CF VaR threshold
    # This avoids needing a closed form ES for CF (messy).
    for col in df.columns:
        s = df[col].astype(float)

        if window is None:
            x = s.values
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

            var_cf = mu + z_cf * sig
            tail = x[x <= var_cf]
            es = -float(np.mean(tail)) if len(tail) > 0 else np.nan

            out[col] = np.nan
            out.loc[s.index[-1], col] = es
        else:
            w = int(window)
            vals = s.values
            idx = s.index

            for t in range(w, len(vals) + 1):
                x = vals[t - w:t]

                mu = float(np.mean(x))
                sig = float(np.std(x, ddof=int(ddof)))
                sk = float(pd.Series(x).skew())
                ku = float(pd.Series(x).kurtosis())

                z_cf = (
                    z
                    + (1.0 / 6.0) * (z * z - 1.0) * sk
                    + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * ku
                    - (1.0 / 36.0) * (2.0 * z ** 3 - 5.0 * z) * (sk ** 2)
                )

                var_cf = mu + z_cf * sig
                tail = x[x <= var_cf]
                es = -float(np.mean(tail)) if len(tail) > 0 else np.nan

                out.loc[idx[t - 1], col] = es

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("ES_cornish_fisher")
    return out


def es_monte_carlo(returns, alpha=0.05, window=252, n_paths=100000, seed=None, demean=True, ddof=1):
    df = _clean_returns(returns)

    a = float(alpha)
    n_paths = int(n_paths)
    if n_paths < 1000:
        raise ValueError("n_paths should be at least 1000 (e.g. 100000).")

    if seed is not None:
        np.random.seed(int(seed))

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col].astype(float)

        if window is None:
            x = s.values
            mu = float(np.mean(x)) if demean else 0.0
            sig = float(np.std(x, ddof=int(ddof)))
            sims = np.random.normal(mu, sig, size=n_paths)
            q = np.quantile(sims, a)
            tail = sims[sims <= q]
            es = -float(np.mean(tail)) if len(tail) > 0 else np.nan
            out[col] = np.nan
            out.loc[s.index[-1], col] = es
        else:
            w = int(window)
            vals = s.values
            idx = s.index

            for t in range(w, len(vals) + 1):
                x = vals[t - w:t]
                mu = float(np.mean(x)) if demean else 0.0
                sig = float(np.std(x, ddof=int(ddof)))
                sims = np.random.normal(mu, sig, size=n_paths)
                q = np.quantile(sims, a)
                tail = sims[sims <= q]
                es = -float(np.mean(tail)) if len(tail) > 0 else np.nan
                out.loc[idx[t - 1], col] = es

    if isinstance(returns, pd.Series):
        return out.iloc[:, 0].rename("ES_monte_carlo")
    return out
