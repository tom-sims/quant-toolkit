# === PORTFOLIO OPTIMIZATION INPUTS ===
# expected_returns : Series or array-like (length N)
# covariance : DataFrame or ndarray (NxN)
#
# Constraints:
# - long_only=True enforces weights >= 0
# - weight_bounds can override long_only, e.g. (-0.2, 0.2)
# - fully_invested=True enforces sum(weights)=1
#
# How to call:
#
# mu = rets.mean() * 252
# cov = rets.cov() * 252
#
# w_minvar = min_variance_weights(cov)
# w_maxsh = max_sharpe_weights(mu, cov, risk_free=0.03)
# w_mvo = mean_variance_weights(mu, cov, risk_aversion=5.0)
#
# Returns:
# weights -> pandas Series indexed by asset names (if provided)

import numpy as np
import pandas as pd

def _as_series(x, index=None, name=None):
    if isinstance(x, pd.Series):
        s = x.copy()
        if name is not None:
            s.name = name
        return s
    arr = np.asarray(x, dtype=float).ravel()
    if index is None:
        index = [str(i) for i in range(len(arr))]
    return pd.Series(arr, index=index, name=name)


def _as_cov(cov, index):
    if isinstance(cov, pd.DataFrame):
        C = cov.reindex(index=index, columns=index).values.astype(float)
        return C
    C = np.asarray(cov, dtype=float)
    if C.shape[0] != C.shape[1] or C.shape[0] != len(index):
        raise ValueError("covariance must be NxN and match expected_returns length.")
    return C


def _normalize(w):
    w = np.asarray(w, dtype=float).ravel()
    s = w.sum()
    if s == 0:
        raise ValueError("weights sum to 0.")
    return w / s


def _project_bounds(w, lb, ub):
    w = np.asarray(w, dtype=float).ravel()
    w = np.maximum(w, lb)
    w = np.minimum(w, ub)
    return w


def _simplex_projection(w):
    # Projects to weights >=0, sum=1 (Duchi et al.)
    w = np.asarray(w, dtype=float).ravel()
    n = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u - (cssv - 1) / (np.arange(n) + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    return np.maximum(w - theta, 0.0)


def _solve_qp_projected_gradient(Q, c=None, lb=0.0, ub=1.0, fully_invested=True, max_iter=5000, lr=0.05, tol=1e-10):
    # Minimize: 0.5 w'Qw + c'w
    n = Q.shape[0]
    w = np.ones(n) / n

    Q = 0.5 * (Q + Q.T)

    if c is None:
        c = np.zeros(n, dtype=float)
    else:
        c = np.asarray(c, dtype=float).ravel()

    for _ in range(int(max_iter)):
        grad = Q @ w + c
        w_new = w - float(lr) * grad

        # bounds
        w_new = _project_bounds(w_new, lb, ub)

        if fully_invested:
            # if long-only bounds include negatives, simplex won't work; fallback to normalize after clipping
            if lb >= 0:
                w_new = _simplex_projection(w_new)
            else:
                w_new = _normalize(w_new)

        if np.max(np.abs(w_new - w)) < float(tol):
            w = w_new
            break

        w = w_new

    return w


def min_variance_weights(covariance, long_only=True, weight_bounds=None, fully_invested=True):
    if isinstance(covariance, pd.DataFrame):
        names = list(covariance.columns)
    else:
        names = None

    if names is None:
        n = np.asarray(covariance).shape[0]
        names = [str(i) for i in range(n)]

    C = _as_cov(covariance, names)

    if weight_bounds is not None:
        lb, ub = float(weight_bounds[0]), float(weight_bounds[1])
    else:
        lb, ub = (0.0, 1.0) if long_only else (-1.0, 1.0)

    w = _solve_qp_projected_gradient(C, c=None, lb=lb, ub=ub, fully_invested=fully_invested)
    return pd.Series(w, index=names, name="weights")


def mean_variance_weights(expected_returns, covariance, risk_aversion=5.0, long_only=True, weight_bounds=None, fully_invested=True):
    mu = _as_series(expected_returns)
    names = list(mu.index)

    C = _as_cov(covariance, names)
    lam = float(risk_aversion)
    if lam <= 0:
        raise ValueError("risk_aversion must be > 0.")

    # Minimize: 0.5*lam*w' C w - mu'w  -> Q = lam*C, c = -mu
    Q = lam * C
    c = -mu.values

    if weight_bounds is not None:
        lb, ub = float(weight_bounds[0]), float(weight_bounds[1])
    else:
        lb, ub = (0.0, 1.0) if long_only else (-1.0, 1.0)

    w = _solve_qp_projected_gradient(Q, c=c, lb=lb, ub=ub, fully_invested=fully_invested)
    return pd.Series(w, index=names, name="weights")


def max_sharpe_weights(expected_returns, covariance, risk_free=0.0, long_only=True, weight_bounds=None, fully_invested=True):
    mu = _as_series(expected_returns)
    names = list(mu.index)
    C = _as_cov(covariance, names)

    rf = float(risk_free)
    excess = mu.values - rf

    # We approximate max Sharpe by maximizing (w'excess) / sqrt(w' C w)
    # Use iterative scaling: solve min variance with linear term encouraging excess return
    if weight_bounds is not None:
        lb, ub = float(weight_bounds[0]), float(weight_bounds[1])
    else:
        lb, ub = (0.0, 1.0) if long_only else (-1.0, 1.0)

    # Heuristic: run a few passes increasing emphasis on returns
    w = np.ones(len(names)) / len(names)
    for k in range(5):
        gamma = 10.0 ** k
        Q = C
        c = -gamma * excess
        w = _solve_qp_projected_gradient(Q, c=c, lb=lb, ub=ub, fully_invested=fully_invested, max_iter=3000, lr=0.05)
        # if fully invested and long-only, weights already sum to 1

    return pd.Series(w, index=names, name="weights")


def portfolio_stats(weights, expected_returns, covariance, risk_free=0.0):
    w = _as_series(weights)
    mu = _as_series(expected_returns, index=w.index)
    C = _as_cov(covariance, list(w.index))

    ret = float(np.dot(w.values, mu.values))
    vol = float(np.sqrt(w.values @ C @ w.values))
    sharpe = (ret - float(risk_free)) / vol if vol > 0 else np.nan

    return {
        "return": ret,
        "volatility": vol,
        "sharpe": float(sharpe),
    }
