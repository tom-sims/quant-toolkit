# === RISK PARITY / ERC INPUTS ===
# covariance : DataFrame or ndarray (NxN)
#              covariance matrix (annualised or not, doesn't matter for weights)
#
# target_risk : optional array-like length N
#              target risk contributions (must sum to 1). If None -> equal risk contribution (ERC)
#
# long_only : if True, weights >= 0 (recommended)
#
# weight_bounds : optional tuple (lb, ub)
#              overrides long_only bounds, e.g. (0.0, 0.3)
#
# max_iter, tol : solver settings
#
# How to call:
#
# cov = rets.cov() * 252
# w_erc = risk_parity_weights(cov)
#
# # custom target risk contributions
# target = [0.5, 0.3, 0.2]
# w_trc = risk_parity_weights(cov, target_risk=target)
#
# Returns:
# weights -> pandas Series indexed by asset names
#
# Extra helper:
# risk_contributions(weights, covariance) -> Series

import numpy as np
import pandas as pd

def _as_cov(cov, names):
    if isinstance(cov, pd.DataFrame):
        C = cov.reindex(index=names, columns=names).values.astype(float)
        return C
    C = np.asarray(cov, dtype=float)
    if C.shape[0] != C.shape[1] or C.shape[0] != len(names):
        raise ValueError("covariance must be NxN and match number of assets.")
    return C


def _names_from_cov(cov):
    if isinstance(cov, pd.DataFrame):
        return list(cov.columns)
    n = np.asarray(cov).shape[0]
    return [str(i) for i in range(n)]


def _normalize(w):
    w = np.asarray(w, dtype=float).ravel()
    s = w.sum()
    if s == 0:
        raise ValueError("weights sum to 0.")
    return w / s


def _clip(w, lb, ub):
    w = np.asarray(w, dtype=float).ravel()
    w = np.maximum(w, lb)
    w = np.minimum(w, ub)
    return w


def risk_contributions(weights, covariance):
    if isinstance(weights, pd.Series):
        names = list(weights.index)
        w = weights.values.astype(float)
    else:
        names = _names_from_cov(covariance)
        w = np.asarray(weights, dtype=float).ravel()

    C = _as_cov(covariance, names)
    w = _normalize(w)

    port_var = float(w @ C @ w)
    if port_var <= 0:
        return pd.Series(np.nan, index=names, name="risk_contribution")

    mrc = C @ w
    rc = w * mrc / port_var
    return pd.Series(rc, index=names, name="risk_contribution")


def risk_parity_weights(
    covariance,
    target_risk=None,
    long_only=True,
    weight_bounds=None,
    max_iter=5000,
    tol=1e-10,
    step=0.05,
):
    names = _names_from_cov(covariance)
    C = _as_cov(covariance, names)

    n = len(names)

    if target_risk is None:
        t = np.ones(n, dtype=float) / float(n)
    else:
        t = np.asarray(target_risk, dtype=float).ravel()
        if len(t) != n:
            raise ValueError("target_risk length must match number of assets.")
        if np.any(t < 0):
            raise ValueError("target_risk must be non-negative.")
        if t.sum() == 0:
            raise ValueError("target_risk sum to 0.")
        t = t / t.sum()

    if weight_bounds is not None:
        lb, ub = float(weight_bounds[0]), float(weight_bounds[1])
    else:
        lb, ub = (0.0, 1.0) if long_only else (-1.0, 1.0)

    C = 0.5 * (C + C.T)

    w = np.ones(n, dtype=float) / float(n)
    w = _clip(w, lb, ub)
    w = _normalize(w)

    for _ in range(int(max_iter)):
        port_var = float(w @ C @ w)
        if port_var <= 0:
            break

        mrc = C @ w
        rc = w * mrc / port_var

        # error vs target
        err = rc - t
        if np.max(np.abs(err)) < float(tol):
            break

        # multiplicative update (stable for long-only)
        # w_new = w * (target / current) ^ step
        adj = np.ones(n, dtype=float)
        good = (rc > 0) & np.isfinite(rc)
        adj[good] = (t[good] / rc[good]) ** float(step)

        w_new = w * adj
        w_new = _clip(w_new, lb, ub)

        # If long-only, keep on simplex; otherwise normalize after clipping
        if lb >= 0:
            w_new = np.maximum(w_new, 0.0)
        w_new = _normalize(w_new)

        if np.max(np.abs(w_new - w)) < float(tol):
            w = w_new
            break

        w = w_new

    return pd.Series(w, index=names, name="weights")
