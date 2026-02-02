from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..data.validators import validate_timeseries


@dataclass(frozen=True)
class CovarianceEstimate:
    method: str
    cov: pd.DataFrame
    corr: pd.DataFrame
    annualization: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "annualization": float(self.annualization),
            "cov": self.cov,
            "corr": self.corr,
        }


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    df = df.dropna(how="any")
    if len(df) < 3:
        raise ValueError("Not enough rows after dropping NaNs")
    return df.astype(float)


def sample_covariance(returns: pd.DataFrame, *, annualize: int = 252, ddof: int = 1) -> CovarianceEstimate:
    df = _clean_returns(returns)
    cov = df.cov(ddof=int(ddof)) * float(annualize)
    corr = df.corr()
    return CovarianceEstimate(method="sample", cov=cov, corr=corr, annualization=float(annualize))


def ewma_covariance(
    returns: pd.DataFrame,
    *,
    annualize: int = 252,
    lam: float = 0.94,
    min_periods: int = 20,
) -> CovarianceEstimate:
    df = _clean_returns(returns)
    l = float(lam)
    if not (0.0 < l < 1.0):
        raise ValueError("lam must be in (0, 1)")
    if len(df) < int(min_periods):
        raise ValueError("Not enough rows for ewma_covariance")

    x = df.to_numpy(dtype=float, copy=False)
    x = x - x.mean(axis=0, keepdims=True)

    n, k = x.shape
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for i in range(1, n):
        w[i] = w[i - 1] * l
    w = w[::-1]
    w = w / w.sum()

    cov = np.zeros((k, k), dtype=float)
    for i in range(n):
        v = x[i : i + 1].T
        cov += w[i] * (v @ v.T)

    cov_df = pd.DataFrame(cov * float(annualize), index=df.columns, columns=df.columns)
    corr_df = cov_df.copy()
    d = np.sqrt(np.diag(cov_df.to_numpy(dtype=float)))
    denom = np.outer(d, d)
    corr_arr = cov_df.to_numpy(dtype=float) / denom
    corr_df.iloc[:, :] = corr_arr
    return CovarianceEstimate(method="ewma", cov=cov_df, corr=corr_df, annualization=float(annualize))


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualize: int = 252,
    ddof: int = 1,
) -> CovarianceEstimate:
    df = _clean_returns(returns)
    try:
        from sklearn.covariance import LedoitWolf
    except Exception as e:
        raise ImportError("scikit-learn is required for ledoit_wolf_covariance") from e

    x = df.to_numpy(dtype=float, copy=False)
    lw = LedoitWolf().fit(x)
    cov = pd.DataFrame(lw.covariance_ * float(annualize), index=df.columns, columns=df.columns)
    d = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    corr = cov.to_numpy(dtype=float) / np.outer(d, d)
    corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
    return CovarianceEstimate(method="ledoit_wolf", cov=cov, corr=corr_df, annualization=float(annualize))


def covariance_matrix(
    returns: pd.DataFrame,
    *,
    method: str = "sample",
    annualize: int = 252,
    ddof: int = 1,
    lam: float = 0.94,
) -> CovarianceEstimate:
    m = str(method).lower().strip()
    if m in {"sample", "simple"}:
        return sample_covariance(returns, annualize=annualize, ddof=ddof)
    if m in {"ewma", "exp"}:
        return ewma_covariance(returns, annualize=annualize, lam=lam)
    if m in {"ledoit_wolf", "lw", "shrinkage"}:
        return ledoit_wolf_covariance(returns, annualize=annualize, ddof=ddof)
    raise ValueError(f"Unsupported covariance method: {method}")
