from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.align import align_returns
from ..data.validators import validate_timeseries
from ..portfolio.covariance import covariance_matrix, CovarianceEstimate
from ..types import RiskContribution
from ..utils.math import normalize_weights, safe_divide


@dataclass(frozen=True)
class PortfolioStats:
    expected_return_annual: float
    volatility_annual: float
    variance_annual: float
    sharpe: float
    weights: Dict[str, float]
    contributions: Tuple[RiskContribution, ...]
    covariance: CovarianceEstimate

    def as_dict(self) -> Dict[str, object]:
        return {
            "expected_return_annual": float(self.expected_return_annual),
            "volatility_annual": float(self.volatility_annual),
            "variance_annual": float(self.variance_annual),
            "sharpe": float(self.sharpe),
            "weights": dict(self.weights),
            "contributions": [c.as_dict() for c in self.contributions],
            "covariance_method": self.covariance.method,
        }


def portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    if df.shape[1] == 0:
        raise ValueError("No columns in returns")
    w = normalize_weights({str(k): float(v) for k, v in weights.items()})
    cols = [c for c in df.columns if str(c) in w]
    if len(cols) == 0:
        raise ValueError("No overlapping tickers between returns and weights")
    df2 = df[cols].dropna(how="any")
    if len(df2) == 0:
        raise ValueError("No rows after dropping NaNs")
    vec = np.array([w[str(c)] for c in cols], dtype=float)
    pr = df2.to_numpy(dtype=float, copy=False) @ vec
    out = pd.Series(pr, index=df2.index, name="portfolio")
    return out


def marginal_contributions(cov_annual: pd.DataFrame, weights: Dict[str, float]) -> Tuple[RiskContribution, ...]:
    w = normalize_weights({str(k): float(v) for k, v in weights.items()})
    cols = [c for c in cov_annual.columns if str(c) in w]
    if len(cols) == 0:
        raise ValueError("No overlapping tickers between covariance and weights")
    cov = cov_annual.loc[cols, cols].to_numpy(dtype=float, copy=False)
    wv = np.array([w[str(c)] for c in cols], dtype=float)
    port_var = float(wv.T @ cov @ wv)
    if not np.isfinite(port_var) or port_var <= 0.0:
        raise ValueError("Invalid portfolio variance")
    port_vol = float(np.sqrt(port_var))
    mcr = (cov @ wv) / port_vol
    ctr = wv * mcr
    out = tuple(
        RiskContribution(ticker=str(cols[i]), weight=float(wv[i]), mcr=float(mcr[i]), contribution=float(ctr[i]))
        for i in range(len(cols))
    )
    return out


def portfolio_stats(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    *,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
    cov_method: str = "sample",
    cov_lam: float = 0.94,
) -> PortfolioStats:
    df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    df = df.dropna(how="any")
    if len(df) < 3:
        raise ValueError("Not enough rows for portfolio stats")

    w = normalize_weights({str(k): float(v) for k, v in weights.items()})
    cols = [c for c in df.columns if str(c) in w]
    if len(cols) == 0:
        raise ValueError("No overlapping tickers between returns and weights")

    df2 = df[cols]
    cov_est = covariance_matrix(df2, method=cov_method, annualize=int(periods_per_year), lam=float(cov_lam))
    cov_annual = cov_est.cov

    mu_daily = df2.mean().to_numpy(dtype=float, copy=False)
    wv = np.array([w[str(c)] for c in cols], dtype=float)

    exp_ret_annual = float(mu_daily @ wv) * float(periods_per_year)
    var_annual = float(wv.T @ cov_annual.to_numpy(dtype=float, copy=False) @ wv)
    vol_annual = float(np.sqrt(var_annual)) if var_annual >= 0 else np.nan
    sharpe = safe_divide(exp_ret_annual - float(risk_free_annual), vol_annual, default=np.nan)

    contr = marginal_contributions(cov_annual, w)

    return PortfolioStats(
        expected_return_annual=float(exp_ret_annual),
        volatility_annual=float(vol_annual),
        variance_annual=float(var_annual),
        sharpe=float(sharpe),
        weights=dict(w),
        contributions=contr,
        covariance=cov_est,
    )
