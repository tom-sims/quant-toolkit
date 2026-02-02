from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.validators import align_series, validate_returns
from ..utils.math import cumulative_curve, drawdown_curve, max_drawdown, safe_divide
from ..utils.stats import mean, std


@dataclass(frozen=True)
class RiskAdjustedSummary:
    treynor: float
    information_ratio: float
    calmar: float
    tracking_error: float
    beta: float
    alpha_annual: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "treynor": float(self.treynor),
            "information_ratio": float(self.information_ratio),
            "calmar": float(self.calmar),
            "tracking_error": float(self.tracking_error),
            "beta": float(self.beta),
            "alpha_annual": float(self.alpha_annual),
        }


def beta_alpha(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    a = asset_returns.rename("asset")
    b = benchmark_returns.rename("benchmark")
    df = pd.concat([a, b], axis=1).dropna(how="any")
    if len(df) < 3:
        return float("nan"), float("nan")

    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(df.index).astype(float).ffill().bfill()
        ex_a = df["asset"].astype(float) - rf
        ex_b = df["benchmark"].astype(float) - rf
    else:
        rf = float(risk_free)
        ex_a = df["asset"].astype(float) - rf
        ex_b = df["benchmark"].astype(float) - rf

    var_b = float(np.var(ex_b.to_numpy(dtype=float), ddof=1))
    if not np.isfinite(var_b) or var_b <= 0.0:
        return float("nan"), float("nan")

    cov_ab = float(np.cov(ex_a.to_numpy(dtype=float), ex_b.to_numpy(dtype=float), ddof=1)[0, 1])
    beta = cov_ab / var_b

    mu_a = float(np.mean(ex_a.to_numpy(dtype=float)))
    mu_b = float(np.mean(ex_b.to_numpy(dtype=float)))
    alpha_daily = mu_a - beta * mu_b
    alpha_annual = alpha_daily * float(periods_per_year)

    return float(beta), float(alpha_annual)



def treynor_ratio(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    b, _ = beta_alpha(asset_returns, benchmark_returns, risk_free=risk_free, periods_per_year=periods_per_year)
    if not np.isfinite(b) or b == 0.0:
        return np.nan
    r, _ = validate_returns(asset_returns, name=asset_returns.name or "asset")
    ex = r.astype(float) - float(risk_free)
    mu = mean(ex)
    if not np.isfinite(mu):
        return np.nan
    ann_ex = float(mu) * float(periods_per_year)
    return float(ann_ex / b)


def tracking_error(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    a = asset_returns.rename("asset")
    b = benchmark_returns.rename("bench")
    df = pd.concat([a, b], axis=1).dropna(how="any")
    if len(df) < 3:
        return float("nan")
    diff = df["asset"].astype(float) - df["bench"].astype(float)
    v = float(np.std(diff.to_numpy(dtype=float), ddof=int(ddof)))
    return float(v) * float(np.sqrt(float(periods_per_year)))


def information_ratio(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    a = asset_returns.rename("asset")
    b = benchmark_returns.rename("bench")
    df = pd.concat([a, b], axis=1).dropna(how="any")
    if len(df) < 3:
        return float("nan")
    active = df["asset"].astype(float) - df["bench"].astype(float)
    mu = float(np.mean(active.to_numpy(dtype=float)))
    te = float(np.std(active.to_numpy(dtype=float), ddof=int(ddof)))
    if not np.isfinite(te) or te <= 0.0:
        return float("nan")
    return float(mu / te) * float(np.sqrt(float(periods_per_year)))


def calmar_ratio(
    asset_returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    r, _ = validate_returns(asset_returns, name=asset_returns.name or "asset")
    x = r.dropna().to_numpy(dtype=float, copy=False)
    if x.size == 0:
        return np.nan
    total = float(np.prod(1.0 + x) - 1.0)
    years = float(x.size) / float(periods_per_year)
    if years <= 0:
        return np.nan
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if (1.0 + total) > 0 else np.nan
    curve = cumulative_curve(r, start=1.0)
    mdd = float(np.min(drawdown_curve(curve))) if curve.size else np.nan
    denom = abs(mdd) if np.isfinite(mdd) else np.nan
    if not np.isfinite(denom) or denom == 0.0:
        return np.nan
    return float(ann / denom)


def risk_adjusted_summary(
    asset_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> RiskAdjustedSummary:
    if benchmark_returns is not None:
        b, a = beta_alpha(asset_returns, benchmark_returns, risk_free=risk_free, periods_per_year=periods_per_year)
        tr = treynor_ratio(asset_returns, benchmark_returns, risk_free=risk_free, periods_per_year=periods_per_year)
        te = tracking_error(asset_returns, benchmark_returns, periods_per_year=periods_per_year, ddof=1)
        ir = information_ratio(asset_returns, benchmark_returns, periods_per_year=periods_per_year, ddof=1)
    else:
        b, a, tr, te, ir = np.nan, np.nan, np.nan, np.nan, np.nan

    cal = calmar_ratio(asset_returns, periods_per_year=periods_per_year)

    return RiskAdjustedSummary(
        treynor=float(tr),
        information_ratio=float(ir),
        calmar=float(cal),
        tracking_error=float(te),
        beta=float(b),
        alpha_annual=float(a),
    )
