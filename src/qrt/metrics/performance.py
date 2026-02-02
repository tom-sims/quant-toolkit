from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.validators import validate_returns
from ..utils.math import annualise_return, annualise_vol, cumulative_curve, drawdown_curve, max_drawdown
from ..utils.stats import mean, std, skew, kurtosis


@dataclass(frozen=True)
class PerformanceSummary:
    periods: int
    total_return: float
    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    best_day: float
    worst_day: float
    positive_day_frac: float
    skew: float
    kurtosis: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "periods": float(self.periods),
            "total_return": float(self.total_return),
            "cagr": float(self.cagr),
            "vol": float(self.vol),
            "sharpe": float(self.sharpe),
            "sortino": float(self.sortino),
            "max_drawdown": float(self.max_drawdown),
            "best_day": float(self.best_day),
            "worst_day": float(self.worst_day),
            "positive_day_frac": float(self.positive_day_frac),
            "skew": float(self.skew),
            "kurtosis": float(self.kurtosis),
        }


def total_return(returns: pd.Series) -> float:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    x = r.dropna().to_numpy(dtype=float, copy=False)
    if x.size == 0:
        return np.nan
    return float(np.prod(1.0 + x) - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    x = r.dropna().to_numpy(dtype=float, copy=False)
    if x.size == 0:
        return np.nan
    tr = float(np.prod(1.0 + x) - 1.0)
    return float(annualise_return(tr, int(x.size), int(periods_per_year)))


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252, ddof: int = 1) -> float:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    s = std(r, ddof=ddof)
    if not np.isfinite(s):
        return np.nan
    return float(annualise_vol(float(s), int(periods_per_year)))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    rf = float(risk_free)
    ex = r - rf
    mu = mean(ex)
    sig = std(ex, ddof=ddof)
    if not np.isfinite(mu) or not np.isfinite(sig) or sig == 0.0:
        return np.nan
    return float((mu / sig) * np.sqrt(float(periods_per_year)))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    rf = float(risk_free)
    ex = (r - rf).dropna()
    if len(ex) == 0:
        return np.nan
    downside = ex[ex < 0.0].to_numpy(dtype=float, copy=False)
    if downside.size == 0:
        return np.inf
    dd = float(np.sqrt(np.mean(downside**2)))
    mu = float(np.mean(ex.to_numpy(dtype=float, copy=False)))
    if dd == 0.0:
        return np.nan
    return float((mu / dd) * np.sqrt(float(periods_per_year)))


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    curve = cumulative_curve(r, start=float(start))
    return pd.Series(curve, index=r.index, name="equity")


def drawdown_series(returns: pd.Series, start: float = 1.0) -> pd.Series:
    curve = equity_curve(returns, start=start)
    dd = drawdown_curve(curve)
    return pd.Series(dd, index=curve.index, name="drawdown")


def max_drawdown_value(returns: pd.Series, start: float = 1.0) -> float:
    curve = equity_curve(returns, start=start)
    return float(max_drawdown(curve))


def rolling_volatility(returns: pd.Series, window: int = 252, periods_per_year: int = 252) -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    w = int(window)
    vol = r.rolling(w).std(ddof=1) * np.sqrt(float(periods_per_year))
    return vol.rename(f"roll_vol_{w}")


def rolling_sharpe(returns: pd.Series, window: int = 252, risk_free: float = 0.0, periods_per_year: int = 252) -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    w = int(window)
    ex = r - float(risk_free)
    mu = ex.rolling(w).mean()
    sig = ex.rolling(w).std(ddof=1)
    out = (mu / sig) * np.sqrt(float(periods_per_year))
    return out.rename(f"roll_sharpe_{w}")


def performance_summary(
    returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceSummary:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    x = r.dropna()
    n = int(len(x))
    if n == 0:
        return PerformanceSummary(
            periods=0,
            total_return=np.nan,
            cagr=np.nan,
            vol=np.nan,
            sharpe=np.nan,
            sortino=np.nan,
            max_drawdown=np.nan,
            best_day=np.nan,
            worst_day=np.nan,
            positive_day_frac=np.nan,
            skew=np.nan,
            kurtosis=np.nan,
        )

    tr = float(np.prod(1.0 + x.to_numpy(dtype=float, copy=False)) - 1.0)
    c = float(annualise_return(tr, n, int(periods_per_year)))
    v = annualised_volatility(x, periods_per_year=periods_per_year, ddof=1)
    sh = sharpe_ratio(x, risk_free=risk_free, periods_per_year=periods_per_year, ddof=1)
    so = sortino_ratio(x, risk_free=risk_free, periods_per_year=periods_per_year)
    mdd = max_drawdown_value(x)
    best = float(np.max(x.to_numpy(dtype=float, copy=False)))
    worst = float(np.min(x.to_numpy(dtype=float, copy=False)))
    pos = float((x > 0.0).mean())
    sk = float(skew(x))
    ku = float(kurtosis(x, fisher=False))

    return PerformanceSummary(
        periods=n,
        total_return=tr,
        cagr=c,
        vol=v,
        sharpe=sh,
        sortino=so,
        max_drawdown=mdd,
        best_day=best,
        worst_day=worst,
        positive_day_frac=pos,
        skew=sk,
        kurtosis=ku,
    )
