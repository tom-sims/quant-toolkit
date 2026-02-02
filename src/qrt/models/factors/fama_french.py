from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...data.align import align_returns_with_factors
from ...types import RegressionResult
from ...utils.stats import ols


@dataclass(frozen=True)
class FFResult:
    model: str
    alpha_annual: float
    r2: float
    nobs: int
    loadings: dict
    tstats: dict
    regression: RegressionResult

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "alpha_annual": float(self.alpha_annual),
            "r2": float(self.r2),
            "nobs": int(self.nobs),
            "loadings": dict(self.loadings),
            "tstats": dict(self.tstats),
            "regression": self.regression.as_dict(),
        }


def _infer_factor_cols(df: pd.DataFrame) -> Tuple[str, ...]:
    cols = [str(c) for c in df.columns]
    preferred = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    if all(c in cols for c in preferred):
        return tuple(preferred)
    preferred3 = ["MKT_RF", "SMB", "HML"]
    if all(c in cols for c in preferred3):
        return tuple(preferred3)
    return tuple(cols)


def fama_french_regression(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf_col: str = "RF",
    periods_per_year: int = 252,
    factor_cols: Optional[Sequence[str]] = None,
) -> FFResult:
    r, x, rf, _ = align_returns_with_factors(asset_returns, factors, rf_col=rf_col, how="inner", dropna=True)

    cols = tuple(factor_cols) if factor_cols is not None else _infer_factor_cols(x)
    x2 = x[list(cols)].astype(float)

    y = r.astype(float)
    if rf is not None:
        y = y - rf.astype(float)

    res = ols(y.rename("asset_excess"), x2, add_const=True, dep_name="asset_excess")
    alpha_daily = float(res.params.get("const", np.nan))
    alpha_annual = (1.0 + alpha_daily) ** float(periods_per_year) - 1.0 if np.isfinite(alpha_daily) else np.nan

    loadings = {k: float(v) for k, v in res.params.items() if k != "const"}
    tstats = {k: float(v) for k, v in res.tstats.items() if k != "const"}

    reg = RegressionResult(
        model=f"FamaFrench_{len(loadings)}",
        dep="asset_excess",
        indep=tuple(res.indep),
        params=dict(res.params),
        tstats=dict(res.tstats),
        pvalues=dict(res.pvalues),
        r2=float(res.r2),
        adj_r2=float(res.adj_r2),
        nobs=int(res.nobs),
        stderr=dict(res.stderr),
        resid_std=None,
    )

    model_name = "ff5" if set(["RMW", "CMA"]).issubset(set(cols)) else "ff3"

    return FFResult(
        model=model_name,
        alpha_annual=float(alpha_annual),
        r2=float(res.r2),
        nobs=int(res.nobs),
        loadings=loadings,
        tstats=tstats,
        regression=reg,
    )
