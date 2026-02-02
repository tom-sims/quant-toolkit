from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...data.align import align_asset_benchmark_rf
from ...data.validators import validate_returns
from ...types import RegressionResult
from ...utils.stats import ols


@dataclass(frozen=True)
class CAPMResult:
    alpha_annual: float
    beta: float
    r2: float
    nobs: int
    regression: RegressionResult

    def as_dict(self) -> dict:
        return {
            "alpha_annual": float(self.alpha_annual),
            "beta": float(self.beta),
            "r2": float(self.r2),
            "nobs": int(self.nobs),
            "regression": self.regression.as_dict(),
        }


def capm_regression(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> CAPMResult:
    a, b, rf, _ = align_asset_benchmark_rf(asset_returns, benchmark_returns, risk_free_returns, how="inner", dropna=True)

    y = a.astype(float)
    x = pd.DataFrame({"MKT": b.astype(float)})

    if rf is not None:
        y = y - rf.astype(float)
        x["MKT"] = x["MKT"] - rf.astype(float)

    res = ols(y.rename("asset"), x, add_const=True, dep_name="asset")
    alpha_daily = float(res.params.get("const", np.nan))
    alpha_annual = (1.0 + alpha_daily) ** float(periods_per_year) - 1.0 if np.isfinite(alpha_daily) else np.nan
    beta = float(res.params.get("MKT", np.nan))

    reg = RegressionResult(
        model="CAPM",
        dep="asset",
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

    return CAPMResult(
        alpha_annual=float(alpha_annual),
        beta=float(beta),
        r2=float(res.r2),
        nobs=int(res.nobs),
        regression=reg,
    )
