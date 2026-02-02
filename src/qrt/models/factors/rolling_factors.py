from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from ...data.align import align_asset_benchmark_rf, align_returns_with_factors
from ...utils.stats import rolling_ols


def rolling_capm_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_returns: Optional[pd.Series] = None,
    window: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    a, b, rf, _ = align_asset_benchmark_rf(asset_returns, benchmark_returns, risk_free_returns, how="inner", dropna=True)

    y = a.astype(float)
    x = pd.DataFrame({"MKT": b.astype(float)})

    if rf is not None:
        y = y - rf.astype(float)
        x["MKT"] = x["MKT"] - rf.astype(float)

    params = rolling_ols(y.rename("asset_excess"), x, window=int(window), add_const=True, min_periods=min_periods)
    beta = params["MKT"].rename(f"beta_{int(window)}")
    return beta


def rolling_factor_loadings(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf_col: str = "RF",
    window: int = 252,
    factor_cols: Optional[Sequence[str]] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    r, x, rf, _ = align_returns_with_factors(asset_returns, factors, rf_col=rf_col, how="inner", dropna=True)

    cols = tuple(factor_cols) if factor_cols is not None else tuple(x.columns)
    x2 = x[list(cols)].astype(float)

    y = r.astype(float)
    if rf is not None:
        y = y - rf.astype(float)

    params = rolling_ols(y.rename("asset_excess"), x2, window=int(window), add_const=True, min_periods=min_periods)
    return params
