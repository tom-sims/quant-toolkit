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
    r = pd.concat([asset_returns.rename("a"), benchmark_returns.rename("b")], axis=1).dropna(how="any")
    if risk_free_returns is not None:
        rf = risk_free_returns.reindex(r.index).astype(float).ffill().bfill()
        y = r["a"].astype(float) - rf
        x = (r["b"].astype(float) - rf).rename("mkt")
    else:
        y = r["a"].astype(float)
        x = r["b"].astype(float).rename("mkt")

    w = int(window)
    mp = int(min_periods) if min_periods is not None else w
    if len(y) < mp:
        return pd.Series(dtype=float, name="beta")

    params = rolling_ols(y.rename("asset_excess"), x, window=w, add_const=True, min_periods=mp)
    if "mkt" in params.columns:
        return params["mkt"].rename("beta")
    if "x" in params.columns:
        return params["x"].rename("beta")
    return pd.Series(dtype=float, name="beta")



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
