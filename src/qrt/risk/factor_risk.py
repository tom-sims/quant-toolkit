from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.align import align_returns_with_factors
from ..types import VaRResult, ESResult
from ..utils.stats import ols
from .var import historical_var
from .expected_shortfall import historical_es



@dataclass(frozen=True)
class FactorRiskResult:
    model: str
    window: int
    var: Dict[float, float]
    es: Dict[float, float]
    idiosyncratic_vol: float
    factor_vol: Dict[str, float]
    loadings: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "window": int(self.window),
            "var": {float(k): float(v) for k, v in self.var.items()},
            "es": {float(k): float(v) for k, v in self.es.items()},
            "idiosyncratic_vol": float(self.idiosyncratic_vol),
            "factor_vol": dict(self.factor_vol),
            "loadings": dict(self.loadings),
        }


def factor_var_es(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf_col: str = "RF",
    window: int = 252,
    levels: Sequence[float] = (0.95, 0.99),
) -> FactorRiskResult:
    r, x, rf, _ = align_returns_with_factors(asset_returns, factors, rf_col=rf_col, how="inner", dropna=True)
    w = int(window)
    if w <= 10:
        raise ValueError("window must be > 10")
    if len(r) < w:
        raise ValueError("Not enough observations for factor risk window")

    r2 = r.iloc[-w:].astype(float)
    x2 = x.iloc[-w:].astype(float)
    y = r2
    if rf is not None:
        y = y - rf.reindex(r2.index).astype(float)

    res = ols(y.rename("asset_excess"), x2, add_const=True, dep_name="asset_excess")
    params = dict(res.params)
    loadings = {k: float(v) for k, v in params.items() if k != "const"}

    import statsmodels.api as sm

    df = pd.concat([y.rename("y"), x2], axis=1).dropna(how="any")
    y3 = df["y"]
    x3 = sm.add_constant(df.drop(columns=["y"]), has_constant="add")
    fit = sm.OLS(y3, x3).fit()
    resid = pd.Series(fit.resid, index=df.index, name="resid")

    idio_vol = float(np.std(resid.to_numpy(dtype=float), ddof=1))

    factor_vol = {str(c): float(np.std(x2[c].to_numpy(dtype=float), ddof=1)) for c in x2.columns}

    proj = pd.Series(0.0, index=df.index, name="factor_component")
    for k, b in loadings.items():
        if k in df.columns:
            proj = proj + float(b) * df[k].astype(float)

    var_out: Dict[float, float] = {}
    es_out: Dict[float, float] = {}
    for lv in levels:
        v = historical_var(proj, level=float(lv), horizon_days=1, window=None)
        e = historical_es(proj, level=float(lv), horizon_days=1, window=None)
        var_out[float(lv)] = float(v.var)
        es_out[float(lv)] = float(e.es)

    model_name = "ff5" if set(["RMW", "CMA"]).issubset(set(x2.columns)) else "factors"

    return FactorRiskResult(
        model=model_name,
        window=w,
        var=var_out,
        es=es_out,
        idiosyncratic_vol=idio_vol,
        factor_vol=factor_vol,
        loadings=loadings,
    )
