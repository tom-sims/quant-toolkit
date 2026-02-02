from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..data.validators import validate_returns


@dataclass(frozen=True)
class StressScenario:
    name: str
    shock: float

    def as_dict(self) -> Dict[str, float]:
        return {"shock": float(self.shock)}


def default_scenarios() -> Tuple[StressScenario, ...]:
    return (
        StressScenario(name="market_-5pct", shock=-0.05),
        StressScenario(name="marketnarket_-10pct", shock=-0.10),
        StressScenario(name="market_-20pct", shock=-0.20),
        StressScenario(name="market_+5pct", shock=0.05),
    )


def apply_shock(returns: pd.Series, shock: float) -> pd.Series:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    out = r.copy().astype(float)
    if len(out) == 0:
        return out
    out.iloc[-1] = float(shock)
    return out


def stress_summary(
    returns: pd.Series,
    *,
    scenarios: Optional[Sequence[StressScenario]] = None,
) -> pd.DataFrame:
    r, _ = validate_returns(returns, name=returns.name or "returns")
    sc = tuple(scenarios) if scenarios is not None else default_scenarios()
    if len(sc) == 0:
        raise ValueError("No scenarios provided")
    rows: List[Dict[str, float]] = []
    last = float(r.dropna().iloc[-1]) if len(r.dropna()) else np.nan
    for s in sc:
        shocked = float(s.shock)
        delta = shocked - last if np.isfinite(last) else np.nan
        rows.append({"scenario": s.name, "shock": shocked, "delta_vs_last": float(delta) if np.isfinite(delta) else np.nan})
    df = pd.DataFrame(rows).set_index("scenario")
    return df
