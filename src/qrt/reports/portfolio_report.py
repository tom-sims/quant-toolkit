from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import ReportConfig
from ..data.loaders import load_prices_many, load_risk_free, load_fama_french_factors
from ..data.validators import validate_timeseries
from ..features.returns import returns_from_prices
from ..analytics.portfolio import portfolio_returns, portfolio_stats
from ..metrics.performance import performance_summary, equity_curve, drawdown_series
from ..risk.var import var_curve
from ..risk.expected_shortfall import es_curve
from ..risk.stress import stress_summary
from ..types import FigureRef, ReportMeta, ReportResult, ReportSection, TableRef
from ..viz.timeseries import plot_series, plot_equity_from_returns, plot_drawdown_from_returns
from ..viz.distributions import plot_return_histogram
from ..viz.risk import plot_var_es_curve, plot_risk_contributions


def _slug(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "portfolio"


def _ensure_dirs(root: Union[str, Path]) -> Tuple[Path, Path, Path]:
    base = root if isinstance(root, Path) else Path(str(root))
    reports_dir = base / "reports"
    figures_dir = base / "figures"
    tables_dir = base / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, figures_dir, tables_dir


def _save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def build_portfolio_report(
    weights: Dict[str, float],
    cfg: ReportConfig,
    *,
    start: Optional[Union[str, date]] = None,
    end: Optional[Union[str, date]] = None,
    output_root: Union[str, Path] = "outputs",
    return_kind: str = "log",
    var_levels: Sequence[float] = (0.95, 0.99),
    cov_method: str = "sample",
    cov_lam: float = 0.94,
) -> ReportResult:
    reports_dir, figures_dir, tables_dir = _ensure_dirs(output_root)

    tickers = [str(k).upper().strip() for k in weights.keys() if str(k).strip()]
    if len(tickers) == 0:
        raise ValueError("No tickers in weights")

    s = start if start is not None else cfg.data.start
    e = end if end is not None else cfg.data.end

    prices = load_prices_many(tickers, start=s, end=e, field=cfg.data.price_field, cache=cfg.data.cache)
    returns = prices.apply(lambda col: returns_from_prices(col, kind=str(return_kind)))

    returns_df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    returns_df = returns_df.dropna(how="any")
    if len(returns_df) == 0:
        raise ValueError("No returns after alignment")

    rf = load_risk_free(
        start=s,
        end=e,
        annual_rate=cfg.data.risk_free_annual,
        trading_days=int(cfg.data.trading_days),
        fred_series=None,
        cache=cfg.data.cache,
    )
    rf = rf.reindex(returns_df.index).ffill().bfill()
    rf_daily = float(rf.mean()) if len(rf.dropna()) else 0.0

    port_r = portfolio_returns(returns_df, weights).rename("portfolio")
    perf = performance_summary(port_r, risk_free=rf_daily, periods_per_year=int(cfg.data.trading_days))

    stats = portfolio_stats(
        returns_df,
        weights,
        risk_free_annual=float(cfg.data.risk_free_annual),
        periods_per_year=int(cfg.data.trading_days),
        cov_method=str(cov_method),
        cov_lam=float(cov_lam),
    )

    contr_df = pd.DataFrame([c.as_dict() for c in stats.contributions]).set_index("ticker")
    contr_path = tables_dir / "portfolio_risk_contributions.csv"
    _save_table(contr_df, contr_path)

    levels = tuple(float(x) for x in var_levels)
    var_s = var_curve(port_r, levels=levels, horizon_days=1, method="historical", window=None)
    es_s = es_curve(port_r, levels=levels, horizon_days=1, method="historical", window=None)
    var_es_df = pd.DataFrame({"VaR": var_s, "ES": es_s})
    var_es_path = tables_dir / "portfolio_var_es.csv"
    _save_table(var_es_df, var_es_path)

    stress_df = stress_summary(port_r)
    stress_path = tables_dir / "portfolio_stress.csv"
    _save_table(stress_df, stress_path)

    fig_price_path = figures_dir / "portfolio_prices.png"
    fig_eq_path = figures_dir / "portfolio_equity.png"
    fig_dd_path = figures_dir / "portfolio_drawdown.png"
    fig_hist_path = figures_dir / "portfolio_returns_hist.png"
    fig_var_es_path = figures_dir / "portfolio_var_es_curve.png"
    fig_contrib_path = figures_dir / "portfolio_risk_contrib.png"

    _, _, p1 = plot_series(prices, title="Portfolio Constituents Prices", ylabel="Price", outpath=fig_price_path)
    _, _, p2 = plot_equity_from_returns(port_r, title="Portfolio Equity Curve", outpath=fig_eq_path)
    _, _, p3 = plot_drawdown_from_returns(port_r, title="Portfolio Drawdown", outpath=fig_dd_path)
    _, _, p4 = plot_return_histogram(port_r, title="Portfolio Returns Histogram", outpath=fig_hist_path)
    _, _, p5 = plot_var_es_curve(
        levels=levels,
        var_vals=[float(var_s.loc[l]) for l in levels],
        es_vals=[float(es_s.loc[l]) for l in levels],
        title="Portfolio VaR / ES",
        outpath=fig_var_es_path,
    )
    _, _, p6 = plot_risk_contributions(contr_df, title="Portfolio Risk Contributions", outpath=fig_contrib_path)

    core_figs = tuple(
        FigureRef(title=title, path=path, kind="png")
        for title, path in [
            ("Constituent Prices", p1),
            ("Equity Curve", p2),
            ("Drawdown", p3),
            ("Returns Histogram", p4),
            ("VaR / ES Curve", p5),
            ("Risk Contributions", p6),
        ]
        if path is not None
    )

    sections: List[ReportSection] = []

    sections.append(
        ReportSection(
            title="Overview",
            text="",
            metrics={
                "start": str(returns_df.index.min().date()) if len(returns_df) else None,
                "end": str(returns_df.index.max().date()) if len(returns_df) else None,
                "observations": int(len(port_r)),
                "num_assets": int(len(tickers)),
                "covariance_method": str(cov_method),
            },
            figures=core_figs,
            tables=(
                TableRef(title="Risk Contributions", path=contr_path, format="csv"),
                TableRef(title="VaR / ES Table", path=var_es_path, format="csv"),
                TableRef(title="Stress Scenarios", path=stress_path, format="csv"),
            ),
            extras={"weights": dict(weights)},
        )
    )

    sections.append(
        ReportSection(
            title="Performance",
            text="",
            metrics=perf.as_dict(),
            figures=tuple(),
            tables=tuple(),
            extras={},
        )
    )

    sections.append(
        ReportSection(
            title="Risk and Allocation",
            text="",
            metrics=stats.as_dict(),
            figures=tuple(),
            tables=tuple(),
            extras={
                "covariance": {
                    "method": stats.covariance.method,
                    "annualization": float(stats.covariance.annualization),
                    "tickers": list(stats.covariance.cov.columns.astype(str)),
                }
            },
        )
    )

    meta = ReportMeta(
        report_type="portfolio",
        subject="portfolio",
        as_of=port_r.index.max().date() if len(port_r) else None,
        sample=None,
        benchmark=str(cfg.data.benchmark_ticker),
        risk_free_desc="series" if rf is not None else "constant",
        return_type=str(return_kind),
        frequency="daily",
    )

    fig_paths = tuple((f.path for s in sections for f in s.figures))
    table_paths = tuple((tb.path for s in sections for tb in s.tables))

    return ReportResult(
        meta=meta,
        sections=tuple(sections),
        output_path=None,
        figure_paths=fig_paths,
        table_paths=table_paths,
        artifacts={},
    )
