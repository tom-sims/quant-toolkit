from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import ReportConfig
from ..data.loaders import load_inputs_for_asset
from ..features.returns import align_asset_benchmark_rf, returns_from_prices
from ..metrics.performance import (
    drawdown_series,
    equity_curve,
    performance_summary,
)
from ..metrics.risk_adjusted import risk_adjusted_summary
from ..models.factors.capm import capm_regression
from ..models.factors.fama_french import fama_french_regression
from ..models.factors.rolling_factors import rolling_capm_beta, rolling_factor_loadings
from ..risk.backtesting import backtest_var
from ..risk.expected_shortfall import es_curve
from ..risk.factor_risk import factor_var_es
from ..risk.stress import stress_summary
from ..risk.var import var_curve
from ..types import FigureRef, ReportMeta, ReportResult, ReportSection, TableRef
from ..viz.distributions import plot_qq, plot_return_histogram
from ..viz.factors import plot_factor_loadings, plot_rolling_loadings
from ..viz.risk import plot_risk_contributions, plot_rolling_series, plot_var_es_curve
from ..viz.timeseries import plot_drawdown_from_returns, plot_equity_from_returns, plot_series


def _slug(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "asset"


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


def _save_series(s: pd.Series, path: Path, name: Optional[str] = None) -> Path:
    df = s.to_frame(name=name or (s.name or "value"))
    return _save_table(df, path)


def _daily_rf(rf: Optional[pd.Series]) -> float:
    if rf is None:
        return 0.0
    x = rf.dropna()
    if len(x) == 0:
        return 0.0
    return float(x.astype(float).mean())


def build_asset_report(
    ticker: str,
    cfg: ReportConfig,
    *,
    start: Optional[Union[str, date]] = None,
    end: Optional[Union[str, date]] = None,
    output_root: Union[str, Path] = "outputs",
    return_kind: str = "log",
    var_levels: Sequence[float] = (0.95, 0.99),
    var_window: int = 252,
    rolling_window: int = 252,
) -> ReportResult:
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("ticker is empty")

    reports_dir, figures_dir, tables_dir = _ensure_dirs(output_root)
    tag = _slug(t)

    inputs = load_inputs_for_asset(t, cfg, start=start, end=end)
    prices: pd.Series = inputs["prices"]
    benchmark_prices: pd.Series = inputs["benchmark_prices"]
    factors: Optional[pd.DataFrame] = inputs.get("factors", None)
    risk_free: Optional[pd.Series] = inputs.get("risk_free", None)

    asset_r, bench_r, rf_r = align_asset_benchmark_rf(
        prices,
        benchmark_prices=benchmark_prices,
        risk_free=risk_free,
        kind=str(return_kind),
    )
    rf_daily = _daily_rf(rf_r)

    perf = performance_summary(asset_r, risk_free=rf_daily, periods_per_year=int(cfg.data.trading_days))
    ra = risk_adjusted_summary(
        asset_r,
        benchmark_returns=bench_r if bench_r is not None else None,
        risk_free=rf_daily,
        periods_per_year=int(cfg.data.trading_days),
    )

    eq = equity_curve(asset_r)
    dd = drawdown_series(asset_r)

    fig_prices_path = figures_dir / f"{tag}_prices.png"
    fig_eq_path = figures_dir / f"{tag}_equity.png"
    fig_dd_path = figures_dir / f"{tag}_drawdown.png"
    fig_hist_path = figures_dir / f"{tag}_returns_hist.png"
    fig_qq_path = figures_dir / f"{tag}_qq.png"

    _, _, p1 = plot_series(prices.rename(t), title=f"{t} Price", ylabel="Price", outpath=fig_prices_path)
    _, _, p2 = plot_equity_from_returns(asset_r.rename(t), title=f"{t} Equity Curve", outpath=fig_eq_path)
    _, _, p3 = plot_drawdown_from_returns(asset_r.rename(t), title=f"{t} Drawdown", outpath=fig_dd_path)
    _, _, p4 = plot_return_histogram(asset_r.rename(t), title=f"{t} Returns Histogram", outpath=fig_hist_path)
    _, _, p5 = plot_qq(asset_r.rename(t), dist="normal", title=f"{t} Q-Q (Normal)", outpath=fig_qq_path)

    figures_core = tuple(
        FigureRef(title=title, path=path, kind="png")
        for title, path in [
            ("Price", p1),
            ("Equity Curve", p2),
            ("Drawdown", p3),
            ("Returns Histogram", p4),
            ("Q-Q Plot", p5),
        ]
        if path is not None
    )

    levels = tuple(float(x) for x in var_levels)
    var_s = var_curve(asset_r, levels=levels, horizon_days=1, method="historical", window=None)
    es_s = es_curve(asset_r, levels=levels, horizon_days=1, method="historical", window=None)

    var_es_table = pd.DataFrame({"VaR": var_s, "ES": es_s})
    var_es_table_path = tables_dir / f"{tag}_var_es.csv"
    _save_table(var_es_table, var_es_table_path)

    fig_var_es_path = figures_dir / f"{tag}_var_es_curve.png"
    _, _, p6 = plot_var_es_curve(
        levels=levels,
        var_vals=[float(var_s.loc[l]) for l in levels],
        es_vals=[float(es_s.loc[l]) for l in levels],
        title=f"{t} VaR / ES",
        outpath=fig_var_es_path,
    )
    fig_var_es = FigureRef(title="VaR / ES Curve", path=p6, kind="png") if p6 is not None else None

    bt_res, bt_forecast, bt_viol = backtest_var(
        asset_r,
        level=float(levels[0]),
        window=int(var_window),
        horizon_days=1,
        method="historical",
        dist="normal",
        test="kupiec_uc",
    )

    bt_table = pd.DataFrame(
        {
            "VaR": bt_forecast,
            "return": asset_r.reindex(bt_forecast.index),
            "violation": bt_viol,
        }
    ).dropna(how="any")

    bt_table_path = tables_dir / f"{tag}_var_backtest.csv"
    _save_table(bt_table, bt_table_path)

    stress_df = stress_summary(asset_r)
    stress_path = tables_dir / f"{tag}_stress.csv"
    _save_table(stress_df, stress_path)

    sections: List[ReportSection] = []

    sections.append(
        ReportSection(
            title="Overview",
            text="",
            metrics={
                "ticker": t,
                "start": str(asset_r.index.min().date()) if len(asset_r) else None,
                "end": str(asset_r.index.max().date()) if len(asset_r) else None,
                "observations": int(len(asset_r)),
                "benchmark": str(benchmark_prices.name or cfg.data.benchmark_ticker),
            },
            figures=figures_core,
            tables=tuple(),
            extras={},
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
            title="Risk Adjusted",
            text="",
            metrics=ra.as_dict(),
            figures=tuple(),
            tables=tuple(),
            extras={},
        )
    )

    risk_figs: List[FigureRef] = []
    if fig_var_es is not None:
        risk_figs.append(fig_var_es)

    sections.append(
        ReportSection(
            title="VaR and Expected Shortfall",
            text="",
            metrics={
                "levels": list(levels),
            },
            figures=tuple(risk_figs),
            tables=(
                TableRef(title="VaR / ES Table", path=var_es_table_path, format="csv"),
                TableRef(title="VaR Backtest Table", path=bt_table_path, format="csv"),
                TableRef(title="Stress Scenarios", path=stress_path, format="csv"),
            ),
            extras={"backtest": (bt_res.as_dict() if hasattr(bt_res, "as_dict") else dict(bt_res))},

        )
    )

    if bench_r is not None:
        capm = capm_regression(asset_r, bench_r, risk_free_returns=rf_r, periods_per_year=int(cfg.data.trading_days))
        capm_table = pd.DataFrame(
            {
                "param": list(capm.regression.params.keys()),
                "value": list(capm.regression.params.values()),
                "tstat": [capm.regression.tstats.get(k, np.nan) for k in capm.regression.params.keys()],
                "pvalue": [capm.regression.pvalues.get(k, np.nan) for k in capm.regression.params.keys()],
            }
        )
        capm_path = tables_dir / f"{tag}_capm.csv"
        _save_table(capm_table, capm_path)

        roll_beta = rolling_capm_beta(asset_r, bench_r, risk_free_returns=rf_r, window=int(rolling_window))
        fig_beta_path = figures_dir / f"{tag}_rolling_beta.png"
        _, _, p7 = plot_rolling_series(
            roll_beta.rename("beta"),
            title=f"{t} Rolling CAPM Beta",
            ylabel="Beta",
            outpath=fig_beta_path,
        )

        capm_figs: List[FigureRef] = []
        if p7 is not None:
            capm_figs.append(FigureRef(title="Rolling Beta", path=p7, kind="png"))

        sections.append(
            ReportSection(
                title="CAPM",
                text="",
                metrics={
                    "alpha_annual": float(capm.alpha_annual),
                    "beta": float(capm.beta),
                    "r2": float(capm.r2),
                    "nobs": int(capm.nobs),
                },
                figures=tuple(capm_figs),
                tables=(TableRef(title="CAPM Regression", path=capm_path, format="csv"),),
                extras={"regression": capm.regression.as_dict()},
            )
        )

    if factors is not None and isinstance(factors, pd.DataFrame) and len(factors) > 0:
        ff = fama_french_regression(asset_r, factors, rf_col="RF", periods_per_year=int(cfg.data.trading_days))
        ff_path = tables_dir / f"{tag}_{ff.model}_regression.csv"
        ff_table = pd.DataFrame(
            {
                "param": list(ff.regression.params.keys()),
                "value": list(ff.regression.params.values()),
                "tstat": [ff.regression.tstats.get(k, np.nan) for k in ff.regression.params.keys()],
                "pvalue": [ff.regression.pvalues.get(k, np.nan) for k in ff.regression.params.keys()],
            }
        )
        _save_table(ff_table, ff_path)

        fig_load_path = figures_dir / f"{tag}_{ff.model}_loadings.png"
        _, _, p8 = plot_factor_loadings(ff.loadings, title=f"{t} {ff.model.upper()} Loadings", outpath=fig_load_path)

        rolling_params = rolling_factor_loadings(
            asset_r,
            factors,
            rf_col="RF",
            window=int(rolling_window),
            factor_cols=None,
            min_periods=None,
        )
        fig_roll_path = figures_dir / f"{tag}_{ff.model}_rolling_loadings.png"
        _, _, p9 = plot_rolling_loadings(rolling_params, title=f"{t} Rolling Loadings", outpath=fig_roll_path)

        fr = factor_var_es(asset_r, factors, rf_col="RF", window=int(var_window), levels=levels)
        fr_path = tables_dir / f"{tag}_factor_risk.csv"
        fr_df = pd.DataFrame(
            {
                "metric": ["idiosyncratic_vol"] + [f"var_{l:g}" for l in levels] + [f"es_{l:g}" for l in levels],
                "value": [fr.idiosyncratic_vol]
                + [fr.var[float(l)] for l in levels]
                + [fr.es[float(l)] for l in levels],
            }
        )
        _save_table(fr_df, fr_path)

        ff_figs: List[FigureRef] = []
        if p8 is not None:
            ff_figs.append(FigureRef(title="Factor Loadings", path=p8, kind="png"))
        if p9 is not None:
            ff_figs.append(FigureRef(title="Rolling Loadings", path=p9, kind="png"))

        sections.append(
            ReportSection(
                title="Factor Model",
                text="",
                metrics={
                    "model": ff.model,
                    "alpha_annual": float(ff.alpha_annual),
                    "r2": float(ff.r2),
                    "nobs": int(ff.nobs),
                    "idiosyncratic_vol": float(fr.idiosyncratic_vol),
                },
                figures=tuple(ff_figs),
                tables=(
                    TableRef(title="Factor Regression", path=ff_path, format="csv"),
                    TableRef(title="Factor Risk", path=fr_path, format="csv"),
                ),
                extras={
                    "loadings": dict(ff.loadings),
                    "tstats": dict(ff.tstats),
                    "regression": ff.regression.as_dict(),
                    "factor_risk": fr.as_dict(),
                },
            )
        )

    meta = ReportMeta(
        report_type="asset",
        subject=t,
        as_of=asset_r.index.max().date() if len(asset_r) else None,
        sample=None,
        benchmark=str(benchmark_prices.name or cfg.data.benchmark_ticker),
        risk_free_desc="series" if risk_free is not None else "constant",
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
