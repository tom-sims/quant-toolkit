from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import ReportConfig
from ..data.loaders import load_prices_many, load_risk_free, load_fama_french_factors
from ..data.validators import validate_timeseries
from ..features.returns import returns_from_prices
from ..metrics.performance import performance_summary
from ..models.factors.capm import capm_regression
from ..models.factors.fama_french import fama_french_regression
from ..models.factors.rolling_factors import rolling_capm_beta
from ..portfolio.construction import add_asset
from ..analytics.portfolio import portfolio_returns, portfolio_stats
from ..risk.expected_shortfall import es_curve
from ..risk.var import var_curve
from ..types import FigureRef, ReportMeta, ReportResult, ReportSection, TableRef
from ..viz.risk import plot_risk_contributions, plot_rolling_series, plot_var_es_curve
from ..viz.timeseries import plot_equity_from_returns, plot_series


def _slug(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "portfolio_fit"


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


def _series_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(how="any")
    if len(df) < 3:
        return np.nan
    return float(df["a"].corr(df["b"]))


def build_portfolio_fit_report(
    ticker: str,
    weights: Dict[str, float],
    cfg: ReportConfig,
    *,
    candidate_weight: float = 0.05,
    start: Optional[Union[str, date]] = None,
    end: Optional[Union[str, date]] = None,
    output_root: Union[str, Path] = "outputs",
    return_kind: str = "log",
    var_levels: Sequence[float] = (0.95, 0.99),
    cov_method: str = "sample",
    cov_lam: float = 0.94,
    rolling_window: int = 252,
) -> ReportResult:
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("ticker is empty")

    s = start if start is not None else cfg.data.start
    e = end if end is not None else cfg.data.end

    reports_dir, figures_dir, tables_dir = _ensure_dirs(output_root)
    tag = _slug(f"fit_{t}")

    base_tickers = [str(k).upper().strip() for k in weights.keys() if str(k).strip()]
    if len(base_tickers) == 0:
        raise ValueError("No tickers in weights")

    all_tickers = sorted(set(base_tickers + [t]))
    prices = load_prices_many(all_tickers, start=s, end=e, field=cfg.data.price_field, cache=cfg.data.cache)
    returns = prices.apply(lambda col: returns_from_prices(col, kind=str(return_kind)))

    returns_df, _ = validate_timeseries(returns, "returns", allow_nan=True, require_positive=False)
    returns_df = returns_df.dropna(how="any")
    if len(returns_df) == 0:
        raise ValueError("No returns after alignment")

    if t not in returns_df.columns:
        raise ValueError("Candidate ticker is missing from returns")

    asset_r = returns_df[t].rename(t)
    port_returns_df = returns_df.drop(columns=[t])

    base_weights = {str(k).upper(): float(v) for k, v in weights.items()}
    new_weights = add_asset(base_weights, t, float(candidate_weight), renormalize_existing=True)

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

    port_base_r = portfolio_returns(port_returns_df, base_weights).rename("portfolio_base")
    port_new_r = portfolio_returns(returns_df, new_weights).rename("portfolio_new")

    asset_perf = performance_summary(asset_r, risk_free=rf_daily, periods_per_year=int(cfg.data.trading_days))
    base_perf = performance_summary(port_base_r, risk_free=rf_daily, periods_per_year=int(cfg.data.trading_days))
    new_perf = performance_summary(port_new_r, risk_free=rf_daily, periods_per_year=int(cfg.data.trading_days))

    base_stats = portfolio_stats(
        port_returns_df,
        base_weights,
        risk_free_annual=float(cfg.data.risk_free_annual),
        periods_per_year=int(cfg.data.trading_days),
        cov_method=str(cov_method),
        cov_lam=float(cov_lam),
    )
    new_stats = portfolio_stats(
        returns_df,
        new_weights,
        risk_free_annual=float(cfg.data.risk_free_annual),
        periods_per_year=int(cfg.data.trading_days),
        cov_method=str(cov_method),
        cov_lam=float(cov_lam),
    )

    base_contr = pd.DataFrame([c.as_dict() for c in base_stats.contributions]).set_index("ticker")
    new_contr = pd.DataFrame([c.as_dict() for c in new_stats.contributions]).set_index("ticker")

    base_contr_path = tables_dir / f"{tag}_base_contributions.csv"
    new_contr_path = tables_dir / f"{tag}_new_contributions.csv"
    _save_table(base_contr, base_contr_path)
    _save_table(new_contr, new_contr_path)

    corr_port = _series_corr(asset_r, port_base_r)
    corr_table = pd.Series({c: _series_corr(asset_r, returns_df[c]) for c in base_tickers}, name="corr").sort_values(ascending=False)
    corr_df = corr_table.to_frame()
    corr_path = tables_dir / f"{tag}_asset_correlations.csv"
    _save_table(corr_df, corr_path)

    capm = capm_regression(asset_r, port_base_r, risk_free_returns=rf, periods_per_year=int(cfg.data.trading_days))
    roll_beta = rolling_capm_beta(asset_r, port_base_r, risk_free_returns=rf, window=int(rolling_window))

    factors = None
    try:
        factors = load_fama_french_factors(start=s, end=e, cache=cfg.data.cache)
    except Exception:
        factors = None

    ff = None
    if factors is not None and isinstance(factors, pd.DataFrame) and len(factors) > 0:
        try:
            ff = fama_french_regression(asset_r, factors, rf_col="RF", periods_per_year=int(cfg.data.trading_days))
        except Exception:
            ff = None

    levels = tuple(float(x) for x in var_levels)
    base_var = var_curve(port_base_r, levels=levels, horizon_days=1, method="historical", window=None)
    base_es = es_curve(port_base_r, levels=levels, horizon_days=1, method="historical", window=None)
    new_var = var_curve(port_new_r, levels=levels, horizon_days=1, method="historical", window=None)
    new_es = es_curve(port_new_r, levels=levels, horizon_days=1, method="historical", window=None)

    var_es_df = pd.DataFrame(
        {
            "base_var": base_var,
            "base_es": base_es,
            "new_var": new_var,
            "new_es": new_es,
        }
    )
    var_es_path = tables_dir / f"{tag}_var_es.csv"
    _save_table(var_es_df, var_es_path)

    weights_df = pd.DataFrame(
        {
            "base_weight": pd.Series({k: float(v) for k, v in base_weights.items()}),
            "new_weight": pd.Series({k: float(v) for k, v in new_weights.items()}),
        }
    ).fillna(0.0)
    weights_path = tables_dir / f"{tag}_weights.csv"
    _save_table(weights_df, weights_path)

    fig_prices_path = figures_dir / f"{tag}_prices.png"
    fig_equity_path = figures_dir / f"{tag}_equity.png"
    fig_beta_path = figures_dir / f"{tag}_rolling_beta.png"
    fig_contrib_base_path = figures_dir / f"{tag}_contrib_base.png"
    fig_contrib_new_path = figures_dir / f"{tag}_contrib_new.png"
    fig_var_es_path = figures_dir / f"{tag}_var_es.png"

    _, _, p1 = plot_series(prices, title=f"Prices (Portfolio + {t})", ylabel="Price", outpath=fig_prices_path)

    eq_df = pd.DataFrame(
        {
            "portfolio_base": (1.0 + port_base_r.fillna(0.0)).cumprod(),
            "portfolio_new": (1.0 + port_new_r.fillna(0.0)).cumprod(),
            t: (1.0 + asset_r.fillna(0.0)).cumprod(),
        }
    )
    _, _, p2 = plot_series(eq_df, title=f"Equity Curves (Base vs New vs {t})", ylabel="Value", outpath=fig_equity_path)
    _, _, p3 = plot_rolling_series(roll_beta.rename("beta"), title=f"{t} Rolling Beta vs Portfolio", ylabel="Beta", outpath=fig_beta_path)
    _, _, p4 = plot_risk_contributions(base_contr, title="Risk Contributions (Base)", outpath=fig_contrib_base_path)
    _, _, p5 = plot_risk_contributions(new_contr, title=f"Risk Contributions (With {t})", outpath=fig_contrib_new_path)
    _, _, p6 = plot_var_es_curve(
        levels=levels,
        var_vals=[float(new_var.loc[l]) for l in levels],
        es_vals=[float(new_es.loc[l]) for l in levels],
        title="VaR / ES (New Portfolio)",
        outpath=fig_var_es_path,
    )

    figures = tuple(
        FigureRef(title=title, path=path, kind="png")
        for title, path in [
            ("Prices", p1),
            ("Equity Curves", p2),
            ("Rolling Beta", p3),
            ("Base Risk Contributions", p4),
            ("New Risk Contributions", p5),
            ("New Portfolio VaR / ES", p6),
        ]
        if path is not None
    )

    delta_stats = {
        "delta_expected_return_annual": float(new_stats.expected_return_annual - base_stats.expected_return_annual),
        "delta_volatility_annual": float(new_stats.volatility_annual - base_stats.volatility_annual),
        "delta_sharpe": float(new_stats.sharpe - base_stats.sharpe),
        "asset_portfolio_corr": float(corr_port),
        "candidate_weight": float(candidate_weight),
    }

    capm_metrics = {
        "alpha_annual": float(capm.alpha_annual),
        "beta": float(capm.beta),
        "r2": float(capm.r2),
        "nobs": int(capm.nobs),
    }

    ff_metrics = {}
    if ff is not None:
        ff_metrics = {
            "model": ff.model,
            "alpha_annual": float(ff.alpha_annual),
            "r2": float(ff.r2),
            "nobs": int(ff.nobs),
        }

    sections: List[ReportSection] = []

    sections.append(
        ReportSection(
            title="Overview",
            text="",
            metrics={
                "candidate": t,
                "start": str(returns_df.index.min().date()) if len(returns_df) else None,
                "end": str(returns_df.index.max().date()) if len(returns_df) else None,
                "num_assets_base": int(len(base_tickers)),
                "num_assets_new": int(len(new_weights)),
                "covariance_method": str(cov_method),
            },
            figures=figures,
            tables=(
                TableRef(title="Weights (Base vs New)", path=weights_path, format="csv"),
                TableRef(title="Correlations (Candidate vs Constituents)", path=corr_path, format="csv"),
                TableRef(title="VaR / ES (Base vs New)", path=var_es_path, format="csv"),
                TableRef(title="Risk Contributions (Base)", path=base_contr_path, format="csv"),
                TableRef(title="Risk Contributions (New)", path=new_contr_path, format="csv"),
            ),
            extras={"delta": delta_stats},
        )
    )

    sections.append(
        ReportSection(
            title="Performance Comparison",
            text="",
            metrics={
                "asset": asset_perf.as_dict(),
                "portfolio_base": base_perf.as_dict(),
                "portfolio_new": new_perf.as_dict(),
            },
            figures=tuple(),
            tables=tuple(),
            extras={},
        )
    )

    sections.append(
        ReportSection(
            title="Portfolio Impact",
            text="",
            metrics={
                "base": base_stats.as_dict(),
                "new": new_stats.as_dict(),
                "delta": delta_stats,
            },
            figures=tuple(),
            tables=tuple(),
            extras={},
        )
    )

    sections.append(
        ReportSection(
            title="Asset Fit",
            text="",
            metrics={
                "capm_vs_portfolio": capm_metrics,
                "fama_french": ff_metrics,
            },
            figures=tuple(),
            tables=tuple(),
            extras={
                "capm_regression": capm.regression.as_dict(),
                "fama_french_regression": ff.regression.as_dict() if ff is not None else None,
                "fama_french_loadings": dict(ff.loadings) if ff is not None else None,
            },
        )
    )

    meta = ReportMeta(
        report_type="portfolio_fit",
        subject=t,
        as_of=returns_df.index.max().date() if len(returns_df) else None,
        sample=None,
        benchmark="portfolio_base",
        risk_free_desc="series",
        return_type=str(return_kind),
        frequency="daily",
    )

    fig_paths = tuple((f.path for sct in sections for f in sct.figures))
    table_paths = tuple((tb.path for sct in sections for tb in sct.tables))

    return ReportResult(
        meta=meta,
        sections=tuple(sections),
        output_path=None,
        figure_paths=fig_paths,
        table_paths=table_paths,
        artifacts={},
    )
