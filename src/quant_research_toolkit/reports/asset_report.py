import numpy as np
import pandas as pd

from ..metrics.performance import (
    equity_curve,
    drawdown_series,
    cumulative_return,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    hit_rate,
    skewness,
    kurtosis,
)

from ..metrics.risk_adjusted import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    beta as capm_beta,
    alpha as capm_alpha,
    r_squared,
    information_ratio,
    tracking_error,
)

from ..models.factors import CAPM, FF5, get_fama_french_5_factors_daily, rolling_factor_regression
from ..models.volatility import ewma_volatility, realized_volatility

from ..risk.var import (
    var_historical,
    var_gaussian,
    var_cornish_fisher,
    var_ewma,
    var_monte_carlo,
)

from ..risk.expected_shortfall import (
    es_historical,
    es_gaussian,
    es_cornish_fisher,
    es_monte_carlo,
)

try:
    from ..models.volatility import fit_garch
except Exception:
    fit_garch = None


def _to_series(x, name=None):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected Series or single-column DataFrame.")
        s = x.iloc[:, 0].copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    raise TypeError("Expected a pandas Series or single-column DataFrame.")


def _clean_series(x):
    s = _to_series(x)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _returns_from_prices(prices):
    p = _clean_series(prices)
    return p.pct_change().dropna()


def _align(a, b):
    if a is None or b is None:
        return a, b
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def _model_summary_sm(model):
    if model is None:
        return None

    params = model.params
    tvals = getattr(model, "tvalues", None)
    pvals = getattr(model, "pvalues", None)

    out = {
        "params": params.to_dict() if hasattr(params, "to_dict") else dict(params),
        "rsquared": float(getattr(model, "rsquared", np.nan)),
        "rsquared_adj": float(getattr(model, "rsquared_adj", np.nan)),
        "nobs": float(getattr(model, "nobs", np.nan)),
    }

    if tvals is not None:
        out["tvalues"] = tvals.to_dict() if hasattr(tvals, "to_dict") else dict(tvals)
    if pvals is not None:
        out["pvalues"] = pvals.to_dict() if hasattr(pvals, "to_dict") else dict(pvals)

    return out


def _last_value(x):
    if x is None:
        return np.nan
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            return np.nan
    if isinstance(x, pd.Series):
        xs = x.dropna()
        return float(xs.iloc[-1]) if len(xs) else np.nan
    return np.nan


def _risk_var_es_block(r, alpha=0.05, window=252, include_mc=False, mc_paths=100000):
    out = {"var": {}, "es": {}, "latest": {"var": {}, "es": {}}}

    v_hist = var_historical(r, alpha=alpha, window=window)
    v_gaus = var_gaussian(r, alpha=alpha, window=window)
    v_cf = var_cornish_fisher(r, alpha=alpha, window=window)
    v_ew = var_ewma(r, alpha=alpha, lam=0.94, window=None)

    out["var"]["historical"] = v_hist
    out["var"]["gaussian"] = v_gaus
    out["var"]["cornish_fisher"] = v_cf
    out["var"]["ewma"] = v_ew

    out["latest"]["var"]["historical"] = _last_value(v_hist)
    out["latest"]["var"]["gaussian"] = _last_value(v_gaus)
    out["latest"]["var"]["cornish_fisher"] = _last_value(v_cf)
    out["latest"]["var"]["ewma"] = _last_value(v_ew)

    if include_mc:
        v_mc = var_monte_carlo(r, alpha=alpha, window=window, n_paths=mc_paths)
        out["var"]["monte_carlo"] = v_mc
        out["latest"]["var"]["monte_carlo"] = _last_value(v_mc)

    e_hist = es_historical(r, alpha=alpha, window=window)
    e_gaus = es_gaussian(r, alpha=alpha, window=window)
    e_cf = es_cornish_fisher(r, alpha=alpha, window=window)

    out["es"]["historical"] = e_hist
    out["es"]["gaussian"] = e_gaus
    out["es"]["cornish_fisher"] = e_cf

    out["latest"]["es"]["historical"] = _last_value(e_hist)
    out["latest"]["es"]["gaussian"] = _last_value(e_gaus)
    out["latest"]["es"]["cornish_fisher"] = _last_value(e_cf)

    if include_mc:
        e_mc = es_monte_carlo(r, alpha=alpha, window=window, n_paths=mc_paths)
        out["es"]["monte_carlo"] = e_mc
        out["latest"]["es"]["monte_carlo"] = _last_value(e_mc)

    return out


def analyze_asset(
    prices=None,
    returns=None,
    benchmark_prices=None,
    benchmark_returns=None,
    risk_free=0.0,
    periods_per_year=252,
    ff5=True,
    rolling_window=252,
    use_garch=False,
    garch_kwargs=None,
    ff5_use_cache=True,
    ff5_cache_dir=None,
    include_var_es=True,
    var_es_alpha=0.05,
    var_es_window=252,
    var_es_include_mc=False,
    var_es_mc_paths=100000,
):
    if prices is None and returns is None:
        raise ValueError("Provide either prices or returns.")
    if prices is not None and returns is not None:
        raise ValueError("Provide only one of prices or returns (not both).")
    if benchmark_prices is not None and benchmark_returns is not None:
        raise ValueError("Provide only one of benchmark_prices or benchmark_returns (not both).")

    if prices is not None:
        asset_returns = _returns_from_prices(prices).rename(_to_series(prices).name or "Asset")
    else:
        asset_returns = _clean_series(returns).rename(_to_series(returns).name or "Asset")

    bench = None
    if benchmark_prices is not None:
        bench = _returns_from_prices(benchmark_prices).rename(_to_series(benchmark_prices).name or "Benchmark")
    elif benchmark_returns is not None:
        bench = _clean_series(benchmark_returns).rename(_to_series(benchmark_returns).name or "Benchmark")

    asset_returns, bench = _align(asset_returns, bench)

    series = {
        "equity_curve": equity_curve(asset_returns),
        "drawdown": drawdown_series(asset_returns),
    }

    perf = {
        "cumulative_return": float(cumulative_return(asset_returns)),
        "cagr": float(annualized_return(asset_returns, periods_per_year=periods_per_year)),
        "annualized_volatility": float(annualized_volatility(asset_returns, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(asset_returns)),
        "hit_rate": float(hit_rate(asset_returns)),
        "skewness": float(skewness(asset_returns)),
        "kurtosis": float(kurtosis(asset_returns)),
    }

    risk_adj = {
        "sharpe": float(sharpe_ratio(asset_returns, risk_free=risk_free, periods_per_year=periods_per_year)),
        "sortino": float(sortino_ratio(asset_returns, risk_free=risk_free, periods_per_year=periods_per_year)),
        "calmar": float(calmar_ratio(asset_returns, periods_per_year=periods_per_year)),
    }

    if bench is not None:
        risk_adj.update({
            "beta": float(capm_beta(asset_returns, bench)),
            "alpha": float(capm_alpha(asset_returns, bench, risk_free=risk_free, periods_per_year=periods_per_year)),
            "r_squared": float(r_squared(asset_returns, bench)),
            "tracking_error": float(tracking_error(asset_returns, bench, periods_per_year=periods_per_year)),
            "information_ratio": float(information_ratio(asset_returns, bench, periods_per_year=periods_per_year)),
        })

    factor_models = {}

    if bench is not None:
        capm_model = CAPM(asset_returns, bench, risk_free=risk_free, periods_per_year=periods_per_year)
        factor_models["capm"] = {"model": capm_model, "summary": _model_summary_sm(capm_model)}

        capm_factors = pd.DataFrame({"MKT": bench})
        factor_models["rolling_capm"] = rolling_factor_regression(
            asset_returns,
            capm_factors,
            rf=None,
            window=int(rolling_window),
            add_const=True,
        )

    if ff5:
        try:
            factors, rf_series = get_fama_french_5_factors_daily(use_cache=ff5_use_cache, cache_dir=ff5_cache_dir)
            ff5_model = FF5(asset_returns, factors=factors, rf=rf_series, use_cache=False)

            factor_models["ff5"] = {"model": ff5_model, "summary": _model_summary_sm(ff5_model)}
            factor_models["rolling_ff5"] = rolling_factor_regression(
                asset_returns,
                factors,
                rf=rf_series,
                window=int(rolling_window),
                add_const=True,
            )
            factor_models["ff5_data"] = {"factors": factors, "rf": rf_series}
        except Exception as e:
            factor_models["ff5_error"] = str(e)

    vol_block = {
        "realized_vol_20": realized_volatility(asset_returns, window=20, method="std", periods_per_year=periods_per_year),
        "realized_vol_63": realized_volatility(asset_returns, window=63, method="std", periods_per_year=periods_per_year),
        "realized_vol_252": realized_volatility(asset_returns, window=252, method="std", periods_per_year=periods_per_year),
        "ewma_vol": ewma_volatility(asset_returns, lam=0.94, periods_per_year=periods_per_year),
    }

    if use_garch:
        if fit_garch is None:
            vol_block["garch_error"] = "arch not installed (pip install arch)"
        else:
            if garch_kwargs is None:
                garch_kwargs = {}
            try:
                vol_block["garch"] = fit_garch(asset_returns, periods_per_year=periods_per_year, **garch_kwargs)
            except Exception as e:
                vol_block["garch_error"] = str(e)

    risk_block = None
    if include_var_es:
        try:
            risk_block = _risk_var_es_block(
                asset_returns,
                alpha=var_es_alpha,
                window=var_es_window,
                include_mc=var_es_include_mc,
                mc_paths=var_es_mc_paths,
            )
        except Exception as e:
            risk_block = {"error": str(e)}

    return {
        "inputs": {
            "asset_name": asset_returns.name,
            "benchmark_name": bench.name if bench is not None else None,
            "periods_per_year": int(periods_per_year),
            "risk_free": float(risk_free) if risk_free is not None else None,
            "rolling_window": int(rolling_window),
            "var_es_alpha": float(var_es_alpha),
            "var_es_window": int(var_es_window) if var_es_window is not None else None,
        },
        "returns": asset_returns,
        "benchmark_returns": bench,
        "performance": perf,
        "risk_adjusted": risk_adj,
        "factor_models": factor_models,
        "volatility": vol_block,
        "risk": risk_block,
        "series": series,
    }
