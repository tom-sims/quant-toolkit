import numpy as np
import pandas as pd

from ..portfolio.construction import returns_from_prices, normalize_weights, portfolio_returns, equity_curve_from_returns

from ..metrics.performance import (
    drawdown_series,
    cumulative_return,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    hit_rate,
    skewness,
    kurtosis,
    performance_summary,
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

from ..risk.var import var_historical, var_gaussian, var_cornish_fisher, var_ewma, var_monte_carlo
from ..risk.expected_shortfall import es_historical, es_gaussian, es_cornish_fisher, es_monte_carlo

try:
    from ..models.volatility import fit_garch
except Exception:
    fit_garch = None


def _clean_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name or "asset")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame or Series.")
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return out


def _clean_series(s, name=None):
    if isinstance(s, pd.DataFrame):
        if s.shape[1] != 1:
            raise ValueError("Expected Series or single-column DataFrame.")
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series.")
    out = s.copy()
    if out.name is None and name is not None:
        out.name = name
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _align_series_and_df(s, df):
    idx = s.index.intersection(df.index)
    return s.loc[idx], df.loc[idx]


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


def analyze_portfolio(
    prices=None,
    returns=None,
    weights=None,
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
        prices_df = _clean_df(prices)
        asset_returns = returns_from_prices(prices_df)
    else:
        asset_returns = _clean_df(returns)

    if asset_returns.shape[1] < 1:
        raise ValueError("returns/prices must have at least 1 column.")

    if weights is None:
        w = np.ones(asset_returns.shape[1], dtype=float) / float(asset_returns.shape[1])
        weights = w

    weights = normalize_weights(weights, columns=asset_returns.columns, as_series=False)
    port_ret = portfolio_returns(asset_returns, weights, rebalance=None).rename("Portfolio")

    bench = None
    if benchmark_prices is not None:
        bench = returns_from_prices(_clean_df(benchmark_prices)).iloc[:, 0].rename(_clean_series(benchmark_prices).name or "Benchmark")
    elif benchmark_returns is not None:
        bench = _clean_series(benchmark_returns, name="Benchmark")

    if bench is not None:
        port_ret, bench_df = _align_series_and_df(port_ret, bench.to_frame("Benchmark"))
        bench = bench_df["Benchmark"]
        asset_returns = asset_returns.loc[port_ret.index]

    series = {
        "equity_curve": equity_curve_from_returns(port_ret),
        "drawdown": drawdown_series(port_ret),
    }

    perf = {
        "cumulative_return": float(cumulative_return(port_ret)),
        "cagr": float(annualized_return(port_ret, periods_per_year=periods_per_year)),
        "annualized_volatility": float(annualized_volatility(port_ret, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(port_ret)),
        "hit_rate": float(hit_rate(port_ret)),
        "skewness": float(skewness(port_ret)),
        "kurtosis": float(kurtosis(port_ret)),
    }

    asset_perf_table = performance_summary(asset_returns)

    risk_adj = {
        "sharpe": float(sharpe_ratio(port_ret, risk_free=risk_free, periods_per_year=periods_per_year)),
        "sortino": float(sortino_ratio(port_ret, risk_free=risk_free, periods_per_year=periods_per_year)),
        "calmar": float(calmar_ratio(port_ret, periods_per_year=periods_per_year)),
    }

    if bench is not None:
        risk_adj.update({
            "beta": float(capm_beta(port_ret, bench)),
            "alpha": float(capm_alpha(port_ret, bench, risk_free=risk_free, periods_per_year=periods_per_year)),
            "r_squared": float(r_squared(port_ret, bench)),
            "tracking_error": float(tracking_error(port_ret, bench, periods_per_year=periods_per_year)),
            "information_ratio": float(information_ratio(port_ret, bench, periods_per_year=periods_per_year)),
        })

    factor_models = {}

    if bench is not None:
        capm_model = CAPM(port_ret, bench, risk_free=risk_free, periods_per_year=periods_per_year)
        factor_models["capm"] = {"model": capm_model, "summary": _model_summary_sm(capm_model)}

        capm_factors = pd.DataFrame({"MKT": bench})
        factor_models["rolling_capm"] = rolling_factor_regression(
            port_ret,
            capm_factors,
            rf=None,
            window=int(rolling_window),
            add_const=True,
        )

    if ff5:
        try:
            factors, rf_series = get_fama_french_5_factors_daily(use_cache=ff5_use_cache, cache_dir=ff5_cache_dir)
            ff5_model = FF5(port_ret, factors=factors, rf=rf_series, use_cache=False)

            factor_models["ff5"] = {"model": ff5_model, "summary": _model_summary_sm(ff5_model)}
            factor_models["rolling_ff5"] = rolling_factor_regression(
                port_ret,
                factors,
                rf=rf_series,
                window=int(rolling_window),
                add_const=True,
            )
            factor_models["ff5_data"] = {"factors": factors, "rf": rf_series}
        except Exception as e:
            factor_models["ff5_error"] = str(e)

    vol_block = {
        "realized_vol_20": realized_volatility(port_ret, window=20, method="std", periods_per_year=periods_per_year),
        "realized_vol_63": realized_volatility(port_ret, window=63, method="std", periods_per_year=periods_per_year),
        "realized_vol_252": realized_volatility(port_ret, window=252, method="std", periods_per_year=periods_per_year),
        "ewma_vol": ewma_volatility(port_ret, lam=0.94, periods_per_year=periods_per_year),
    }

    if use_garch:
        if fit_garch is None:
            vol_block["garch_error"] = "arch not installed (pip install arch)"
        else:
            if garch_kwargs is None:
                garch_kwargs = {}
            try:
                vol_block["garch"] = fit_garch(port_ret, periods_per_year=periods_per_year, **garch_kwargs)
            except Exception as e:
                vol_block["garch_error"] = str(e)

    risk_block = None
    if include_var_es:
        try:
            risk_block = _risk_var_es_block(
                port_ret,
                alpha=var_es_alpha,
                window=var_es_window,
                include_mc=var_es_include_mc,
                mc_paths=var_es_mc_paths,
            )
        except Exception as e:
            risk_block = {"error": str(e)}

    return {
        "inputs": {
            "asset_names": list(asset_returns.columns),
            "weights": {c: float(w) for c, w in zip(asset_returns.columns, weights)},
            "benchmark_name": bench.name if bench is not None else None,
            "periods_per_year": int(periods_per_year),
            "risk_free": float(risk_free) if risk_free is not None else None,
            "rolling_window": int(rolling_window),
            "var_es_alpha": float(var_es_alpha),
            "var_es_window": int(var_es_window) if var_es_window is not None else None,
        },
        "asset_returns": asset_returns,
        "portfolio_returns": port_ret,
        "benchmark_returns": bench,
        "asset_performance_table": asset_perf_table,
        "performance": perf,
        "risk_adjusted": risk_adj,
        "factor_models": factor_models,
        "volatility": vol_block,
        "risk": risk_block,
        "series": series,
    }
