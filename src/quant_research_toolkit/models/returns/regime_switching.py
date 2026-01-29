# === REGIME SWITCHING INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns, e.g. prices.pct_change().dropna()
#
# k_regimes : number of regimes (typically 2)
#
# model_type : "regression" or "ar"
#           "regression" : MarkovRegression with switching mean and/or variance
#           "ar"         : MarkovAutoregression (regime-switching AR process)
#
# order : AR order (only used if model_type="ar")
#
# switching_mean : if True, mean differs by regime
# switching_variance : if True, variance differs by regime
#
# trend : "n" none, "c" constant (most common for returns), "t", "ct"
#
# scale : if True, multiplies returns by 100 before fitting (often more stable)
#         outputs (residuals / fittedvalues) are scaled back to decimal units
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# out = fit_regime_switching(aapl, k_regimes=2, model_type="regression", switching_mean=True, switching_variance=True)
# res = out["result"]
# print(res.summary())
# probs = out["smoothed_probabilities"]
#
# Returns:
# - Series input:
#   dict with keys: model, result, params, aic, bic, llf, smoothed_probabilities, filtered_probabilities, regime
# - DataFrame input:
#   dict with keys: results (dict per column), params (DataFrame), smoothed_probabilities (dict)

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
except Exception as e:
    raise ImportError("statsmodels is required for regime switching. Install statsmodels.") from e

def _to_df(x):
    if isinstance(x, pd.Series):
        return x.to_frame(x.name or "returns")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("returns must be a pandas Series or DataFrame.")


def _clean_returns(returns):
    df = _to_df(returns).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def _fit_one(series, k_regimes, model_type, order, switching_mean, switching_variance, trend, scale):
    y = series.copy()

    if scale:
        y = y * 100.0

    model_type = str(model_type).strip().lower()
    if model_type not in ("regression", "ar"):
        raise ValueError("model_type must be 'regression' or 'ar'.")

    k_regimes = int(k_regimes)
    if k_regimes < 2:
        raise ValueError("k_regimes must be >= 2.")

    trend = str(trend).strip().lower()
    if trend not in ("n", "c", "t", "ct"):
        raise ValueError("trend must be one of: 'n','c','t','ct'.")

    if model_type == "regression":
        # MarkovRegression: switching trend controls whether intercept differs by regime
        # switching_variance controls whether sigma^2 differs by regime
        switching_trend = bool(switching_mean)

        model = MarkovRegression(
            y,
            k_regimes=k_regimes,
            trend=trend,
            switching_trend=switching_trend,
            switching_variance=bool(switching_variance),
        )
    else:
        order = int(order)
        if order < 1:
            raise ValueError("order must be >= 1 when model_type='ar'.")

        # MarkovAutoregression: you can switch mean and variance; the AR coefficients can be switching too
        model = MarkovAutoregression(
            y,
            k_regimes=k_regimes,
            order=order,
            trend=trend,
            switching_ar=False,
            switching_variance=bool(switching_variance),
        )

    res = model.fit(disp=False)

    smoothed = res.smoothed_marginal_probabilities.copy()
    filtered = res.filtered_marginal_probabilities.copy()

    if scale:
        # keep probabilities as-is, but if you want fitted/resid in decimals later
        pass

    # Pick most likely regime at each time
    regime = smoothed.idxmax(axis=1).rename("regime")

    return model, res, smoothed, filtered, regime


def fit_regime_switching(
    returns,
    k_regimes=2,
    model_type="regression",
    order=1,
    switching_mean=True,
    switching_variance=True,
    trend="c",
    scale=True,
):
    df = _clean_returns(returns)

    results = {}
    params_rows = []
    probs = {}

    for col in df.columns:
        model, res, smoothed, filtered, regime = _fit_one(
            df[col],
            k_regimes=k_regimes,
            model_type=model_type,
            order=order,
            switching_mean=switching_mean,
            switching_variance=switching_variance,
            trend=trend,
            scale=scale,
        )

        results[col] = {
            "model": model,
            "result": res,
            "params": res.params.copy(),
            "aic": float(res.aic) if hasattr(res, "aic") else np.nan,
            "bic": float(res.bic) if hasattr(res, "bic") else np.nan,
            "llf": float(res.llf) if hasattr(res, "llf") else np.nan,
            "smoothed_probabilities": smoothed,
            "filtered_probabilities": filtered,
            "regime": regime,
        }

        probs[col] = smoothed

        row = res.params.copy()
        row.name = col
        params_rows.append(row)

    params_df = pd.DataFrame(params_rows)

    if isinstance(returns, pd.Series):
        one = results[df.columns[0]]
        return {
            "model": one["model"],
            "result": one["result"],
            "params": one["params"],
            "aic": one["aic"],
            "bic": one["bic"],
            "llf": one["llf"],
            "smoothed_probabilities": one["smoothed_probabilities"],
            "filtered_probabilities": one["filtered_probabilities"],
            "regime": one["regime"],
        }

    return {
        "results": results,
        "params": params_df,
        "smoothed_probabilities": probs,
    }
