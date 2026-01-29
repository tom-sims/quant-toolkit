# === ARIMA INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns, e.g. prices.pct_change().dropna()
#
# order : tuple (p, d, q)
#         p = AR lags, d = differencing, q = MA lags
#         common starter: (1, 0, 1) or (2, 0, 2)
#
# trend : "n" (none), "c" (constant), "t" (trend), "ct" (const+trend)
#         default "n" for returns (usually mean ~0)
#
# scale : if True, multiply returns by 100 before fitting (sometimes more stable)
#         forecasts/residuals are scaled back to decimal units
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# out = fit_arima(aapl, order=(1,0,1), trend="n", scale=True)
# print(out["params"])
# print(out["aic"])
# fc = arima_forecast(aapl, steps=10, order=(1,0,1), scale=True)
#
# Returns:
# - For Series input:
#   dict with keys: model, result, params, aic, bic, llf, resid
# - For DataFrame input:
#   dict with keys: results (dict per column), params (DataFrame)

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as e:
    raise ImportError("statsmodels is required for ARIMA. Install statsmodels.") from e

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


def _fit_one(series, order, trend, scale):
    y = series.copy()

    if scale:
        y = y * 100.0

    model = ARIMA(y, order=tuple(order), trend=str(trend))
    res = model.fit()

    resid = pd.Series(res.resid, index=series.index, name=series.name)
    if scale:
        resid = resid / 100.0

    return model, res, resid


def fit_arima(returns, order=(1, 0, 1), trend="n", scale=True):
    df = _clean_returns(returns)

    results = {}
    params_rows = []

    for col in df.columns:
        model, res, resid = _fit_one(df[col], order=order, trend=trend, scale=scale)

        results[col] = {
            "model": model,
            "result": res,
            "params": res.params.copy(),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "llf": float(res.llf),
            "resid": resid,
        }

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
            "resid": one["resid"],
        }

    return {
        "results": results,
        "params": params_df,
    }


def arima_forecast(returns, steps=10, order=(1, 0, 1), trend="n", scale=True, alpha=0.05):
    df = _clean_returns(returns)

    if isinstance(returns, pd.DataFrame):
        raise ValueError("arima_forecast currently expects a Series (single asset).")

    series = df.iloc[:, 0]
    model, res, _ = _fit_one(series, order=order, trend=trend, scale=scale)

    fc = res.get_forecast(steps=int(steps))
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=float(alpha))

    mean = pd.Series(mean, name="forecast")
    ci.columns = ["lower", "upper"]

    if scale:
        mean = mean / 100.0
        ci = ci / 100.0

    idx = pd.RangeIndex(1, int(steps) + 1, name="step")
    mean.index = idx
    ci.index = idx

    return {
        "forecast": mean,
        "ci": ci,
        "result": res,
    }
