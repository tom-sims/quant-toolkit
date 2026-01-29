# === GARCH VOLATILITY INPUTS ===
# returns : Series (single asset) or DataFrame (multiple assets)
#           values must be simple periodic returns (daily typical), e.g.
#           returns = prices.pct_change().dropna()
#
# model : "GARCH" (default) or "EGARCH" or "GJR-GARCH"
# p, q : GARCH orders (default p=1, q=1)
# o    : asymmetry term (used for GJR-GARCH), default 0
#
# mean : "Zero" or "Constant" (default "Zero")
# dist : "normal" or "t" (default "normal")
#
# scale : if True, multiplies returns by 100 before fitting (arch often behaves better)
#         output vol is scaled back to decimal units
#
# periods_per_year : optional
#         annualises volatility by sqrt(periods_per_year)
#
# How to call:
#
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# out = fit_garch(aapl, model="GARCH", p=1, q=1, mean="Zero", dist="t", scale=True, periods_per_year=252)
# print(out["params"])
# vol = out["vol"]
#
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# out = fit_garch(rets, model="GJR-GARCH", p=1, o=1, q=1, scale=True, periods_per_year=252)
# vols = out["vol"]
#
# Returns:
# - For Series input:
#   dict with keys: model, result, params, aic, bic, loglik, vol
# - For DataFrame input:
#   dict with keys: results (dict per column), params (DataFrame), vol (DataFrame)

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except Exception as e:
    raise ImportError("arch is required for GARCH models. Install arch.") from e

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


def _map_mean(mean):
    m = str(mean).strip().lower()
    if m in ("zero", "0", "none"):
        return "Zero"
    if m in ("constant", "const", "c"):
        return "Constant"
    raise ValueError("mean must be 'Zero' or 'Constant'.")


def _map_dist(dist):
    d = str(dist).strip().lower()
    if d in ("normal", "gaussian", "norm"):
        return "normal"
    if d in ("t", "student", "studentst", "students_t"):
        return "t"
    raise ValueError("dist must be 'normal' or 't'.")


def _map_model(model):
    m = str(model).strip().upper()
    if m in ("GARCH",):
        return "GARCH"
    if m in ("EGARCH",):
        return "EGARCH"
    if m in ("GJR", "GJR-GARCH", "GJRGARCH"):
        return "GARCH"  # still "GARCH" in arch_model, but with o>0
    raise ValueError("model must be 'GARCH', 'EGARCH', or 'GJR-GARCH'.")


def _fit_one(series, model, p, o, q, mean, dist, scale):
    s = series.copy()

    if scale:
        s = s * 100.0

    mean = _map_mean(mean)
    dist = _map_dist(dist)

    m = str(model).strip().upper()
    if m in ("GJR", "GJR-GARCH", "GJRGARCH"):
        model_type = "GARCH"
        o = int(o) if o is not None else 1
    else:
        model_type = _map_model(model)
        o = int(o) if o is not None else 0

    am = arch_model(
        s,
        mean=mean,
        vol=model_type,
        p=int(p),
        o=int(o),
        q=int(q),
        dist=dist,
        rescale=False,
    )

    res = am.fit(disp="off")

    vol = res.conditional_volatility
    vol = pd.Series(vol, index=series.index, name=series.name)

    if scale:
        vol = vol / 100.0

    return am, res, vol


def fit_garch(
    returns,
    model="GARCH",
    p=1,
    q=1,
    o=0,
    mean="Zero",
    dist="normal",
    scale=True,
    periods_per_year=None,
):
    df = _clean_returns(returns)

    results = {}
    vols = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    params_rows = []
    for col in df.columns:
        am, res, vol = _fit_one(df[col], model=model, p=p, o=o, q=q, mean=mean, dist=dist, scale=scale)

        if periods_per_year is not None:
            vol = vol * np.sqrt(float(periods_per_year))

        vols[col] = vol.values
        results[col] = {
            "model": am,
            "result": res,
            "params": res.params.copy(),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "loglik": float(res.loglikelihood),
            "vol": vol,
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
            "loglik": one["loglik"],
            "vol": one["vol"],
        }

    return {
        "results": results,
        "params": params_df,
        "vol": vols,
    }


def garch_forecast_vol(
    returns,
    horizon=20,
    model="GARCH",
    p=1,
    q=1,
    o=0,
    mean="Zero",
    dist="normal",
    scale=True,
    periods_per_year=None,
):
    df = _clean_returns(returns)

    if isinstance(returns, pd.DataFrame):
        raise ValueError("garch_forecast_vol currently expects a Series (single asset).")

    series = df.iloc[:, 0]
    _, res, _ = _fit_one(series, model=model, p=p, o=o, q=q, mean=mean, dist=dist, scale=scale)

    f = res.forecast(horizon=int(horizon), reindex=False)
    var = f.variance.values[-1]  # shape (horizon,)
    vol = np.sqrt(var)

    if scale:
        vol = vol / 100.0

    if periods_per_year is not None:
        vol = vol * np.sqrt(float(periods_per_year))

    idx = pd.RangeIndex(1, int(horizon) + 1, name="step")
    return pd.Series(vol, index=idx, name="forecast_vol")
