import numpy as np
import pandas as pd


def _fmt(x, pct=False, ndp=4):
    try:
        v = float(x)
    except Exception:
        return str(x)

    if np.isnan(v):
        return "nan"

    if pct:
        return f"{v*100:.2f}%"
    return f"{v:.{int(ndp)}f}"


def _print_kv(title, d, order=None, pct_keys=None):
    print("")
    print(title)
    print("-" * len(title))

    if d is None:
        print("None")
        return

    pct_keys = set(pct_keys or [])
    keys = order if order is not None else list(d.keys())

    for k in keys:
        if k not in d:
            continue
        v = d[k]
        is_pct = k in pct_keys
        if isinstance(v, (int, float, np.floating)):
            print(f"{k:>22}: {_fmt(v, pct=is_pct)}")
        else:
            print(f"{k:>22}: {v}")


def _print_model_summary(name, summary):
    print("")
    print(name)
    print("-" * len(name))

    if summary is None:
        print("None")
        return

    params = summary.get("params", {})
    rsq = summary.get("rsquared", np.nan)
    ars = summary.get("rsquared_adj", np.nan)
    nobs = summary.get("nobs", np.nan)

    print(f"{'nobs':>22}: {_fmt(nobs, ndp=0)}")
    print(f"{'rsquared':>22}: {_fmt(rsq, ndp=4)}")
    print(f"{'rsquared_adj':>22}: {_fmt(ars, ndp=4)}")

    if params:
        print("")
        print("params")
        for k, v in params.items():
            print(f"  {k:>18}: {_fmt(v, ndp=6)}")


def _safe_head(x, n=5):
    try:
        if hasattr(x, "head"):
            return x.head(n)
    except Exception:
        pass
    return None


def _render_var_es(risk_block):
    if not isinstance(risk_block, dict):
        return
    if "error" in risk_block:
        print("")
        print("VaR / ES")
        print("-" * 8)
        print(f"error: {risk_block['error']}")
        return

    latest = risk_block.get("latest", {})
    lvar = latest.get("var", {})
    les = latest.get("es", {})

    if not lvar and not les:
        return

    print("")
    print("VaR / ES (latest)")
    print("-" * 16)

    if lvar:
        print("VaR (positive loss threshold):")
        for k, v in lvar.items():
            print(f"  {k:>16}: {_fmt(v, pct=True)}")

    if les:
        print("ES (positive average tail loss):")
        for k, v in les.items():
            print(f"  {k:>16}: {_fmt(v, pct=True)}")


def render_asset_report(report, show_series_head=False):
    print("")
    print("ASSET REPORT")
    print("=" * 12)

    inputs = report.get("inputs", {})
    print("")
    print("Inputs")
    print("-" * 6)
    for k, v in inputs.items():
        print(f"{k:>22}: {v}")

    perf = report.get("performance", {})
    risk = report.get("risk_adjusted", {})

    _print_kv(
        "Performance",
        perf,
        order=[
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "max_drawdown",
            "hit_rate",
            "skewness",
            "kurtosis",
        ],
        pct_keys={"cumulative_return", "cagr", "annualized_volatility", "max_drawdown", "hit_rate"},
    )

    _print_kv(
        "Risk-adjusted",
        risk,
        order=[
            "sharpe",
            "sortino",
            "calmar",
            "beta",
            "alpha",
            "r_squared",
            "tracking_error",
            "information_ratio",
        ],
        pct_keys={"alpha", "tracking_error"},
    )

    fm = report.get("factor_models", {})
    if "capm" in fm:
        _print_model_summary("CAPM", fm["capm"].get("summary"))
    if "ff5" in fm:
        _print_model_summary("FF5", fm["ff5"].get("summary"))
    if "ff5_error" in fm:
        print("")
        print("FF5 ERROR")
        print("-" * 9)
        print(fm["ff5_error"])

    vol = report.get("volatility", {})
    print("")
    print("Volatility (latest)")
    print("-" * 18)
    for k in ["realized_vol_20", "realized_vol_63", "realized_vol_252", "ewma_vol"]:
        if k in vol:
            s = vol[k]
            if isinstance(s, (pd.Series, pd.DataFrame)) and len(s) > 0:
                v = s.dropna().iloc[-1]
                if isinstance(v, pd.Series):
                    v = float(v.iloc[0])
                print(f"{k:>22}: {_fmt(v, pct=True)}")

    if "garch_error" in vol:
        print(f"{'garch_error':>22}: {vol['garch_error']}")
    if "garch" in vol and isinstance(vol["garch"], dict) and "vol" in vol["garch"]:
        gvol = vol["garch"]["vol"]
        if isinstance(gvol, pd.Series) and len(gvol.dropna()) > 0:
            print(f"{'garch_vol_last':>22}: {_fmt(float(gvol.dropna().iloc[-1]), pct=True)}")

    _render_var_es(report.get("risk"))

    if show_series_head:
        print("")
        print("Series preview (head)")
        print("-" * 20)
        for name, s in report.get("series", {}).items():
            h = _safe_head(s, 5)
            if h is not None:
                print("")
                print(name)
                print(h)


def render_portfolio_report(report, show_series_head=False, show_asset_table=False):
    print("")
    print("PORTFOLIO REPORT")
    print("=" * 16)

    inputs = report.get("inputs", {})
    print("")
    print("Inputs")
    print("-" * 6)
    for k, v in inputs.items():
        if k == "weights" and isinstance(v, dict):
            print(f"{k:>22}:")
            for a, w in v.items():
                print(f"  {a:>18}: {_fmt(w, ndp=6)}")
        else:
            print(f"{k:>22}: {v}")

    perf = report.get("performance", {})
    risk = report.get("risk_adjusted", {})

    _print_kv(
        "Performance",
        perf,
        order=[
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "max_drawdown",
            "hit_rate",
            "skewness",
            "kurtosis",
        ],
        pct_keys={"cumulative_return", "cagr", "annualized_volatility", "max_drawdown", "hit_rate"},
    )

    _print_kv(
        "Risk-adjusted",
        risk,
        order=[
            "sharpe",
            "sortino",
            "calmar",
            "beta",
            "alpha",
            "r_squared",
            "tracking_error",
            "information_ratio",
        ],
        pct_keys={"alpha", "tracking_error"},
    )

    fm = report.get("factor_models", {})
    if "capm" in fm:
        _print_model_summary("CAPM", fm["capm"].get("summary"))
    if "ff5" in fm:
        _print_model_summary("FF5", fm["ff5"].get("summary"))
    if "ff5_error" in fm:
        print("")
        print("FF5 ERROR")
        print("-" * 9)
        print(fm["ff5_error"])

    vol = report.get("volatility", {})
    print("")
    print("Volatility (latest)")
    print("-" * 18)
    for k in ["realized_vol_20", "realized_vol_63", "realized_vol_252", "ewma_vol"]:
        if k in vol:
            s = vol[k]
            if isinstance(s, (pd.Series, pd.DataFrame)) and len(s) > 0:
                v = s.dropna().iloc[-1]
                if isinstance(v, pd.Series):
                    v = float(v.iloc[0])
                print(f"{k:>22}: {_fmt(v, pct=True)}")

    if "garch_error" in vol:
        print(f"{'garch_error':>22}: {vol['garch_error']}")
    if "garch" in vol and isinstance(vol["garch"], dict) and "vol" in vol["garch"]:
        gvol = vol["garch"]["vol"]
        if isinstance(gvol, pd.Series) and len(gvol.dropna()) > 0:
            print(f"{'garch_vol_last':>22}: {_fmt(float(gvol.dropna().iloc[-1]), pct=True)}")

    _render_var_es(report.get("risk"))

    if show_asset_table:
        tbl = report.get("asset_performance_table", None)
        if isinstance(tbl, pd.DataFrame):
            print("")
            print("Asset performance table")
            print("-" * 22)
            print(tbl)

    if show_series_head:
        print("")
        print("Series preview (head)")
        print("-" * 20)
        for name, s in report.get("series", {}).items():
            h = _safe_head(s, 5)
            if h is not None:
                print("")
                print(name)
                print(h)
