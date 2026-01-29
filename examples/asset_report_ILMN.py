import yfinance as yf

from quant_research_toolkit.reports import analyze_asset, render_asset_report


def main():
    ticker = "ILMN"
    benchmark = "SPY"

    start = "2015-01-01"

    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"].dropna()
    bm = yf.download(benchmark, start=start, auto_adjust=True, progress=False)["Close"].dropna()

    report = analyze_asset(
        prices=px.rename(ticker),
        benchmark_prices=bm.rename(benchmark),
        risk_free=0.0,
        periods_per_year=252,
        ff5=True,
        rolling_window=252,
        use_garch=False,
        include_var_es=True,
        var_es_alpha=0.05,
        var_es_window=252,
        var_es_include_mc=False,
    )

    render_asset_report(report, show_series_head=False)


if __name__ == "__main__":
    main()
