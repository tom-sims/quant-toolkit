# quant-research-toolkit

A modular Python toolkit for generating **asset** and **portfolio** risk reports from tickers/weights. It produces a pitch-ready bundle of **Markdown + HTML reports**, plus saved **figures (PNGs)** and **tables (CSVs)**.

## What it does

### Asset report (single ticker)

Given a ticker (and optional date window), the toolkit generates:

- Price, equity curve, drawdown
- Return distribution diagnostics (histogram, Q-Q)
- Performance metrics (CAGR, vol, Sharpe, Sortino, max drawdown, skew/kurtosis)
- Risk-adjusted metrics (beta/alpha, tracking error, information ratio, Calmar, Treynor)
- Tail risk: Historical VaR + Expected Shortfall, backtest + stress table
- CAPM regression + rolling beta

### Portfolio report (weights)

Given a weights file (YAML), the toolkit generates:

- Portfolio returns + summary risk/performance stats
- Portfolio VaR/ES and supporting tables/figures
- Benchmark comparisons (if configured)

### Portfolio fit report (ticker vs portfolio)

Given a candidate ticker and portfolio weights, the toolkit evaluates how the candidate fits alongside the portfolio (e.g. risk impact, relative behaviour).

## Repo layout

- `src/qrt/` — library code (data loading, metrics, models, risk, portfolio, reporting)
- `scripts/` — terminal entry points to run reports
- `configs/` — YAML configs for report defaults + portfolio weights
- `outputs/` — generated reports, figures, and tables
- `notebooks/` — exploratory notebooks and examples

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```
