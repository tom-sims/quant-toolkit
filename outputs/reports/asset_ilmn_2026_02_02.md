# ASSET — ILMN

## Meta

- **report_type**: asset
- **subject**: ILMN
- **as_of**: 2026-02-02
- **sample**: None
- **benchmark**: SPY
- **risk_free_desc**: series
- **return_type**: log
- **frequency**: daily
- **created_at**: 2026-02-02T19:20:21.952572

## Overview

### Metrics

- **ticker**: ILMN
- **start**: 2018-01-03
- **end**: 2026-02-02
- **observations**: 2,031.00
- **benchmark**: SPY

### Figures

- Price: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_prices.png`
- Equity Curve: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_equity.png`
- Drawdown: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_drawdown.png`
- Returns Histogram: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_returns_hist.png`
- Q-Q Plot: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_qq.png`

## Performance

### Metrics

- **periods**: 2,031.00
- **total_return**: -0.6920
- **cagr**: -0.1359
- **vol**: 0.4299
- **sharpe**: -0.1709
- **sortino**: -0.1695
- **max_drawdown**: -0.9059
- **best_day**: 0.2213
- **worst_day**: -0.1757
- **positive_day_frac**: 0.5052
- **skew**: 0.1518
- **kurtosis**: 8.8933

## Risk Adjusted

### Metrics

- **treynor**: -0.0633
- **information_ratio**: -0.5102
- **calmar**: -0.1501
- **tracking_error**: 0.3672
- **beta**: 1.1612
- **alpha_annual**: -0.2057

## VaR and Expected Shortfall

### Metrics

- **levels**: [0.95, 0.99]

### Figures

- VaR / ES Curve: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_var_es_curve.png`

### Tables

- VaR / ES Table: `/Users/tom/Desktop/quant-toolkit/outputs/tables/ilmn_var_es.csv`
- VaR Backtest Table: `/Users/tom/Desktop/quant-toolkit/outputs/tables/ilmn_var_backtest.csv`
- Stress Scenarios: `/Users/tom/Desktop/quant-toolkit/outputs/tables/ilmn_stress.csv`

### Extras

- **backtest**: {'test': 'kupiec_uc', 'level': 0.95, 'horizon_days': 1, 'window': 252, 'nobs': 1779, 'violations': 88, 'violation_rate': 0.04946599213041034, 'expected_rate': 0.050000000000000044, 'statistic': 0.010716372319734546, 'pvalue': 0.9175503463683441}

## CAPM

### Metrics

- **alpha_annual**: -0.1860
- **beta**: 1.1612
- **r2**: 0.2755
- **nobs**: 2,031.00

### Figures

- Rolling Beta: `/Users/tom/Desktop/quant-toolkit/outputs/figures/ilmn_rolling_beta.png`

### Tables

- CAPM Regression: `/Users/tom/Desktop/quant-toolkit/outputs/tables/ilmn_capm.csv`

### Extras

- **regression**: {'model': 'CAPM', 'dep': 'asset', 'indep': ['const', 'MKT'], 'params': {'const': -0.0008162911030305138, 'MKT': 1.1611998753010817}, 'tstats': {'const': -1.5945526420701093, 'MKT': 27.775825255529952}, 'pvalues': {'const': 0.11096805395189045, 'MKT': 3.510224787398896e-144}, 'r2': 0.2754856066661371, 'adj_r2': 0.27512852712284774, 'nobs': 2031, 'stderr': {'const': 0.0005119248380353084, 'MKT': 0.041806134097488096}, 'resid_std': None}
