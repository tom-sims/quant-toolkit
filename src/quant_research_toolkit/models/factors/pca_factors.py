# === PCA FACTOR MODEL INPUTS ===
# returns : DataFrame (required)
#           values must be simple periodic returns (daily typical), e.g.
#           returns = prices[["AAPL","MSFT","GOOG"]].pct_change().dropna()
#
# target : optional Series
#           if provided, this is what you regress on the PCA factors
#           if not provided:
#               - if weights is None: regress each column? (we do NOT do that here)
#               - if weights is provided: target becomes the portfolio return built from returns + weights
#
# weights : optional (only used if target is None)
#           list of weights same length as returns columns, will be normalized automatically
#           used to construct a portfolio return series named "Portfolio"
#
# n_factors : number of principal components to use (default 5)
#
# standardize : if True, standardize each asset return series before PCA
#               (demean + divide by std). If False, PCA uses demean only.
#
# How to call:
#
# # PCA factors from a cross-section, regress a portfolio on them
# rets = prices[["AAPL","MSFT","GOOG","AMZN","META"]].pct_change().dropna()
# model = PCAFactors(rets, weights=[0.2,0.2,0.2,0.2,0.2], n_factors=3)
# print(model.params)
# print(model.rsquared_adj)
#
# # Use your own target series (e.g. a strategy return) but factors from a universe
# strat = my_strategy_returns.rename("Strategy")
# model = PCAFactors(rets, target=strat, n_factors=5)
#
# Extra outputs attached to model:
# model.pca                     -> fitted sklearn PCA object
# model.factor_returns           -> DataFrame of factor returns (PC1..PCk)
# model.explained_variance_ratio -> variance ratio per PC
# model.assets_used              -> list of asset columns used in PCA

import pandas as pd
import numpy as np
import statsmodels.api as sm

try:
    from sklearn.decomposition import PCA
except Exception as e:
    raise ImportError("scikit-learn is required for PCA factors. Install scikit-learn.") from e

def _clean_df(df):
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(how="any")
    return x


def _to_series(x, name=None):
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a Series or single-column DataFrame.")
        s = x.iloc[:, 0].copy()
        if s.name is None and name is not None:
            s.name = name
        return s
    raise TypeError("Input must be a pandas Series or DataFrame.")


def _portfolio_from_returns(returns_df, weights):
    if weights is None:
        raise ValueError("weights is required if target is not provided.")

    w = np.array(weights, dtype=float)
    if len(w) != returns_df.shape[1]:
        raise ValueError("weights length must match number of columns in returns DataFrame.")
    if w.sum() == 0:
        raise ValueError("weights sum to 0.")

    w = w / w.sum()
    port = returns_df.values @ w
    return pd.Series(port.ravel(), index=returns_df.index, name="Portfolio")


def _align_target_and_universe(target, universe_df):
    t = _clean_df(_to_series(target, name=target.name or "Target").to_frame("Target"))["Target"]
    u = _clean_df(universe_df)

    idx = t.index.intersection(u.index)
    t = t.loc[idx]
    u = u.loc[idx]
    return t, u


def _make_pca_factors(universe_returns, n_factors, standardize):
    X = universe_returns.values.astype(float)

    mu = np.nanmean(X, axis=0)
    Xc = X - mu

    if standardize:
        sig = np.nanstd(Xc, axis=0, ddof=1)
        sig[sig == 0] = 1.0
        Xc = Xc / sig

    pca = PCA(n_components=int(n_factors))
    scores = pca.fit_transform(Xc)

    cols = ["PC" + str(i + 1) for i in range(scores.shape[1])]
    factor_returns = pd.DataFrame(scores, index=universe_returns.index, columns=cols)

    return pca, factor_returns


def PCAFactors(returns, target=None, weights=None, n_factors=5, standardize=True):
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (universe returns).")

    universe = _clean_df(returns)

    if universe.shape[1] < 2:
        raise ValueError("PCA requires at least 2 assets (2+ columns) in returns.")

    if target is None:
        if weights is None:
            raise ValueError("Provide either target (Series) or weights (to build a portfolio target).")
        y = _portfolio_from_returns(universe, weights)
        y, universe = _align_target_and_universe(y, universe)
    else:
        y, universe = _align_target_and_universe(target, universe)

    pca, factors = _make_pca_factors(universe, n_factors=n_factors, standardize=standardize)

    X = sm.add_constant(factors)
    model = sm.OLS(y, X).fit()

    model.pca = pca
    model.factor_returns = factors
    model.explained_variance_ratio = pca.explained_variance_ratio_
    model.assets_used = list(universe.columns)

    return model
