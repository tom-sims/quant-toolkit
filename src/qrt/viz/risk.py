from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.plotting import _maybe_save


def plot_var_es_curve(
    levels: Sequence[float],
    var_vals: Sequence[float],
    es_vals: Sequence[float],
    *,
    title: str = "VaR / ES",
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    lv = np.asarray(list(levels), dtype=float)
    v = np.asarray(list(var_vals), dtype=float)
    e = np.asarray(list(es_vals), dtype=float)
    order = np.argsort(lv)
    lv = lv[order]
    v = v[order]
    e = e[order]

    fig, ax = plt.subplots()
    ax.plot(lv, v, label="VaR")
    ax.plot(lv, e, label="ES")
    ax.set_title(title)
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Return Threshold")
    ax.legend()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_rolling_series(
    series: pd.Series,
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    s = series.dropna()
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or (s.name or "Rolling"))
    ax.set_xlabel("Date")
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_risk_contributions(
    contributions: pd.DataFrame,
    *,
    weight_col: str = "weight",
    contrib_col: str = "contribution",
    title: str = "Risk Contributions",
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    df = contributions.copy()
    if weight_col not in df.columns or contrib_col not in df.columns:
        raise ValueError("Missing columns in contributions DataFrame")
    names = df.index.astype(str).tolist()
    vals = df[contrib_col].to_numpy(dtype=float, copy=False)

    fig, ax = plt.subplots()
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_xlabel("Asset")
    ax.set_ylabel(contrib_col)
    ax.tick_params(axis="x", rotation=90)
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved
