from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.math import cumulative_curve, drawdown_curve
from ..utils.plotting import _maybe_save


def plot_series(
    series: Union[pd.Series, pd.DataFrame],
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    fig, ax = plt.subplots()
    if isinstance(series, pd.Series):
        s = series.dropna()
        ax.plot(s.index, s.to_numpy(dtype=float))
        ax.set_title(title or (s.name or "Series"))
    else:
        df = series.dropna(how="any")
        for c in df.columns:
            ax.plot(df.index, df[c].to_numpy(dtype=float), label=str(c))
        ax.legend()
        ax.set_title(title or "Series")
    ax.set_xlabel("Date")
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_equity_from_returns(
    returns: pd.Series,
    *,
    start: float = 1.0,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    r = returns.dropna()
    curve = cumulative_curve(r, start=float(start))
    s = pd.Series(curve, index=r.index, name="equity")
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or "Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_drawdown_from_returns(
    returns: pd.Series,
    *,
    start: float = 1.0,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    r = returns.dropna()
    curve = cumulative_curve(r, start=float(start))
    dd = drawdown_curve(curve)
    s = pd.Series(dd, index=r.index, name="drawdown")
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or "Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved
