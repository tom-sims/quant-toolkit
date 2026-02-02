from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotPaths:
    figures_dir: Path

    def ensure(self) -> "PlotPaths":
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _maybe_save(fig: plt.Figure, outpath: Optional[Union[str, Path]], dpi: int = 160) -> Optional[Path]:
    if outpath is None:
        return None
    path = _as_path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    return path


def plot_price_series(
    prices: pd.Series,
    *,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    s = prices.dropna()
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or (s.name or "Price"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_equity_curve(
    curve: pd.Series,
    *,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    s = curve.dropna()
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or (s.name or "Equity Curve"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_drawdown(
    drawdown: pd.Series,
    *,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    s = drawdown.dropna()
    fig, ax = plt.subplots()
    ax.plot(s.index, s.to_numpy(dtype=float))
    ax.set_title(title or (s.name or "Drawdown"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_histogram(
    x: Union[pd.Series, np.ndarray],
    *,
    bins: int = 50,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    if isinstance(x, pd.Series):
        vals = x.dropna().to_numpy(dtype=float, copy=False)
        name = x.name or "Histogram"
    else:
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
        name = "Histogram"
    fig, ax = plt.subplots()
    ax.hist(vals, bins=int(bins))
    ax.set_title(title or name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    df = corr.copy()
    fig, ax = plt.subplots()
    im = ax.imshow(df.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=90)
    ax.set_yticklabels([str(i) for i in df.index])
    ax.set_title(title or "Correlation")
    fig.colorbar(im, ax=ax)
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved
