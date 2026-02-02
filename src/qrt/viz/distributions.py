from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.plotting import _maybe_save


def plot_return_histogram(
    returns: pd.Series,
    *,
    bins: int = 60,
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    r = returns.dropna().to_numpy(dtype=float, copy=False)
    fig, ax = plt.subplots()
    ax.hist(r, bins=int(bins))
    ax.set_title(title or (returns.name or "Returns Histogram"))
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_qq(
    returns: pd.Series,
    *,
    dist: str = "normal",
    title: Optional[str] = None,
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    x = returns.dropna().to_numpy(dtype=float, copy=False)
    d = str(dist).lower().strip()
    fig, ax = plt.subplots()

    if d in {"normal", "gaussian"}:
        stats.probplot(x, dist="norm", plot=ax)
        ax.set_title(title or "Q-Q Plot (Normal)")
    elif d in {"t", "student", "student_t"}:
        stats.probplot(x, dist=stats.t, sparams=(5,), plot=ax)
        ax.set_title(title or "Q-Q Plot (t, df=5)")
    else:
        raise ValueError(f"Unsupported dist: {dist}")

    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved
