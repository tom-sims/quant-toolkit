from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.plotting import _maybe_save


def plot_factor_loadings(
    loadings: Dict[str, float],
    *,
    title: str = "Factor Loadings",
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    names = list(loadings.keys())
    vals = np.asarray([float(loadings[k]) for k in names], dtype=float)

    fig, ax = plt.subplots()
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Loading")
    ax.tick_params(axis="x", rotation=90)
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved


def plot_rolling_loadings(
    rolling_params: pd.DataFrame,
    *,
    title: str = "Rolling Factor Loadings",
    outpath: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> Tuple[plt.Figure, plt.Axes, Optional[Path]]:
    df = rolling_params.dropna(how="all")
    if len(df) == 0:
        raise ValueError("No rolling parameters to plot")

    fig, ax = plt.subplots()
    cols = [c for c in df.columns if str(c) != "const"]
    for c in cols:
        ax.plot(df.index, df[c].to_numpy(dtype=float, copy=False), label=str(c))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Loading")
    fig.autofmt_xdate()
    saved = _maybe_save(fig, outpath, dpi=dpi)
    return fig, ax, saved
