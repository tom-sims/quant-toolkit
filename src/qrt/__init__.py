from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional
import logging


def _resolve_version() -> str:
    for dist_name in ("quant-research-toolkit", "quant-toolkit", "qrt"):
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
    return "0.0.0"


__version__ = _resolve_version()

package_dir = Path(__file__).resolve().parent


def find_repo_root(start: Optional[Path] = None) -> Path:
    start_path = start or package_dir
    markers = ("pyproject.toml", ".git", "requirements.txt", "README.md")
    for p in (start_path, *start_path.parents):
        if any((p / m).exists() for m in markers):
            return p
    try:
        return package_dir.parents[2]
    except IndexError:
        return package_dir.parent


repo_root = find_repo_root()

configs_dir = repo_root / "configs"
outputs_dir = repo_root / "outputs"
reports_dir = outputs_dir / "reports"
figures_dir = outputs_dir / "figures"
tables_dir = outputs_dir / "tables"
cache_dir = repo_root / "cache"

templates_dir = package_dir / "reports" / "templates"


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    configs: Path
    outputs: Path
    reports: Path
    figures: Path
    tables: Path
    cache: Path
    templates: Path


paths = RuntimePaths(
    repo_root=repo_root,
    configs=configs_dir,
    outputs=outputs_dir,
    reports=reports_dir,
    figures=figures_dir,
    tables=tables_dir,
    cache=cache_dir,
    templates=templates_dir,
)


def ensure_runtime_dirs(p: RuntimePaths = paths) -> None:
    p.outputs.mkdir(parents=True, exist_ok=True)
    p.reports.mkdir(parents=True, exist_ok=True)
    p.figures.mkdir(parents=True, exist_ok=True)
    p.tables.mkdir(parents=True, exist_ok=True)
    p.cache.mkdir(parents=True, exist_ok=True)


default_logger_name = "qrt"
default_log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
default_log_datefmt = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=default_log_format, datefmt=default_log_datefmt)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str = default_logger_name) -> logging.Logger:
    return logging.getLogger(name)


__all__ = [
    "__version__",
    "package_dir",
    "repo_root",
    "configs_dir",
    "outputs_dir",
    "reports_dir",
    "figures_dir",
    "tables_dir",
    "cache_dir",
    "templates_dir",
    "RuntimePaths",
    "paths",
    "find_repo_root",
    "ensure_runtime_dirs",
    "configure_logging",
    "get_logger",
]
