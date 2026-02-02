from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import os

import yaml

from . import paths


DateLike = Union[str, date, datetime, None]


def _to_date(x: DateLike) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return datetime.fromisoformat(s).date()
    raise TypeError(f"Unsupported date type: {type(x)}")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping at top level: {path}")
    return dict(data)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _as_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("Empty numeric string")
        return float(s)
    raise TypeError(f"Unsupported numeric type: {type(x)}")


def _as_int(x: Any) -> int:
    if isinstance(x, bool):
        raise TypeError("Bool is not a valid int for this field")
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        if int(x) != x:
            raise ValueError(f"Expected integer-like value, got {x}")
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("Empty integer string")
        return int(s)
    raise TypeError(f"Unsupported int type: {type(x)}")


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    raise TypeError(f"Unsupported bool type: {type(x)}")


def _get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _get_env_bool(key: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.getenv(key)
    if val is None:
        return default
    return _as_bool(val)


def _get_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    val = os.getenv(key)
    if val is None:
        return default
    return _as_float(val)


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    dir: Path = paths.cache

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CacheConfig":
        enabled = _as_bool(d.get("enabled", True))
        dir_val = d.get("dir", str(paths.cache))
        dir_path = Path(dir_val) if isinstance(dir_val, str) else Path(dir_val)
        return CacheConfig(enabled=enabled, dir=dir_path)


@dataclass(frozen=True)
class DataConfig:
    start: Optional[date] = None
    end: Optional[date] = None
    frequency: str = "1D"
    price_field: str = "adj_close"
    return_type: str = "log"
    trading_days: int = 252
    benchmark_ticker: str = "SPY"
    risk_free_annual: float = 0.0
    cache: CacheConfig = field(default_factory=CacheConfig)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "DataConfig":
        cache = CacheConfig.from_dict(d.get("cache", {}) or {})
        start = _to_date(d.get("start"))
        end = _to_date(d.get("end"))
        frequency = str(d.get("frequency", "1D"))
        price_field = str(d.get("price_field", "adj_close"))
        return_type = str(d.get("return_type", "log")).lower()
        trading_days = _as_int(d.get("trading_days", 252))
        benchmark_ticker = str(d.get("benchmark_ticker", "SPY")).upper()
        risk_free_annual = _as_float(d.get("risk_free_annual", 0.0))
        return DataConfig(
            start=start,
            end=end,
            frequency=frequency,
            price_field=price_field,
            return_type=return_type,
            trading_days=trading_days,
            benchmark_ticker=benchmark_ticker,
            risk_free_annual=risk_free_annual,
            cache=cache,
        )


@dataclass(frozen=True)
class RiskConfig:
    var_levels: Tuple[float, ...] = (0.95, 0.99)
    var_window: int = 252
    es_window: int = 252
    var_method: str = "historical"
    es_method: str = "historical"
    mc_paths: int = 100000

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "RiskConfig":
        levels_raw = d.get("var_levels", (0.95, 0.99))
        if isinstance(levels_raw, (list, tuple)):
            levels = tuple(float(x) for x in levels_raw)
        else:
            levels = (float(levels_raw),)
        var_window = _as_int(d.get("var_window", 252))
        es_window = _as_int(d.get("es_window", 252))
        var_method = str(d.get("var_method", "historical")).lower()
        es_method = str(d.get("es_method", "historical")).lower()
        mc_paths = _as_int(d.get("mc_paths", 100000))
        return RiskConfig(
            var_levels=levels,
            var_window=var_window,
            es_window=es_window,
            var_method=var_method,
            es_method=es_method,
            mc_paths=mc_paths,
        )


@dataclass(frozen=True)
class FactorConfig:
    enabled: bool = True
    model: str = "ff5"
    rolling_window: int = 252
    use_cache: bool = True

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FactorConfig":
        enabled = _as_bool(d.get("enabled", True))
        model = str(d.get("model", "ff5")).lower()
        rolling_window = _as_int(d.get("rolling_window", 252))
        use_cache = _as_bool(d.get("use_cache", True))
        return FactorConfig(enabled=enabled, model=model, rolling_window=rolling_window, use_cache=use_cache)


@dataclass(frozen=True)
class VolatilityConfig:
    model: str = "ewma"
    ewma_lambda: float = 0.94
    use_garch: bool = False
    garch_p: int = 1
    garch_q: int = 1
    garch_dist: str = "normal"

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "VolatilityConfig":
        model = str(d.get("model", "ewma")).lower()
        ewma_lambda = _as_float(d.get("ewma_lambda", 0.94))
        use_garch = _as_bool(d.get("use_garch", False))
        garch_p = _as_int(d.get("garch_p", 1))
        garch_q = _as_int(d.get("garch_q", 1))
        garch_dist = str(d.get("garch_dist", "normal")).lower()
        return VolatilityConfig(
            model=model,
            ewma_lambda=ewma_lambda,
            use_garch=use_garch,
            garch_p=garch_p,
            garch_q=garch_q,
            garch_dist=garch_dist,
        )


@dataclass(frozen=True)
class RenderConfig:
    format: str = "md"
    write_figures: bool = True
    write_tables: bool = True
    report_dir: Path = paths.reports
    figures_dir: Path = paths.figures
    tables_dir: Path = paths.tables

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "RenderConfig":
        fmt = str(d.get("format", "md")).lower()
        write_figures = _as_bool(d.get("write_figures", True))
        write_tables = _as_bool(d.get("write_tables", True))
        report_dir = Path(d.get("report_dir", str(paths.reports)))
        figures_dir = Path(d.get("figures_dir", str(paths.figures)))
        tables_dir = Path(d.get("tables_dir", str(paths.tables)))
        return RenderConfig(
            format=fmt,
            write_figures=write_figures,
            write_tables=write_tables,
            report_dir=report_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
        )


@dataclass(frozen=True)
class ReportConfig:
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ReportConfig":
        merged = _deep_merge(self.to_dict(), overrides)
        return ReportConfig.from_dict(merged)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data"]["start"] = self.data.start.isoformat() if self.data.start else None
        d["data"]["end"] = self.data.end.isoformat() if self.data.end else None
        d["data"]["cache"]["dir"] = str(self.data.cache.dir)
        d["render"]["report_dir"] = str(self.render.report_dir)
        d["render"]["figures_dir"] = str(self.render.figures_dir)
        d["render"]["tables_dir"] = str(self.render.tables_dir)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ReportConfig":
        data = DataConfig.from_dict(d.get("data", {}) or {})
        risk = RiskConfig.from_dict(d.get("risk", {}) or {})
        factors = FactorConfig.from_dict(d.get("factors", {}) or {})
        volatility = VolatilityConfig.from_dict(d.get("volatility", {}) or {})
        render = RenderConfig.from_dict(d.get("render", {}) or {})
        return ReportConfig(data=data, risk=risk, factors=factors, volatility=volatility, render=render)


@dataclass(frozen=True)
class PortfolioConfig:
    name: str = "portfolio"
    holdings: Dict[str, float] = field(default_factory=dict)
    normalize: bool = True

    def normalized(self) -> "PortfolioConfig":
        if not self.holdings:
            return self
        total = sum(self.holdings.values())
        if total == 0:
            raise ValueError("Portfolio weights sum to zero")
        if not self.normalize:
            return self
        weights = {k.upper(): v / total for k, v in self.holdings.items()}
        return replace(self, holdings=weights)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PortfolioConfig":
        name = str(d.get("name", "portfolio"))
        normalize = _as_bool(d.get("normalize", True))
        raw = d.get("holdings", d.get("weights", {}))
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("Portfolio holdings must be a mapping of ticker -> weight")
        holdings = {str(k).upper(): _as_float(v) for k, v in raw.items()}
        cfg = PortfolioConfig(name=name, holdings=holdings, normalize=normalize)
        return cfg.normalized()


def default_report_config() -> ReportConfig:
    data_overrides: Dict[str, Any] = {}
    rf = _get_env_float("QRT_RISK_FREE_ANNUAL")
    if rf is not None:
        data_overrides["risk_free_annual"] = rf
    bench = _get_env_str("QRT_BENCHMARK_TICKER")
    if bench is not None:
        data_overrides["benchmark_ticker"] = bench
    cache_enabled = _get_env_bool("QRT_CACHE_ENABLED")
    cache_dir_env = _get_env_str("QRT_CACHE_DIR")
    base = ReportConfig()
    merged = base.to_dict()
    if data_overrides:
        merged["data"] = _deep_merge(merged.get("data", {}), data_overrides)
    if cache_enabled is not None:
        merged["data"] = _deep_merge(merged.get("data", {}), {"cache": {"enabled": cache_enabled}})
    if cache_dir_env is not None:
        merged["data"] = _deep_merge(merged.get("data", {}), {"cache": {"dir": cache_dir_env}})
    return ReportConfig.from_dict(merged)


def load_report_config(path: Optional[Union[str, Path]] = None, overrides: Optional[Mapping[str, Any]] = None) -> ReportConfig:
    cfg = default_report_config()
    config_path = Path(path) if path is not None else (paths.configs / "report.yaml")
    file_data = _read_yaml(config_path)
    if file_data:
        cfg = cfg.with_overrides(file_data)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def load_portfolio_config(path: Optional[Union[str, Path]] = None, overrides: Optional[Mapping[str, Any]] = None) -> PortfolioConfig:
    config_path = Path(path) if path is not None else (paths.configs / "portfolio.yaml")
    file_data = _read_yaml(config_path)
    if overrides:
        file_data = _deep_merge(file_data, overrides)
    return PortfolioConfig.from_dict(file_data)


def ensure_config_files() -> None:
    paths.configs.mkdir(parents=True, exist_ok=True)
    report_path = paths.configs / "report.yaml"
    portfolio_path = paths.configs / "portfolio.yaml"
    if not report_path.exists():
        cfg = default_report_config().to_dict()
        with report_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    if not portfolio_path.exists():
        example = {"name": "portfolio", "normalize": True, "holdings": {}}
        with portfolio_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(example, f, sort_keys=False)
