from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Scalar = Union[str, int, float, bool, None]
JsonValue = Union[Scalar, Dict[str, Any], List[Any]]


@dataclass(frozen=True)
class TimeRange:
    start: Optional[date] = None
    end: Optional[date] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


@dataclass(frozen=True)
class FigureRef:
    title: str
    path: Path
    kind: str = "png"
    caption: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "path": str(self.path),
            "kind": self.kind,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class TableRef:
    title: str
    path: Path
    format: str = "csv"
    caption: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "path": str(self.path),
            "format": self.format,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class ReportSection:
    title: str
    text: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    figures: Tuple[FigureRef, ...] = field(default_factory=tuple)
    tables: Tuple[TableRef, ...] = field(default_factory=tuple)
    extras: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "metrics": self.metrics,
            "figures": [f.as_dict() for f in self.figures],
            "tables": [t.as_dict() for t in self.tables],
            "extras": self.extras,
        }


@dataclass(frozen=True)
class ReportMeta:
    report_type: str
    subject: str
    as_of: Optional[date] = None
    sample: Optional[TimeRange] = None
    benchmark: Optional[str] = None
    risk_free_desc: Optional[str] = None
    return_type: Optional[str] = None
    frequency: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "report_type": self.report_type,
            "subject": self.subject,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "sample": self.sample.as_dict() if self.sample else None,
            "benchmark": self.benchmark,
            "risk_free_desc": self.risk_free_desc,
            "return_type": self.return_type,
            "frequency": self.frequency,
            "created_at": self.created_at.replace(microsecond=0).isoformat() + "Z",
        }


@dataclass(frozen=True)
class ReportResult:
    meta: ReportMeta
    sections: Tuple[ReportSection, ...]
    output_path: Optional[Path] = None
    figure_paths: Tuple[Path, ...] = field(default_factory=tuple)
    table_paths: Tuple[Path, ...] = field(default_factory=tuple)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.as_dict(),
            "sections": [s.as_dict() for s in self.sections],
            "output_path": str(self.output_path) if self.output_path else None,
            "figure_paths": [str(p) for p in self.figure_paths],
            "table_paths": [str(p) for p in self.table_paths],
            "artifacts": self.artifacts,
        }

    def section(self, title: str) -> Optional[ReportSection]:
        for s in self.sections:
            if s.title == title:
                return s
        return None

    def metrics_flat(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for s in self.sections:
            for k, v in s.metrics.items():
                out[f"{s.title}.{k}"] = v
        return out


@dataclass(frozen=True)
class RegressionResult:
    model: str
    dep: str
    indep: Tuple[str, ...]
    params: Dict[str, float]
    tstats: Optional[Dict[str, float]] = None
    pvalues: Optional[Dict[str, float]] = None
    r2: Optional[float] = None
    adj_r2: Optional[float] = None
    nobs: Optional[int] = None
    stderr: Optional[Dict[str, float]] = None
    resid_std: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dep": self.dep,
            "indep": list(self.indep),
            "params": self.params,
            "tstats": self.tstats,
            "pvalues": self.pvalues,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "nobs": self.nobs,
            "stderr": self.stderr,
            "resid_std": self.resid_std,
        }


@dataclass(frozen=True)
class VaRResult:
    level: float
    horizon_days: int
    method: str
    var: float
    window: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "horizon_days": self.horizon_days,
            "method": self.method,
            "var": self.var,
            "window": self.window,
        }


@dataclass(frozen=True)
class ESResult:
    level: float
    horizon_days: int
    method: str
    es: float
    window: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "horizon_days": self.horizon_days,
            "method": self.method,
            "es": self.es,
            "window": self.window,
        }


@dataclass(frozen=True)
class BacktestResult:
    test: str
    level: float
    horizon_days: int
    window: Optional[int]
    nobs: int
    violations: int
    violation_rate: float
    expected_rate: float
    statistic: Optional[float] = None
    pvalue: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "test": self.test,
            "level": self.level,
            "horizon_days": self.horizon_days,
            "window": self.window,
            "nobs": self.nobs,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "expected_rate": self.expected_rate,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
        }


@dataclass(frozen=True)
class PortfolioImpact:
    asset: str
    weight: float
    before: Dict[str, float]
    after: Dict[str, float]
    delta: Dict[str, float]
    method: str = "pro_rata"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "weight": self.weight,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "method": self.method,
        }


@dataclass(frozen=True)
class RiskContribution:
    ticker: str
    weight: float
    mcr: float
    contribution: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "weight": self.weight,
            "mcr": self.mcr,
            "contribution": self.contribution,
        }
