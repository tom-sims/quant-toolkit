from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _slug(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "report"


def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _format_number(x: Any, digits: int = 4) -> Any:
    try:
        v = float(x)
    except Exception:
        return x
    if v != v:
        return "nan"
    if abs(v) >= 1_000_000:
        return f"{v:,.0f}"
    if abs(v) >= 1_000:
        return f"{v:,.2f}"
    return f"{v:.{int(digits)}f}"


def _default_markdown(report: Any) -> str:
    meta = getattr(report, "meta", None)
    sections = getattr(report, "sections", None) or ()

    meta_d = _to_plain(meta) if meta is not None else {}
    title = f"{meta_d.get('report_type', 'report').upper()} — {meta_d.get('subject', '')}".strip()

    lines = []
    lines.append(f"# {title}")
    lines.append("")

    if meta_d:
        lines.append("## Meta")
        lines.append("")
        for k, v in meta_d.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    for sec in sections:
        sec_title = getattr(sec, "title", None) or "Section"
        lines.append(f"## {sec_title}")
        lines.append("")

        text = getattr(sec, "text", None) or ""
        if text.strip():
            lines.append(str(text).strip())
            lines.append("")

        metrics = getattr(sec, "metrics", None) or {}
        if metrics:
            lines.append("### Metrics")
            lines.append("")
            for k, v in _to_plain(metrics).items():
                lines.append(f"- **{k}**: {_format_number(v)}")
            lines.append("")

        figures = getattr(sec, "figures", None) or ()
        if figures:
            lines.append("### Figures")
            lines.append("")
            for f in figures:
                ft = getattr(f, "title", None) or "Figure"
                fp = getattr(f, "path", None)
                if fp is None:
                    continue
                lines.append(f"- {ft}: `{fp}`")
            lines.append("")

        tables = getattr(sec, "tables", None) or ()
        if tables:
            lines.append("### Tables")
            lines.append("")
            for tb in tables:
                tt = getattr(tb, "title", None) or "Table"
                tp = getattr(tb, "path", None)
                if tp is None:
                    continue
                lines.append(f"- {tt}: `{tp}`")
            lines.append("")

        extras = getattr(sec, "extras", None) or {}
        if extras:
            lines.append("### Extras")
            lines.append("")
            for k, v in _to_plain(extras).items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _env(template_dir: Union[str, Path]) -> Environment:
    loader = FileSystemLoader(str(_as_path(template_dir)))
    e = Environment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters["num"] = _format_number
    return e


def render_report_md(
    report: Any,
    *,
    template_dir: Union[str, Path],
    template_name: Optional[str] = None,
    output_dir: Union[str, Path] = "outputs/reports",
    filename: Optional[str] = None,
) -> Tuple[str, Path]:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = getattr(report, "meta", None)
    meta_d = _to_plain(meta) if meta is not None else {}
    report_type = str(meta_d.get("report_type", "report"))
    subject = str(meta_d.get("subject", ""))
    as_of = meta_d.get("as_of", None)

    default_name = f"{_slug(report_type)}_{_slug(subject)}"
    if as_of:
        default_name = f"{default_name}_{_slug(str(as_of))}"
    fname = filename or f"{default_name}.md"
    out_path = out_dir / fname

    tname = template_name
    if tname is None:
        tname = f"{report_type}_report.md.j2"

        md = ""
    try:
        e = _env(template_dir)
        t = e.get_template(tname)
        payload = _to_plain(report)
        md = t.render(report=payload)
        if not str(md).strip():
            md = _default_markdown(report)
    except Exception:
        md = _default_markdown(report)


    out_path.write_text(md, encoding="utf-8")
    return md, out_path
