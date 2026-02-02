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


def _default_html(report: Any) -> str:
    payload = _to_plain(report) or {}
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    sections = payload.get("sections", []) if isinstance(payload, dict) else []

    title = f"{str(meta.get('report_type', 'report')).upper()} — {meta.get('subject', '')}".strip()
    parts = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append(f"<title>{title}</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;max-width:980px}"
        "h1,h2,h3{margin:0.6em 0 0.3em}"
        "code,pre{background:#f5f5f5;padding:2px 6px;border-radius:6px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:left}"
        "th{background:#fafafa}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}"
        ".card{border:1px solid #e5e5e5;border-radius:12px;padding:12px}"
        ".muted{color:#666}"
        "img{max-width:100%;border-radius:12px;border:1px solid #eee}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>{title}</h1>")

    if meta:
        parts.append("<h2>Meta</h2>")
        parts.append("<div class='grid'>")
        for k, v in meta.items():
            parts.append("<div class='card'>")
            parts.append(f"<div class='muted'>{k}</div>")
            parts.append(f"<div>{_format_number(v)}</div>")
            parts.append("</div>")
        parts.append("</div>")

    for sec in sections:
        st = sec.get("title", "Section")
        parts.append(f"<h2>{st}</h2>")

        text = (sec.get("text") or "").strip()
        if text:
            parts.append(f"<p>{text}</p>")

        metrics = sec.get("metrics") or {}
        if metrics:
            parts.append("<h3>Metrics</h3>")
            parts.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
            for k, v in metrics.items():
                parts.append(f"<tr><td>{k}</td><td>{_format_number(v)}</td></tr>")
            parts.append("</tbody></table>")

        figs = sec.get("figures") or []
        if figs:
            parts.append("<h3>Figures</h3>")
            parts.append("<div class='grid'>")
            for f in figs:
                fp = f.get("path")
                ft = f.get("title", "Figure")
                if not fp:
                    continue
                parts.append("<div class='card'>")
                parts.append(f"<div class='muted'>{ft}</div>")
                parts.append(f"<img src='{fp}' alt='{ft}'>")
                parts.append("</div>")
            parts.append("</div>")

        tabs = sec.get("tables") or []
        if tabs:
            parts.append("<h3>Tables</h3>")
            parts.append("<ul>")
            for t in tabs:
                tp = t.get("path")
                tt = t.get("title", "Table")
                if not tp:
                    continue
                parts.append(f"<li>{tt}: <code>{tp}</code></li>")
            parts.append("</ul>")

    parts.append("</body></html>")
    return "".join(parts)


def _env(template_dir: Union[str, Path]) -> Environment:
    loader = FileSystemLoader(str(_as_path(template_dir)))
    e = Environment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters["num"] = _format_number
    return e


def render_report_html(
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
    fname = filename or f"{default_name}.html"
    out_path = out_dir / fname

    tname = template_name
    if tname is None:
        tname = f"{report_type}_report.html.j2"

    html = ""
    try:
        e = _env(template_dir)
        t = e.get_template(tname)
        payload = _to_plain(report)
        html = t.render(report=payload)
    except Exception:
        html = _default_html(report)

    out_path.write_text(html, encoding="utf-8")
    return html, out_path
