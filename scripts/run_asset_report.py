from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_date(x: Optional[str]) -> Optional[date]:
    if x is None or str(x).strip() == "":
        return None
    return date.fromisoformat(str(x))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise ImportError("pyyaml is required to load config files") from e
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping")
    return data


def _load_report_config(cfg_path: Path):
    from qrt.config import ReportConfig

    if hasattr(ReportConfig, "from_yaml") and callable(getattr(ReportConfig, "from_yaml")):
        return ReportConfig.from_yaml(str(cfg_path))

    data = _load_yaml(cfg_path)

    if hasattr(ReportConfig, "from_dict") and callable(getattr(ReportConfig, "from_dict")):
        return ReportConfig.from_dict(data)

    return ReportConfig(**data)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_asset_report")
    parser.add_argument("ticker", type=str)
    parser.add_argument("--config", type=str, default="configs/report.yaml")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--return-kind", type=str, default="log")
    parser.add_argument("--var-levels", type=str, default="0.95,0.99")
    parser.add_argument("--var-window", type=int, default=252)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--md", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--template-dir", type=str, default=None)
    args = parser.parse_args(argv)

    root = _repo_root()
    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    cfg = _load_report_config(cfg_path)

    levels = tuple(float(x.strip()) for x in str(args.var_levels).split(",") if x.strip())
    if len(levels) == 0:
        levels = (0.95, 0.99)

    out_root = (root / args.output).resolve()
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    from qrt.reports.asset_report import build_asset_report

    report = build_asset_report(
        args.ticker,
        cfg,
        start=start,
        end=end,
        output_root=out_root,
        return_kind=str(args.return_kind),
        var_levels=levels,
        var_window=int(args.var_window),
        rolling_window=int(args.rolling_window),
    )

    template_dir = Path(args.template_dir) if args.template_dir else (root / "src" / "qrt" / "reports" / "templates")
    wrote_any = False

    if args.md:
        from qrt.reports.render_md import render_report_md

        _, md_path = render_report_md(
            report,
            template_dir=template_dir,
            output_dir=out_root / "reports",
        )
        print(str(md_path))
        wrote_any = True

    if args.html:
        from qrt.reports.render_html import render_report_html

        _, html_path = render_report_html(
            report,
            template_dir=template_dir,
            output_dir=out_root / "reports",
        )
        print(str(html_path))
        wrote_any = True

    if not wrote_any:
        print("Report built. Use --md and/or --html to render.")
        for p in getattr(report, "figure_paths", ()) or ():
            print(str(p))
        for p in getattr(report, "table_paths", ()) or ():
            print(str(p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
