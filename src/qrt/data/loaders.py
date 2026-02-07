from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def _to_ts(x: Optional[Union[str, date, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, date):
        return pd.Timestamp(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none" or s.lower() == "null":
        return None
    return pd.Timestamp(s)


def _biz_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    s = start.normalize()
    e = end.normalize()
    if e < s:
        raise ValueError("end must be >= start")
    return pd.date_range(s, e, freq="B")


def load_prices_many(
    tickers: Sequence[str],
    *,
    start: Optional[Union[str, date, pd.Timestamp]] = None,
    end: Optional[Union[str, date, pd.Timestamp]] = None,
    field: str = "Adj Close",
    cache: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    names = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if len(names) == 0:
        raise ValueError("No tickers provided")

    s = _to_ts(start)
    e = _to_ts(end) or pd.Timestamp.today().normalize()
    if s is None:
        s = e - pd.Timedelta(days=3650)

    df = yf.download(
        tickers=" ".join(names),
        start=s.date().isoformat(),
        end=(e + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(str(x) for x in df.columns.get_level_values(0))
        lvl1 = set(str(x) for x in df.columns.get_level_values(1))

        if field in lvl0:
            out = df[field].copy()
        elif field in lvl1:
            out = df.xs(field, level=1, axis=1).copy()
        else:
            raise ValueError(f"Field '{field}' not found. Available: {sorted(lvl0 | lvl1)}")
    else:
        cols = [str(c) for c in df.columns]
        if field in df.columns:
            out = df[[field]].copy()
            out.columns = [names[0]] if len(names) == 1 else out.columns
        else:
            if set(cols) == set(names):
                out = df.copy()
            else:
                raise ValueError(f"Field '{field}' not found. Available: {sorted(cols)}")

    out = out.dropna(how="all")
    out.columns = [str(c).upper() for c in out.columns]
    if len(out) == 0:
        raise ValueError("No price data returned")
    return out



def load_risk_free(
    *,
    start: Optional[Union[str, date, pd.Timestamp]] = None,
    end: Optional[Union[str, date, pd.Timestamp]] = None,
    annual_rate: float = 0.02,
    trading_days: int = 252,
    fred_series: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    s = _to_ts(start)
    e = _to_ts(end) or pd.Timestamp.today().normalize()
    if s is None:
        s = e - pd.Timedelta(days=3650)

    idx = _biz_index(s, e)
    daily = (1.0 + float(annual_rate)) ** (1.0 / float(trading_days)) - 1.0
    return pd.Series(daily, index=idx, name="RF")


def _ff_zip_url(kind: str) -> str:
    k = str(kind).lower().strip()
    if k in {"ff5", "5", "five"}:
        return "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    if k in {"ff3", "3", "three"}:
        return "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    raise ValueError("kind must be 'ff3' or 'ff5'")


def _read_ff_daily_from_zip(content: bytes) -> pd.DataFrame:
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(content))
    names = z.namelist()
    if len(names) == 0:
        raise ValueError("Empty factors zip")

    raw = z.read(names[0]).decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines() if ln is not None]

    first_data_i = None
    for i, ln in enumerate(lines):
        if "," not in ln:
            continue
        head = ln.split(",", 1)[0].strip()
        if head.isdigit() and len(head) == 8:
            first_data_i = i
            break
    if first_data_i is None:
        raise ValueError("Could not find any daily factor rows")

    header_i = first_data_i - 1 if first_data_i - 1 >= 0 else None
    header = lines[header_i] if header_i is not None else ""
    header_parts = [p.strip() for p in header.split(",")] if "," in header else []

    data_rows = []
    for ln in lines[first_data_i:]:
        if "," not in ln:
            break
        head = ln.split(",", 1)[0].strip()
        if not (head.isdigit() and len(head) == 8):
            break
        data_rows.append(ln)

    if len(data_rows) == 0:
        raise ValueError("No daily factor rows found")

    if len(header_parts) >= 2 and header_parts[0].lower() in {"date", "dates"}:
        csv_text = "\n".join([lines[header_i]] + data_rows)
        df = pd.read_csv(io.StringIO(csv_text), sep=",", engine="python")
    else:
        first_parts = [p.strip() for p in data_rows[0].split(",")]
        ncols = len(first_parts)
        cols = ["Date"] + [f"F{i}" for i in range(1, ncols)]
        csv_text = "\n".join([",".join(cols)] + data_rows)
        df = pd.read_csv(io.StringIO(csv_text), sep=",", engine="python")

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0

    df.index.name = "Date"
    df.columns = [str(c).strip().upper().replace("-", "_") for c in df.columns]

    if "MKT_RF" not in df.columns and "MKT-RF" in df.columns:
        df = df.rename(columns={"MKT-RF": "MKT_RF"})

    return df.dropna(how="all")




def load_fama_french_factors(
    *,
    start: Optional[Union[str, date, pd.Timestamp]] = None,
    end: Optional[Union[str, date, pd.Timestamp]] = None,
    kind: str = "ff5",
    cache: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    s = _to_ts(start)
    e = _to_ts(end) or pd.Timestamp.today().normalize()
    if s is None:
        s = e - pd.Timedelta(days=3650)

    url = _ff_zip_url(kind)
    r = requests.get(url, timeout=(10, 60))
    r.raise_for_status()
    df = _read_ff_daily_from_zip(r.content)

    if all(c.startswith("F") for c in df.columns):
        if str(kind).lower().strip() == "ff5" and len(df.columns) >= 6:
            df.columns = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"][: len(df.columns)]
        if str(kind).lower().strip() == "ff3" and len(df.columns) >= 4:
            df.columns = ["MKT_RF", "SMB", "HML", "RF"][: len(df.columns)]


    df = df.loc[(df.index >= s) & (df.index <= e)]
    want = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    cols = [c for c in want if c in df.columns]
    if "RF" not in cols:
        df["RF"] = 0.0
        cols = cols + ["RF"]
    out = df[cols].copy()
    if len(out) == 0:
        raise ValueError("No factors after date filtering")
    return out


def load_inputs_for_asset(
    ticker: str,
    cfg: Any,
    *,
    start: Optional[Union[str, date, pd.Timestamp]] = None,
    end: Optional[Union[str, date, pd.Timestamp]] = None,
) -> Dict[str, Any]:
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("ticker is empty")

    data_cfg = getattr(cfg, "data", cfg)
    s = start if start is not None else getattr(data_cfg, "start", None)
    e = end if end is not None else getattr(data_cfg, "end", None)

    field = getattr(data_cfg, "price_field", "Adj Close")
    bench = getattr(data_cfg, "benchmark_ticker", "SPY")
    rf_annual = float(getattr(data_cfg, "risk_free_annual", 0.02))
    td = int(getattr(data_cfg, "trading_days", 252))
    cache = getattr(data_cfg, "cache", None)

    px = load_prices_many([t], start=s, end=e, field=str(field), cache=cache)[t].rename(t)
    bpx = load_prices_many([bench], start=s, end=e, field=str(field), cache=cache)[str(bench).upper()].rename(str(bench).upper())

    factors = None
    try:
        factors = load_fama_french_factors(start=s, end=e, kind="ff5", cache=cache)
    except Exception:
        factors = None

    rf = load_risk_free(start=s, end=e, annual_rate=rf_annual, trading_days=td, fred_series=None, cache=cache)

    return {
        "prices": px,
        "benchmark_prices": bpx,
        "factors": factors,
        "risk_free": rf,
    }
