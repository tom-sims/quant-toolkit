from pathlib import Path

import pandas as pd

from .cache import DiskCache, cached_dataframe


class LoaderConfig:
    def __init__(self, cache_dir=None, use_cache=True):
        if cache_dir is None:
            cache_dir = Path.home() / ".quant_research_toolkit" / "cache"
        self.cache_dir = Path(cache_dir)
        self.use_cache = bool(use_cache)


def _ensure_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_csv_timeseries(path, date_col="Date"):
    """
    Load a CSV with a date column into a DataFrame indexed by datetime.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError("CSV missing required date column '" + date_col + "'. Found: " + str(list(df.columns)))
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def load_fred_series(series_id, start=None, end=None):
    """
    Load a single FRED series into a DataFrame with one column named series_id.
    Requires pandas-datareader.
    """
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise ImportError("pandas-datareader is required for FRED loading. Install it via requirements.txt.") from e

    s = pdr.DataReader(series_id, "fred", start=start, end=end)
    s = _ensure_datetime_index(s)
    s.columns = [series_id]
    return s


def load_prices_yfinance(tickers, start, end, auto_adjust=True, group_by="column", config=None):
    """
    Download OHLCV prices from yfinance.
    """
    if config is None:
        config = LoaderConfig()

    cache = DiskCache(config.cache_dir)

    def _download():
        try:
            import yfinance as yf
        except Exception as e:
            raise ImportError("yfinance is required for price loading. Install it via requirements.txt.") from e

        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            group_by=group_by,
            progress=False,
        )
        if df is None or len(df) == 0:
            raise ValueError("No data returned from yfinance. Check tickers/date range.")
        return _ensure_datetime_index(df)

    if not config.use_cache:
        return _download()

    cached_download = cached_dataframe(cache)(_download)
    return cached_download()
