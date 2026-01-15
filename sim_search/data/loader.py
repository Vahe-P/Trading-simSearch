"""
Data Loader Abstraction for Market Similarity Search.

Provides a flexible interface for loading market data from various sources:
- Parquet files (local)
- CSV files
- APIs (Polygon, Databento) - to be implemented in Milestone 3.2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union, Literal
import os
import time
import requests
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    
    # Source type
    source_type: Literal['parquet', 'csv', 'polygon'] = 'parquet'
    
    # File-based sources
    file_path: Optional[str] = None
    
    # API-based sources (for future use)
    symbol: Optional[str] = None
    start_date: Optional[Union[str, date, datetime]] = None
    end_date: Optional[Union[str, date, datetime]] = None
    
    # API settings
    api_key: Optional[str] = None
    limit: int = 50000
    cache_dir: str = 'data/cache'
    
    # Data options
    resample: Optional[str] = None  # e.g., '5min', '1h'
    columns: list[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    timezone: str = 'America/New_York'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.source_type in ('parquet', 'csv') and not self.file_path:
            raise ValueError(f"file_path required for source_type='{self.source_type}'")
        if self.source_type == 'polygon' and not self.symbol:
            raise ValueError(f"symbol required for source_type='{self.source_type}'")


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data according to configuration.
        
        Returns
        -------
        pd.DataFrame
            OHLCV data with datetime index and columns: open, high, low, close, volume
        """
        pass
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and standardize loaded dataframe."""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            else:
                raise ValueError("DataFrame must have datetime index or 'timestamp'/'date' column")
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Sort by index
        df = df.sort_index()
        
        return df
    
    def _apply_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply resampling if configured."""
        if self.config.resample:
            from ..times import resample
            df = resample(df, self.config.resample)
            logger.info(f"Resampled to {self.config.resample}: {len(df)} rows")
        return df


class ParquetLoader(DataLoader):
    """Load data from Parquet files."""
    
    def load(self) -> pd.DataFrame:
        """Load data from parquet file."""
        path = Path(self.config.file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        
        logger.info(f"Loading parquet: {path}")
        df = pd.read_parquet(path)
        
        # Validate and standardize
        df = self._validate_dataframe(df)
        
        # Apply date filtering if specified
        if self.config.start_date:
            start = pd.Timestamp(self.config.start_date)
            df = df[df.index >= start]
        if self.config.end_date:
            end = pd.Timestamp(self.config.end_date)
            df = df[df.index <= end]
        
        # Apply resampling if configured
        df = self._apply_resample(df)
        
        logger.info(f"Loaded {len(df)} rows from {path.name}")
        return df


class CSVLoader(DataLoader):
    """Load data from CSV files."""
    
    def load(self) -> pd.DataFrame:
        """Load data from CSV file."""
        path = Path(self.config.file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        logger.info(f"Loading CSV: {path}")
        
        # Try to detect datetime column
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        
        # Validate and standardize
        df = self._validate_dataframe(df)
        
        # Apply date filtering if specified
        if self.config.start_date:
            start = pd.Timestamp(self.config.start_date)
            df = df[df.index >= start]
        if self.config.end_date:
            end = pd.Timestamp(self.config.end_date)
            df = df[df.index <= end]
        
        # Apply resampling if configured
        df = self._apply_resample(df)
        
        logger.info(f"Loaded {len(df)} rows from {path.name}")
        return df


class PolygonLoader(DataLoader):
    """
    Load data from Polygon.io API.
    
    Features:
    - Automatic pagination
    - Local caching (parquet)
    - Rate limiting handling
    - Auto-converts futures symbols (NQ -> index proxy)
    
    Symbol formats:
    - Stocks: SPY, AAPL, QQQ
    - ETFs: SPY, QQQ, IWM
    - Indices: Use I: prefix (I:SPX, I:NDX, I:DJI)
    - Futures: NQ, ES -> auto-converted to ETF proxy (QQQ, SPY)
    """
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    # Map futures symbols to liquid ETF proxies
    FUTURES_TO_ETF = {
        'NQ': 'QQQ',    # Nasdaq 100 -> QQQ ETF
        'ES': 'SPY',    # S&P 500 -> SPY ETF
        'YM': 'DIA',    # Dow -> DIA ETF
        'RTY': 'IWM',   # Russell 2000 -> IWM ETF
        'CL': 'USO',    # Crude Oil -> USO ETF
        'GC': 'GLD',    # Gold -> GLD ETF
        'SI': 'SLV',    # Silver -> SLV ETF
        'ZN': 'TLT',    # 10Y Treasury -> TLT ETF
    }
    
    def load(self) -> pd.DataFrame:
        """Fetch data from Polygon API or cache."""
        symbol = self.config.symbol.upper()
        
        # Convert futures symbols to ETF proxies
        original_symbol = symbol
        if symbol in self.FUTURES_TO_ETF:
            symbol = self.FUTURES_TO_ETF[symbol]
            logger.info(f"Futures {original_symbol} -> using ETF proxy {symbol}")
        # Default to last 2 years if not specified
        end = pd.Timestamp(self.config.end_date) if self.config.end_date else pd.Timestamp.now()
        start = pd.Timestamp(self.config.start_date) if self.config.start_date else (end - pd.Timedelta(days=730))
        
        # Check cache first
        cache_path = Path(self.config.cache_dir) / f"{symbol}_{start.date()}_{end.date()}.parquet"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return self._finalize(df)
        
        # Fetch from API
        api_key = self.config.api_key or os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("POLYGON_API_KEY not found in config or environment variables")
        
        logger.info(f"Fetching {symbol} from Polygon.io ({start.date()} to {end.date()})")
        
        # Convert resolution (e.g., '1min' -> multiplier=1, timespan='minute')
        multiplier = 1
        timespan = 'minute'
        if self.config.resample:
            # Simple parsing: '5min' -> 5, 'minute'
            if 'min' in self.config.resample:
                multiplier = int(self.config.resample.replace('min', ''))
                timespan = 'minute'
            elif 'h' in self.config.resample:
                multiplier = int(self.config.resample.replace('h', ''))
                timespan = 'hour'
            elif 'd' in self.config.resample:
                multiplier = int(self.config.resample.replace('d', ''))
                timespan = 'day'
        
        url = f"{self.BASE_URL}/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key
        }
        
        all_results = []
        while url:
            try:
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                
                if "results" in data:
                    all_results.extend(data["results"])
                    logger.info(f"Fetched {len(data['results'])} bars...")
                
                # Check for next page
                if "next_url" in data:
                    url = data["next_url"]
                    params = {"apiKey": api_key}  # next_url usually has params embedded, but need key
                    # Rate limit sleep (free tier handles 5req/min, modify as needed)
                    time.sleep(0.25)
                else:
                    url = None
                    
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        if not all_results:
            raise ValueError(f"No data found for {symbol}")
            
        # Parse to DataFrame
        df = pd.DataFrame(all_results)
        # Rename columns: v, vw, o, c, h, l, t, n
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp').sort_index()
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info(f"Cached data to {cache_path}")
        
        return self._finalize(df)

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, filter and resample."""
        df = self._validate_dataframe(df)
        
        # Apply date filters (again, to be precise)
        if self.config.start_date:
            df = df[df.index >= pd.Timestamp(self.config.start_date, tz=df.index.tz)]
        if self.config.end_date:
            df = df[df.index <= pd.Timestamp(self.config.end_date, tz=df.index.tz)]
            
        return df


def create_loader(config: DataLoaderConfig) -> DataLoader:
    """
    Factory function to create appropriate data loader.
    
    Parameters
    ----------
    config : DataLoaderConfig
        Configuration specifying source type and options
    
    Returns
    -------
    DataLoader
        Appropriate loader instance
    
    Example
    -------
    >>> config = DataLoaderConfig(source_type='parquet', file_path='data/NQ.parquet')
    >>> loader = create_loader(config)
    >>> df = loader.load()
    
    >>> # From Polygon.io
    >>> config = DataLoaderConfig(source_type='polygon', symbol='NQ', start_date='2024-01-01')
    >>> df = create_loader(config).load()
    """
    loaders = {
        'parquet': ParquetLoader,
        'csv': CSVLoader,
        'polygon': PolygonLoader,
    }
    
    if config.source_type not in loaders:
        raise ValueError(f"Unknown source_type: {config.source_type}. Available: {list(loaders.keys())}")
    
    return loaders[config.source_type](config)


def load_market_data(
    file_path: Optional[str] = None,
    source_type: str = 'parquet',
    symbol: Optional[str] = None,
    resample: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load market data.
    
    Parameters
    ----------
    file_path : str, optional
        Path to data file (for parquet/csv)
    source_type : str
        Type of data source ('parquet', 'csv', 'polygon', 'databento')
    symbol : str, optional
        Market symbol (for API sources)
    resample : str, optional
        Resample frequency (e.g., '5min')
    start_date : str, optional
        Start date filter
    end_date : str, optional
        End date filter
    
    Returns
    -------
    pd.DataFrame
        OHLCV data
    
    Example
    -------
    >>> df = load_market_data('data/NQ.parquet', resample='5min')
    >>> df = load_market_data(symbol='NQ', source_type='polygon', start_date='2024-01-01')
    """
    config = DataLoaderConfig(
        source_type=source_type,
        file_path=file_path,
        symbol=symbol,
        resample=resample,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    loader = create_loader(config)
    return loader.load()
