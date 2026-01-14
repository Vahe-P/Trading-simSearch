"""
Pluggable pre-filters for similarity search.

All filters follow sklearn's Transformer API:
- fit(X, y=None) - learn parameters from training data
- transform(X) - filter/transform data
- fit_transform(X, y=None) - fit then transform

Filters can be chained using FilterPipeline or sklearn's Pipeline.

Usage:
    # Single filter
    regime_filter = RegimeFilter(enabled=True, vol_method='garman_klass')
    regime_filter.fit(train_collection)
    filtered = regime_filter.transform(train_collection, query=test_window)
    
    # Multiple filters
    pipeline = FilterPipeline([
        RegimeFilter(enabled=True),
        CalendarFilter(exclude_fomc=False, match_fomc_context=True),
    ])
    pipeline.fit(train_collection)
    filtered = pipeline.transform(train_collection, query=test_window)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from datetime import datetime, date
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

from .datastructures import WindowData, WindowCollection
from .volatility import (
    compute_all_window_volatilities,
    compute_regime_thresholds,
    classify_regime,
    classify_all_regimes,
    REGIME_NAMES
)


class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for similarity search filters.
    
    All filters inherit from sklearn's BaseEstimator and TransformerMixin
    for compatibility with sklearn Pipeline.
    
    Subclasses must implement:
    - _fit_impl(collection) - learn parameters from training data
    - _filter_indices(collection, query) - return indices to keep
    """
    
    def __init__(self, enabled: bool = True):
        """
        Parameters
        ----------
        enabled : bool
            If False, filter is a no-op (returns all indices)
        """
        self.enabled = enabled
    
    def fit(self, X: WindowCollection, y=None):
        """
        Fit filter on training data.
        
        Parameters
        ----------
        X : WindowCollection
            Training windows
        y : ignored
            For sklearn compatibility
            
        Returns
        -------
        self
        """
        if self.enabled:
            self._fit_impl(X)
        return self
    
    def transform(self, X: WindowCollection, query: Optional[WindowData] = None) -> np.ndarray:
        """
        Get indices of windows that pass the filter.
        
        Parameters
        ----------
        X : WindowCollection
            Windows to filter
        query : WindowData, optional
            Query window (for context-dependent filtering like regime matching)
            
        Returns
        -------
        np.ndarray
            Indices of windows that pass the filter
        """
        if not self.enabled:
            return np.arange(len(X))
        
        return self._filter_indices(X, query)
    
    @abstractmethod
    def _fit_impl(self, collection: WindowCollection):
        """Implement fitting logic. Override in subclass."""
        pass
    
    @abstractmethod
    def _filter_indices(self, collection: WindowCollection, 
                        query: Optional[WindowData]) -> np.ndarray:
        """Return indices that pass filter. Override in subclass."""
        pass
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {'enabled': self.enabled}
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RegimeFilter(BaseFilter):
    """
    Filter windows by volatility regime.
    
    Only returns windows in the same volatility regime as the query.
    Regimes are determined by Garman-Klass (or Parkinson) volatility
    with percentile-based thresholds.
    
    Parameters
    ----------
    enabled : bool
        Enable/disable filter (default: True)
    vol_method : str
        Volatility estimator: 'garman_klass' or 'parkinson'
    low_percentile : float
        Percentile for LOW/MEDIUM threshold (default: 33)
    high_percentile : float
        Percentile for MEDIUM/HIGH threshold (default: 67)
    min_same_regime : int
        Minimum windows required in same regime before fallback (default: 5)
        
    Example
    -------
    >>> filter = RegimeFilter(enabled=True, vol_method='garman_klass')
    >>> filter.fit(train_collection)
    >>> indices = filter.transform(train_collection, query=test_window)
    """
    
    def __init__(self, 
                 enabled: bool = True,
                 vol_method: str = 'garman_klass',
                 low_percentile: float = 33.0,
                 high_percentile: float = 67.0,
                 min_same_regime: int = 5):
        super().__init__(enabled=enabled)
        self.vol_method = vol_method
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.min_same_regime = min_same_regime
        
        # Fitted attributes
        self.thresholds_: Optional[Tuple[float, float]] = None
    
    def _fit_impl(self, collection: WindowCollection):
        """Compute regime thresholds from training volatilities."""
        vols = collection.volatilities
        
        # Filter out zeros/NaNs
        valid_vols = vols[~np.isnan(vols) & (vols > 0)]
        
        if len(valid_vols) < 10:
            logger.warning("Not enough valid volatilities for regime classification")
            self.thresholds_ = (0.0, float('inf'))
            return
        
        self.thresholds_ = compute_regime_thresholds(
            valid_vols,
            method='percentile',
            low_percentile=self.low_percentile,
            high_percentile=self.high_percentile
        )
        
        logger.debug(f"RegimeFilter fitted: thresholds={self.thresholds_}")
    
    def _filter_indices(self, collection: WindowCollection, 
                        query: Optional[WindowData]) -> np.ndarray:
        """Return indices of windows in same regime as query."""
        if query is None:
            logger.warning("RegimeFilter: No query provided, returning all indices")
            return np.arange(len(collection))
        
        if self.thresholds_ is None:
            raise ValueError("RegimeFilter not fitted. Call fit() first.")
        
        # Classify query regime
        query_regime = classify_regime(query.volatility, self.thresholds_)
        
        # Get same-regime indices
        same_regime_idx = []
        for i, w in enumerate(collection):
            w_regime = classify_regime(w.volatility, self.thresholds_)
            if w_regime == query_regime:
                same_regime_idx.append(i)
        
        same_regime_idx = np.array(same_regime_idx)
        
        # Fallback if not enough windows
        if len(same_regime_idx) < self.min_same_regime:
            logger.warning(
                f"Only {len(same_regime_idx)} windows in regime {REGIME_NAMES[query_regime]}, "
                f"falling back to all {len(collection)} windows"
            )
            return np.arange(len(collection))
        
        logger.debug(
            f"RegimeFilter: Query regime={REGIME_NAMES[query_regime]}, "
            f"matched {len(same_regime_idx)}/{len(collection)} windows"
        )
        
        return same_regime_idx
    
    def classify_window(self, window: WindowData) -> int:
        """Classify a single window's regime."""
        if self.thresholds_ is None:
            return -1
        return classify_regime(window.volatility, self.thresholds_)


class CalendarFilter(BaseFilter):
    """
    Filter windows by calendar/news events (FOMC, CPI, NFP, etc.)
    
    Supports multiple modes:
    - match_fomc_context: FOMC days only match other FOMC days
    - exclude_red_folder: Remove all high-impact event windows
    - days_since_range: Only match windows within N days of similar events
    
    Parameters
    ----------
    enabled : bool
        Enable/disable filter (default: True)
    match_fomc_context : bool
        If True, FOMC days only match other FOMC days (default: True)
    exclude_red_folder : bool
        If True, exclude all high-impact event windows (default: False)
    match_red_folder_context : bool  
        If True, red folder days only match other red folder days (default: False)
    days_since_fomc_tolerance : int
        If set, only match windows within this many days of FOMC (default: None)
        
    Example
    -------
    >>> # FOMC days should match other FOMC days
    >>> filter = CalendarFilter(match_fomc_context=True)
    >>> filter.fit(train_collection)
    >>> indices = filter.transform(train_collection, query=test_window)
    
    >>> # Exclude volatile event days entirely  
    >>> filter = CalendarFilter(exclude_red_folder=True)
    """
    
    def __init__(self,
                 enabled: bool = True,
                 match_fomc_context: bool = True,
                 exclude_red_folder: bool = False,
                 match_red_folder_context: bool = False,
                 days_since_fomc_tolerance: Optional[int] = None):
        super().__init__(enabled=enabled)
        self.match_fomc_context = match_fomc_context
        self.exclude_red_folder = exclude_red_folder
        self.match_red_folder_context = match_red_folder_context
        self.days_since_fomc_tolerance = days_since_fomc_tolerance
    
    def _fit_impl(self, collection: WindowCollection):
        """Calendar filter doesn't need fitting - just validates data."""
        # Check if calendar data is populated
        has_calendar = any(w.is_fomc_day or w.has_red_folder 
                          for w in collection)
        if not has_calendar:
            logger.warning(
                "CalendarFilter: No calendar data found in windows. "
                "Did you run enrich_with_calendar()?"
            )
    
    def _filter_indices(self, collection: WindowCollection,
                        query: Optional[WindowData]) -> np.ndarray:
        """Return indices matching calendar context."""
        indices = []
        
        for i, w in enumerate(collection):
            # Exclude red folder events entirely
            if self.exclude_red_folder and w.has_red_folder:
                continue
            
            # Match FOMC context (FOMC day matches FOMC day)
            if query and self.match_fomc_context:
                if w.is_fomc_day != query.is_fomc_day:
                    continue
            
            # Match red folder context
            if query and self.match_red_folder_context:
                if w.has_red_folder != query.has_red_folder:
                    continue
            
            # Days since FOMC tolerance
            if query and self.days_since_fomc_tolerance is not None:
                if query.days_since_fomc >= 0 and w.days_since_fomc >= 0:
                    diff = abs(w.days_since_fomc - query.days_since_fomc)
                    if diff > self.days_since_fomc_tolerance:
                        continue
            
            indices.append(i)
        
        result = np.array(indices) if indices else np.arange(len(collection))
        
        logger.debug(f"CalendarFilter: {len(result)}/{len(collection)} windows passed")
        return result


class FilterPipeline(BaseFilter):
    """
    Chain multiple filters together.
    
    Filters are applied sequentially - the output indices of one filter
    become the input to the next. The final result is the intersection
    of all filters.
    
    Parameters
    ----------
    filters : list of BaseFilter
        Filters to apply in order
    enabled : bool
        Enable/disable entire pipeline (default: True)
        
    Example
    -------
    >>> pipeline = FilterPipeline([
    ...     RegimeFilter(enabled=True),
    ...     CalendarFilter(match_fomc_context=True),
    ... ])
    >>> pipeline.fit(train_collection)
    >>> indices = pipeline.transform(train_collection, query=test_window)
    """
    
    def __init__(self, filters: List[BaseFilter] = None, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.filters = filters or []
    
    def add_filter(self, filter: BaseFilter):
        """Add a filter to the pipeline."""
        self.filters.append(filter)
        return self
    
    def _fit_impl(self, collection: WindowCollection):
        """Fit all filters."""
        for f in self.filters:
            f.fit(collection)
    
    def _filter_indices(self, collection: WindowCollection,
                        query: Optional[WindowData]) -> np.ndarray:
        """Apply all filters sequentially, returning intersection."""
        # Start with all indices
        current_indices = np.arange(len(collection))
        
        for f in self.filters:
            if not f.enabled:
                continue
            
            # Get indices from this filter
            filter_indices = f.transform(collection, query)
            
            # Intersect with current indices
            current_indices = np.intersect1d(current_indices, filter_indices)
            
            if len(current_indices) == 0:
                logger.warning(f"FilterPipeline: {f.__class__.__name__} reduced to 0 windows")
                break
        
        logger.debug(f"FilterPipeline: {len(current_indices)}/{len(collection)} windows passed all filters")
        return current_indices
    
    def summary(self) -> dict:
        """Get summary of pipeline configuration."""
        return {
            'n_filters': len(self.filters),
            'filters': [
                {'name': f.__class__.__name__, 'enabled': f.enabled}
                for f in self.filters
            ]
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_pipeline(
    regime_filter: bool = True,
    calendar_filter: bool = True,
    vol_method: str = 'garman_klass',
    match_fomc: bool = True,
    exclude_red_folder: bool = False
) -> FilterPipeline:
    """
    Create a filter pipeline with common defaults.
    
    Parameters
    ----------
    regime_filter : bool
        Include volatility regime filter
    calendar_filter : bool
        Include calendar event filter
    vol_method : str
        Volatility method for regime filter
    match_fomc : bool
        Match FOMC context in calendar filter
    exclude_red_folder : bool
        Exclude high-impact events
        
    Returns
    -------
    FilterPipeline
        Configured pipeline
    """
    filters = []
    
    if regime_filter:
        filters.append(RegimeFilter(
            enabled=True,
            vol_method=vol_method
        ))
    
    if calendar_filter:
        filters.append(CalendarFilter(
            enabled=True,
            match_fomc_context=match_fomc,
            exclude_red_folder=exclude_red_folder
        ))
    
    return FilterPipeline(filters)
