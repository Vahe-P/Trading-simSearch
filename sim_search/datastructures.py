"""
Data structures for similarity search.

Provides proper abstractions to avoid dealing with parallel arrays.
All related window data is bundled together in WindowData.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class WindowData:
    """
    Single window with all associated data bundled together.
    
    This replaces the pattern of passing parallel arrays (x, y, labels, regimes...)
    that must stay aligned. Now everything is together in one object.
    
    Attributes
    ----------
    idx : int
        Index in the original dataset (for reference back)
    x : np.ndarray
        Normalized pattern data, shape (n_features, n_timepoints)
    y : np.ndarray
        Future returns (horizon), shape (horizon_len,)
    cutoff : pd.Timestamp
        Forecast origin / cutoff time
    interval : pd.Interval
        Original time interval for this window
        
    # Volatility & Regime
    volatility : float
        Realized volatility (e.g., Garman-Klass)
    regime : int
        Volatility regime: 0=LOW, 1=MEDIUM, 2=HIGH, -1=unclassified
        
    # Calendar Features
    days_since_fomc : int
        Trading days since last FOMC meeting (-1 if unknown)
    is_fomc_day : bool
        Whether this window includes an FOMC announcement
    is_cpi_day : bool
        Whether this window includes CPI release
    is_nfp_day : bool
        Whether this window includes Non-Farm Payrolls
    has_red_folder : bool
        Whether any high-impact event occurred in this window
        
    # Metadata
    metadata : dict
        Any additional features or info
    """
    idx: int
    x: np.ndarray
    y: np.ndarray
    cutoff: pd.Timestamp
    interval: pd.Interval
    
    # Volatility
    volatility: float = 0.0
    regime: int = -1  # -1 = unclassified
    
    # Calendar features
    days_since_fomc: int = -1
    is_fomc_day: bool = False
    is_cpi_day: bool = False
    is_nfp_day: bool = False
    has_red_folder: bool = False
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def regime_name(self) -> str:
        """Human-readable regime name."""
        names = {-1: 'UNKNOWN', 0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        return names.get(self.regime, 'UNKNOWN')
    
    def matches_regime(self, other: 'WindowData') -> bool:
        """Check if this window is in the same regime as another."""
        if self.regime == -1 or other.regime == -1:
            return True  # Unknown regime matches everything
        return self.regime == other.regime
    
    def matches_calendar_context(self, other: 'WindowData', 
                                  match_fomc: bool = True,
                                  match_red_folder: bool = False) -> bool:
        """
        Check if calendar context matches.
        
        Parameters
        ----------
        other : WindowData
            Window to compare against
        match_fomc : bool
            If True, FOMC days only match other FOMC days
        match_red_folder : bool
            If True, red folder events only match other red folder events
        """
        if match_fomc and self.is_fomc_day != other.is_fomc_day:
            return False
        if match_red_folder and self.has_red_folder != other.has_red_folder:
            return False
        return True


@dataclass
class WindowCollection:
    """
    Collection of windows with utilities for filtering and slicing.
    
    This replaces parallel lists/arrays. All windows are stored together
    and can be filtered, split, or transformed as a unit.
    """
    windows: List[WindowData] = field(default_factory=list)
    
    # Cached arrays for efficient KNN (built lazily)
    _x_array: Optional[np.ndarray] = field(default=None, repr=False)
    _y_array: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx) -> WindowData:
        if isinstance(idx, (list, np.ndarray)):
            # Index by list of indices
            return WindowCollection([self.windows[i] for i in idx])
        return self.windows[idx]
    
    def __iter__(self):
        return iter(self.windows)
    
    def append(self, window: WindowData):
        """Add a window, invalidating caches."""
        self.windows.append(window)
        self._x_array = None
        self._y_array = None
    
    @property
    def x_array(self) -> np.ndarray:
        """Get X data as numpy array, shape (n_windows, n_features, n_timepoints).
        
        Handles windows with different lengths by padding shorter ones with zeros.
        """
        if self._x_array is None and self.windows:
            # Find max length
            max_len = max(w.x.shape[-1] for w in self.windows)
            
            # Pad all to same length
            padded = []
            for w in self.windows:
                if w.x.shape[-1] < max_len:
                    pad_width = max_len - w.x.shape[-1]
                    # Pad at the beginning (older data gets zeros)
                    padded_x = np.pad(w.x, ((0, 0), (pad_width, 0)), mode='constant', constant_values=0)
                else:
                    padded_x = w.x
                padded.append(padded_x)
            
            self._x_array = np.stack(padded, axis=0)
        return self._x_array
    
    @property
    def y_array(self) -> np.ndarray:
        """Get Y data as numpy array, shape (n_windows, horizon_len)."""
        if self._y_array is None and self.windows:
            self._y_array = np.stack([w.y for w in self.windows], axis=0)
        return self._y_array
    
    @property
    def cutoffs(self) -> List[pd.Timestamp]:
        """Get all cutoff timestamps."""
        return [w.cutoff for w in self.windows]
    
    @property
    def regimes(self) -> np.ndarray:
        """Get regime labels as array."""
        return np.array([w.regime for w in self.windows])
    
    @property
    def volatilities(self) -> np.ndarray:
        """Get volatilities as array."""
        return np.array([w.volatility for w in self.windows])
    
    def filter_by_regime(self, target_regime: int) -> 'WindowCollection':
        """Return new collection with only windows matching target regime."""
        filtered = [w for w in self.windows if w.regime == target_regime]
        return WindowCollection(filtered)
    
    def filter_by_indices(self, indices: List[int]) -> 'WindowCollection':
        """Return new collection with windows at specified indices."""
        filtered = [self.windows[i] for i in indices]
        return WindowCollection(filtered)
    
    def exclude_red_folder(self) -> 'WindowCollection':
        """Return new collection excluding high-impact event windows."""
        filtered = [w for w in self.windows if not w.has_red_folder]
        return WindowCollection(filtered)
    
    def only_red_folder(self) -> 'WindowCollection':
        """Return new collection with only high-impact event windows."""
        filtered = [w for w in self.windows if w.has_red_folder]
        return WindowCollection(filtered)
    
    def split_train_test(self, test_idx: int = -1) -> tuple['WindowCollection', WindowData]:
        """
        Split into training collection and single test window.
        
        IMPORTANT: Regime classification is done HERE using ONLY training data
        to avoid data leakage / look-ahead bias.
        
        Parameters
        ----------
        test_idx : int
            Index of test window (default: -1 = last window)
            
        Returns
        -------
        tuple
            (train_collection, test_window)
        """
        if test_idx == -1:
            test_idx = len(self.windows) - 1
        
        train_windows = [w for i, w in enumerate(self.windows) if i < test_idx]
        test_window = self.windows[test_idx]
        
        train_collection = WindowCollection(train_windows)
        
        # Classify regimes using ONLY training data (no look-ahead bias)
        vol_method = getattr(self, '_vol_method', None)
        if vol_method and len(train_collection) > 10:
            train_collection._classify_regimes_no_leakage(test_window)
        
        return train_collection, test_window
    
    def _classify_regimes_no_leakage(self, test_window: 'WindowData'):
        """
        Classify regimes using ONLY training data (this collection).
        
        This avoids data leakage by:
        1. Computing thresholds from training volatilities only
        2. Classifying both train AND test windows with those thresholds
        """
        from .volatility import compute_regime_thresholds, classify_regime, REGIME_NAMES
        
        # Get training volatilities only
        train_vols = np.array([w.volatility for w in self.windows])
        valid_vols = train_vols[~np.isnan(train_vols) & (train_vols > 0)]
        
        if len(valid_vols) < 10:
            return
        
        # Compute thresholds from TRAINING data only
        thresholds = compute_regime_thresholds(valid_vols)
        
        # Classify training windows
        for w in self.windows:
            w.regime = classify_regime(w.volatility, thresholds)
        
        # Classify test window using SAME thresholds (no leakage)
        test_window.regime = classify_regime(test_window.volatility, thresholds)
        
        # Log for verification
        from loguru import logger
        regime_counts = {0: 0, 1: 0, 2: 0}
        for w in self.windows:
            regime_counts[w.regime] = regime_counts.get(w.regime, 0) + 1
        logger.info(f"Regime distribution (NO LEAKAGE): " + 
                   " | ".join(f"{REGIME_NAMES[k]}: {v} ({v/len(self.windows)*100:.1f}%)" 
                             for k, v in regime_counts.items()))
    
    def get_same_regime_indices(self, query: WindowData) -> np.ndarray:
        """Get indices of windows in same regime as query."""
        return np.array([i for i, w in enumerate(self.windows) 
                        if w.regime == query.regime])
    
    def get_matching_indices(self, query: WindowData,
                             match_regime: bool = True,
                             match_fomc: bool = False,
                             exclude_red_folder: bool = False) -> np.ndarray:
        """
        Get indices of windows matching query based on filters.
        
        Parameters
        ----------
        query : WindowData
            Query window to match against
        match_regime : bool
            Only include same volatility regime
        match_fomc : bool
            Only include same FOMC context (FOMC day matches FOMC day)
        exclude_red_folder : bool
            Exclude all high-impact event windows
            
        Returns
        -------
        np.ndarray
            Indices of matching windows
        """
        indices = []
        for i, w in enumerate(self.windows):
            # Regime filter
            if match_regime and w.regime != query.regime:
                continue
            
            # FOMC context filter
            if match_fomc and w.is_fomc_day != query.is_fomc_day:
                continue
            
            # Exclude red folder events
            if exclude_red_folder and w.has_red_folder:
                continue
            
            indices.append(i)
        
        return np.array(indices)
    
    def summary(self) -> dict:
        """Get summary statistics of the collection."""
        if not self.windows:
            return {'count': 0}
        
        regimes = self.regimes
        return {
            'count': len(self.windows),
            'regime_counts': {
                'LOW': np.sum(regimes == 0),
                'MEDIUM': np.sum(regimes == 1),
                'HIGH': np.sum(regimes == 2),
                'UNKNOWN': np.sum(regimes == -1)
            },
            'fomc_days': sum(1 for w in self.windows if w.is_fomc_day),
            'red_folder_days': sum(1 for w in self.windows if w.has_red_folder),
            'date_range': (self.windows[0].cutoff, self.windows[-1].cutoff),
            'mean_volatility': np.mean(self.volatilities),
        }
