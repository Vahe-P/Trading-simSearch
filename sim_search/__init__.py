"""
Similarity Search for Market Forecasting.

Main components:
- WindowData, WindowCollection: Data structures for bundling window data
- RegimeFilter, CalendarFilter, FilterPipeline: Pluggable pre-filters
- WindowCollectionBuilder: Builder for creating enriched collections
"""

from .datastructures import WindowData, WindowCollection
from .filters import (
    BaseFilter,
    RegimeFilter, 
    CalendarFilter, 
    FilterPipeline,
    create_default_pipeline
)
from .builder import WindowCollectionBuilder, build_collection_from_df
from .calendar_events import (
    is_fomc_day,
    is_cpi_day,
    is_nfp_day,
    is_red_folder_event,
    days_since_fomc,
    get_event_context,
)
from .volatility import (
    garman_klass_volatility,
    parkinson_volatility,
    REGIME_NAMES,
)

__all__ = [
    # Data structures
    'WindowData',
    'WindowCollection',
    
    # Filters
    'BaseFilter',
    'RegimeFilter',
    'CalendarFilter',
    'FilterPipeline',
    'create_default_pipeline',
    
    # Builder
    'WindowCollectionBuilder',
    'build_collection_from_df',
    
    # Calendar
    'is_fomc_day',
    'is_cpi_day',
    'is_nfp_day',
    'is_red_folder_event',
    'days_since_fomc',
    'get_event_context',
    
    # Volatility
    'garman_klass_volatility',
    'parkinson_volatility',
    'REGIME_NAMES',
]
