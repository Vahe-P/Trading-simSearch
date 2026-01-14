"""
Economic calendar and market events.

Provides FOMC dates, CPI releases, NFP dates, and other "red folder" events
for use in calendar-aware filtering.

FOMC dates are the most critical - markets behave very differently on FOMC days.
"""

from datetime import date, datetime, timedelta
from typing import List, Set, Optional, Dict
import pandas as pd
from loguru import logger


# =============================================================================
# FOMC MEETING DATES
# =============================================================================

# FOMC announcement dates (typically Wednesday at 2pm ET)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES_2024 = [
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
]

FOMC_DATES_2025 = [
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 11, 5),
    date(2025, 12, 17),
]

FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
]

ALL_FOMC_DATES: Set[date] = set(FOMC_DATES_2024 + FOMC_DATES_2025 + FOMC_DATES_2026)


# =============================================================================
# OTHER ECONOMIC EVENTS
# =============================================================================

# CPI Release dates (typically around 8:30 AM ET on release day)
# These are approximate - the exact dates vary
CPI_RELEASE_DATES_2024 = [
    date(2024, 1, 11),
    date(2024, 2, 13),
    date(2024, 3, 12),
    date(2024, 4, 10),
    date(2024, 5, 15),
    date(2024, 6, 12),
    date(2024, 7, 11),
    date(2024, 8, 14),
    date(2024, 9, 11),
    date(2024, 10, 10),
    date(2024, 11, 13),
    date(2024, 12, 11),
]

CPI_RELEASE_DATES_2025 = [
    date(2025, 1, 15),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 4, 10),
    date(2025, 5, 13),
    date(2025, 6, 11),
    date(2025, 7, 11),
    date(2025, 8, 12),
    date(2025, 9, 10),
    date(2025, 10, 10),
    date(2025, 11, 13),
    date(2025, 12, 10),
]

ALL_CPI_DATES: Set[date] = set(CPI_RELEASE_DATES_2024 + CPI_RELEASE_DATES_2025)

# Non-Farm Payrolls (first Friday of each month, 8:30 AM ET)
# We'll generate these programmatically
def get_nfp_dates(year: int) -> List[date]:
    """Get NFP dates (first Friday of each month) for a year."""
    dates = []
    for month in range(1, 13):
        # Find first Friday
        d = date(year, month, 1)
        # Days until Friday (weekday 4)
        days_until_friday = (4 - d.weekday()) % 7
        first_friday = d + timedelta(days=days_until_friday)
        dates.append(first_friday)
    return dates

ALL_NFP_DATES: Set[date] = set(get_nfp_dates(2024) + get_nfp_dates(2025) + get_nfp_dates(2026))


# =============================================================================
# EVENT LOOKUP FUNCTIONS
# =============================================================================

def is_fomc_day(dt: datetime | date | pd.Timestamp) -> bool:
    """Check if date is an FOMC announcement day."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    return dt in ALL_FOMC_DATES


def is_cpi_day(dt: datetime | date | pd.Timestamp) -> bool:
    """Check if date is a CPI release day."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    return dt in ALL_CPI_DATES


def is_nfp_day(dt: datetime | date | pd.Timestamp) -> bool:
    """Check if date is an NFP release day."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    return dt in ALL_NFP_DATES


def is_red_folder_event(dt: datetime | date | pd.Timestamp) -> bool:
    """
    Check if date has any high-impact ("red folder") event.
    
    Red folder events include:
    - FOMC announcements
    - CPI releases
    - Non-Farm Payrolls
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    
    return dt in ALL_FOMC_DATES or dt in ALL_CPI_DATES or dt in ALL_NFP_DATES


def days_since_fomc(dt: datetime | date | pd.Timestamp, 
                    trading_days_only: bool = False) -> int:
    """
    Calculate trading days since most recent FOMC meeting.
    
    Parameters
    ----------
    dt : datetime-like
        Date to check
    trading_days_only : bool
        If True, count only trading days (requires exchange_calendars)
        
    Returns
    -------
    int
        Days since FOMC (-1 if no prior FOMC found)
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    
    # Find most recent FOMC before this date
    prior_fomc = [d for d in ALL_FOMC_DATES if d < dt]
    if not prior_fomc:
        return -1
    
    last_fomc = max(prior_fomc)
    
    if trading_days_only:
        try:
            from exchange_calendars import get_calendar
            nyse = get_calendar('NYSE')
            sessions = nyse.sessions_in_range(
                pd.Timestamp(last_fomc), 
                pd.Timestamp(dt)
            )
            return len(sessions) - 1  # -1 because we don't count the FOMC day itself
        except Exception as e:
            logger.warning(f"Could not calculate trading days: {e}")
    
    return (dt - last_fomc).days


def days_until_fomc(dt: datetime | date | pd.Timestamp) -> int:
    """
    Calculate days until next FOMC meeting.
    
    Returns
    -------
    int
        Days until FOMC (-1 if no future FOMC found)
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    elif isinstance(dt, datetime):
        dt = dt.date()
    
    # Find next FOMC after this date
    future_fomc = [d for d in ALL_FOMC_DATES if d > dt]
    if not future_fomc:
        return -1
    
    next_fomc = min(future_fomc)
    return (next_fomc - dt).days


def get_event_context(dt: datetime | date | pd.Timestamp) -> Dict[str, any]:
    """
    Get full event context for a date.
    
    Returns
    -------
    dict
        Dictionary with all event flags and metrics
    """
    return {
        'is_fomc_day': is_fomc_day(dt),
        'is_cpi_day': is_cpi_day(dt),
        'is_nfp_day': is_nfp_day(dt),
        'has_red_folder': is_red_folder_event(dt),
        'days_since_fomc': days_since_fomc(dt),
        'days_until_fomc': days_until_fomc(dt),
    }


# =============================================================================
# WINDOW ENRICHMENT
# =============================================================================

def enrich_window_with_calendar(window: 'WindowData') -> 'WindowData':
    """
    Enrich a WindowData object with calendar event information.
    
    Uses the window's cutoff date to determine event context.
    
    Parameters
    ----------
    window : WindowData
        Window to enrich
        
    Returns
    -------
    WindowData
        Same window with calendar fields populated
    """
    from .datastructures import WindowData
    
    cutoff_date = window.cutoff.date() if hasattr(window.cutoff, 'date') else window.cutoff
    
    window.is_fomc_day = is_fomc_day(cutoff_date)
    window.is_cpi_day = is_cpi_day(cutoff_date)
    window.is_nfp_day = is_nfp_day(cutoff_date)
    window.has_red_folder = is_red_folder_event(cutoff_date)
    window.days_since_fomc = days_since_fomc(cutoff_date)
    
    return window


def enrich_collection_with_calendar(collection: 'WindowCollection') -> 'WindowCollection':
    """
    Enrich all windows in a collection with calendar data.
    
    Parameters
    ----------
    collection : WindowCollection
        Collection to enrich
        
    Returns
    -------
    WindowCollection
        Same collection with calendar fields populated
    """
    for window in collection:
        enrich_window_with_calendar(window)
    
    # Log summary
    fomc_count = sum(1 for w in collection if w.is_fomc_day)
    red_folder_count = sum(1 for w in collection if w.has_red_folder)
    logger.info(
        f"Enriched {len(collection)} windows with calendar data: "
        f"{fomc_count} FOMC days, {red_folder_count} red folder events"
    )
    
    return collection


def get_fomc_dates_in_range(start: date, end: date) -> List[date]:
    """Get all FOMC dates within a date range."""
    return sorted([d for d in ALL_FOMC_DATES if start <= d <= end])


def print_calendar_summary(start: date, end: date):
    """Print summary of economic events in date range."""
    fomc_dates = get_fomc_dates_in_range(start, end)
    
    print(f"\n📅 Economic Calendar: {start} to {end}")
    print("=" * 50)
    print(f"\nFOMC Meetings ({len(fomc_dates)}):")
    for d in fomc_dates:
        print(f"  • {d.strftime('%Y-%m-%d (%A)')}")
