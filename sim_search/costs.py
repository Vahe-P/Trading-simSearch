"""
Transaction Cost Modeling for Backtesting.

Provides realistic cost estimation for backtesting:
- Commission/brokerage fees
- Slippage (market impact)
- Spread (bid-ask)

Standard values are provided for common instruments.

Usage:
    from sim_search.costs import TransactionCosts, FUTURES_NQ, FUTURES_ES
    
    # Use preset for NQ futures
    costs = FUTURES_NQ
    
    # Or create custom
    costs = TransactionCosts(
        commission_per_side=2.50,
        slippage_ticks=1,
        tick_size=0.25,
        tick_value=5.00
    )
    
    # Calculate round-trip cost
    entry_cost, exit_cost, total_cost = costs.calculate_trade_cost(entry_price=18000.0)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class TransactionCosts:
    """
    Transaction cost model for backtesting.
    
    All costs are expressed per contract/share.
    
    Parameters
    ----------
    commission_per_side : float
        Broker commission per side (entry or exit). Default: $2.50
        Typical values:
        - Futures (discount): $1.00 - $2.50 per side
        - Futures (full-service): $5.00 - $15.00 per side
        - Stocks: $0 (most brokers) to $0.005/share
        
    slippage_ticks : float
        Expected slippage in ticks per side. Default: 1.0 tick
        Typical values:
        - Liquid futures (ES, NQ): 0.5 - 1.0 ticks
        - Less liquid: 1.0 - 2.0 ticks
        - Stocks (liquid): 0.01% - 0.05% of price
        
    tick_size : float
        Minimum price increment. Default: 0.25 (NQ/ES)
        Common values:
        - NQ, ES: 0.25
        - CL (crude): 0.01
        - GC (gold): 0.10
        - Stocks: 0.01
        
    tick_value : float
        Dollar value per tick per contract. Default: $5.00 (NQ)
        Common values:
        - NQ: $5.00 per tick ($20/point)
        - ES: $12.50 per tick ($50/point)
        - CL: $10.00 per tick
        - GC: $10.00 per tick
        
    spread_ticks : float
        Typical bid-ask spread in ticks. Default: 1.0 tick
        For market orders, you pay half the spread on entry and exit.
        Typical values:
        - ES, NQ (liquid hours): 0.25 - 1.0 ticks
        - ES, NQ (overnight): 1.0 - 2.0 ticks
        - Less liquid futures: 2.0 - 5.0 ticks
        
    exchange_fee : float
        Exchange and clearing fees per side. Default: $1.18 (CME)
        Typical values:
        - CME futures: ~$1.18 per side
        - ICE: ~$1.00 per side
        - Stocks: ~$0.003/share (SEC/FINRA)
        
    nfa_fee : float
        NFA (National Futures Association) fee per side. Default: $0.02
        
    enabled : bool
        Whether to apply transaction costs. Default: True
        Set to False to run backtest without costs for comparison.
    
    Notes
    -----
    Total round-trip cost formula:
        cost = 2 * (commission + exchange_fee + nfa_fee) + 
               2 * slippage_ticks * tick_value +
               spread_ticks * tick_value
    
    For NQ with defaults:
        cost = 2 * ($2.50 + $1.18 + $0.02) + 2 * 1.0 * $5.00 + 1.0 * $5.00
             = 2 * $3.70 + $10.00 + $5.00
             = $7.40 + $15.00
             = $22.40 per round-trip per contract
    """
    # Commission
    commission_per_side: float = 2.50
    
    # Slippage
    slippage_ticks: float = 1.0
    tick_size: float = 0.25
    tick_value: float = 5.00
    
    # Spread
    spread_ticks: float = 1.0
    
    # Exchange/regulatory fees
    exchange_fee: float = 1.18
    nfa_fee: float = 0.02
    
    # Master switch
    enabled: bool = True
    
    @property
    def slippage_dollars(self) -> float:
        """Slippage cost in dollars per side."""
        return self.slippage_ticks * self.tick_value
    
    @property
    def spread_dollars(self) -> float:
        """Full spread cost in dollars (paid once per round-trip)."""
        return self.spread_ticks * self.tick_value
    
    @property
    def fees_per_side(self) -> float:
        """Total fees per side (commission + exchange + NFA)."""
        return self.commission_per_side + self.exchange_fee + self.nfa_fee
    
    @property
    def cost_per_side(self) -> float:
        """Total cost per side including slippage."""
        return self.fees_per_side + self.slippage_dollars
    
    @property
    def round_trip_cost(self) -> float:
        """
        Total round-trip cost per contract in dollars.
        
        Includes:
        - 2x commission (entry + exit)
        - 2x exchange fees
        - 2x NFA fees
        - 2x slippage
        - 1x spread (crossing the spread on entry)
        """
        if not self.enabled:
            return 0.0
        return (2 * self.cost_per_side) + (0.5 * self.spread_dollars)
    
    def calculate_trade_cost(
        self, 
        entry_price: float,
        contracts: int = 1,
        include_spread: bool = True
    ) -> Tuple[float, float, float]:
        """
        Calculate entry cost, exit cost, and total round-trip cost.
        
        Parameters
        ----------
        entry_price : float
            Entry price (used for percentage calculations)
        contracts : int
            Number of contracts
        include_spread : bool
            Whether to include spread cost (True for market orders)
        
        Returns
        -------
        entry_cost : float
            Cost at entry in dollars
        exit_cost : float
            Cost at exit in dollars
        total_cost : float
            Total round-trip cost in dollars
        """
        if not self.enabled:
            return 0.0, 0.0, 0.0
        
        entry_cost = self.cost_per_side * contracts
        if include_spread:
            entry_cost += 0.5 * self.spread_dollars * contracts
        
        exit_cost = self.cost_per_side * contracts
        
        total_cost = entry_cost + exit_cost
        return entry_cost, exit_cost, total_cost
    
    def cost_as_return(self, entry_price: float, point_value: Optional[float] = None) -> float:
        """
        Express round-trip cost as a return (decimal).
        
        This is useful for subtracting from trade returns.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        point_value : float, optional
            Dollar value per point. If None, uses tick_value / tick_size.
        
        Returns
        -------
        float
            Cost as a decimal return (e.g., 0.001 = 0.1%)
        
        Example
        -------
        >>> costs = FUTURES_NQ
        >>> cost_ret = costs.cost_as_return(entry_price=18000.0)
        >>> # Subtract from trade return
        >>> net_return = gross_return - cost_ret
        """
        if not self.enabled:
            return 0.0
        
        if point_value is None:
            point_value = self.tick_value / self.tick_size
        
        # Contract notional value
        notional = entry_price * point_value
        
        # Cost as fraction of notional
        return self.round_trip_cost / notional
    
    def adjust_returns(
        self, 
        returns: np.ndarray, 
        entry_price: float,
        point_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Adjust trade returns for transaction costs.
        
        Subtracts half the round-trip cost at entry (first bar)
        and half at exit (last bar).
        
        Parameters
        ----------
        returns : np.ndarray
            Array of bar-by-bar returns
        entry_price : float
            Entry price
        point_value : float, optional
            Dollar value per point
        
        Returns
        -------
        np.ndarray
            Adjusted returns with costs subtracted
        """
        if not self.enabled or len(returns) == 0:
            return returns
        
        cost_ret = self.cost_as_return(entry_price, point_value)
        
        # Create a copy to avoid modifying original
        adjusted = returns.copy()
        
        # Subtract half at entry, half at exit
        adjusted[0] -= cost_ret / 2
        adjusted[-1] -= cost_ret / 2
        
        return adjusted
    
    def summary(self) -> str:
        """Return human-readable cost summary."""
        if not self.enabled:
            return "Transaction costs: DISABLED"
        
        return (
            f"Transaction Costs Summary\n"
            f"{'='*40}\n"
            f"Commission:    ${self.commission_per_side:.2f}/side\n"
            f"Exchange fee:  ${self.exchange_fee:.2f}/side\n"
            f"NFA fee:       ${self.nfa_fee:.2f}/side\n"
            f"Slippage:      {self.slippage_ticks:.1f} ticks (${self.slippage_dollars:.2f}/side)\n"
            f"Spread:        {self.spread_ticks:.1f} ticks (${self.spread_dollars:.2f})\n"
            f"{'='*40}\n"
            f"TOTAL ROUND-TRIP: ${self.round_trip_cost:.2f}/contract\n"
        )


# =============================================================================
# PRESETS FOR COMMON INSTRUMENTS
# =============================================================================

# NQ (E-mini Nasdaq 100) - Most common for algo trading
FUTURES_NQ = TransactionCosts(
    commission_per_side=2.50,      # Discount broker rate
    slippage_ticks=1.0,            # 1 tick = $5
    tick_size=0.25,
    tick_value=5.00,               # $20 per point
    spread_ticks=1.0,              # 1 tick spread typical
    exchange_fee=1.18,             # CME fee
    nfa_fee=0.02,
)

# ES (E-mini S&P 500)
FUTURES_ES = TransactionCosts(
    commission_per_side=2.50,
    slippage_ticks=0.5,            # More liquid, less slippage
    tick_size=0.25,
    tick_value=12.50,              # $50 per point
    spread_ticks=0.5,              # Tighter spread
    exchange_fee=1.18,
    nfa_fee=0.02,
)

# MNQ (Micro E-mini Nasdaq) - 1/10 size of NQ
FUTURES_MNQ = TransactionCosts(
    commission_per_side=0.62,      # Lower commission for micros
    slippage_ticks=1.0,
    tick_size=0.25,
    tick_value=0.50,               # $2 per point
    spread_ticks=1.0,
    exchange_fee=0.30,             # Lower exchange fee
    nfa_fee=0.02,
)

# MES (Micro E-mini S&P)
FUTURES_MES = TransactionCosts(
    commission_per_side=0.62,
    slippage_ticks=0.5,
    tick_size=0.25,
    tick_value=1.25,               # $5 per point
    spread_ticks=0.5,
    exchange_fee=0.30,
    nfa_fee=0.02,
)

# CL (Crude Oil)
FUTURES_CL = TransactionCosts(
    commission_per_side=2.50,
    slippage_ticks=1.0,
    tick_size=0.01,
    tick_value=10.00,              # $1000 per point
    spread_ticks=1.0,
    exchange_fee=1.18,
    nfa_fee=0.02,
)

# GC (Gold)
FUTURES_GC = TransactionCosts(
    commission_per_side=2.50,
    slippage_ticks=1.0,
    tick_size=0.10,
    tick_value=10.00,              # $100 per point
    spread_ticks=1.0,
    exchange_fee=1.18,
    nfa_fee=0.02,
)

# Generic stock (commission-free broker)
STOCKS_COMMISSION_FREE = TransactionCosts(
    commission_per_side=0.0,
    slippage_ticks=0.5,            # ~0.5 cents
    tick_size=0.01,
    tick_value=0.01,               # $1 per dollar move per share
    spread_ticks=1.0,              # 1 cent spread
    exchange_fee=0.003,            # SEC/FINRA fees
    nfa_fee=0.0,
)

# No costs (for comparison)
NO_COSTS = TransactionCosts(enabled=False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(instrument: str) -> TransactionCosts:
    """
    Get transaction cost preset for an instrument.
    
    Parameters
    ----------
    instrument : str
        Instrument code (NQ, ES, MNQ, MES, CL, GC, STOCK)
    
    Returns
    -------
    TransactionCosts
        Cost model for the instrument
    
    Example
    -------
    >>> costs = get_preset('NQ')
    >>> print(costs.round_trip_cost)
    22.40
    """
    presets = {
        'NQ': FUTURES_NQ,
        'ES': FUTURES_ES,
        'MNQ': FUTURES_MNQ,
        'MES': FUTURES_MES,
        'CL': FUTURES_CL,
        'GC': FUTURES_GC,
        'STOCK': STOCKS_COMMISSION_FREE,
        'NONE': NO_COSTS,
    }
    
    instrument = instrument.upper()
    if instrument not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown instrument: {instrument}. Available: {available}")
    
    return presets[instrument]


def estimate_breakeven_return(
    costs: TransactionCosts,
    entry_price: float,
    win_rate: float = 0.5
) -> float:
    """
    Estimate the minimum average return needed to break even after costs.
    
    Parameters
    ----------
    costs : TransactionCosts
        Cost model
    entry_price : float
        Typical entry price
    win_rate : float
        Expected win rate (0.0 to 1.0)
    
    Returns
    -------
    float
        Required average return per trade to break even
    
    Example
    -------
    >>> costs = FUTURES_NQ
    >>> breakeven = estimate_breakeven_return(costs, entry_price=18000, win_rate=0.55)
    >>> print(f"Need {breakeven*100:.3f}% avg return to break even")
    """
    cost_ret = costs.cost_as_return(entry_price)
    
    # With 50% win rate, need to make 2x the cost on winners
    # General formula: avg_win * win_rate - cost = 0
    # So: avg_return_needed = cost / (2 * win_rate - 1) for win_rate > 0.5
    
    if win_rate <= 0.5:
        # Can't break even with <= 50% win rate and costs
        return float('inf')
    
    return cost_ret / (2 * win_rate - 1)
