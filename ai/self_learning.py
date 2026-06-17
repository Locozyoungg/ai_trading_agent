"""
Hedge Fund Trend-Following Strategy

Disciplined, rules-based trend-following strategy designed for institutional-grade
performance. Core principles:

    1. Trend is your friend — only trade in the direction of the dominant trend
    2. Cut losers short — volatility-adaptive ATR stops, not fixed percentages
    3. Let winners run — 2:1 reward-to-risk as the minimum threshold
    4. Flat is a position — in ranging regimes, NO trades
    5. Capital preservation — position sizing via fractional Kelly, halt after 3 losses

This is not a black-box ML system. Every rule is interpretable, testable, and
grounded in proven market mechanics.
"""

import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Data Types ─────────────────────────────────────────────────────────────


class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class MarketState:
    """Complete description of current market conditions."""
    regime: Regime
    trend_slope: float          # normalized EMA slope
    atr: float                  # Average True Range (price units)
    atr_pct: float              # ATR as % of price
    ema_short: float            # fast EMA value
    ema_long: float             # slow EMA value
    price: float                # current close
    volume_ratio: float         # current vol / avg vol
    rsi: float                  # RSI-14
    regime_confidence: float    # 0-1 how confident we are in regime classification


@dataclass
class TradeSignal:
    """Output from the strategy — what to do and how to manage it."""
    action: str                 # 'buy', 'sell', 'hold'
    confidence: float           # 0.0 to 1.0
    stop_price: float           # initial stop-loss level
    target_price: float         # initial take-profit level
    regime: str
    rationale: str              # human-readable reason


@dataclass
class TradeRecord:
    """Record of a completed trade for performance tracking."""
    entry_price: float
    exit_price: float
    side: str
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    regime: str
    timestamp: float


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_EMA_SHORT = 20
DEFAULT_EMA_LONG = 50
DEFAULT_ATR_PERIOD = 14
DEFAULT_RSI_PERIOD = 14
DEFAULT_VOLUME_PERIOD = 20
DEFAULT_RISK_PER_TRADE = 0.01       # 1% account risk per trade
DEFAULT_MIN_RRR = 2.0               # minimum reward-to-risk ratio
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3  # halt after 3 losses
DEFAULT_STOP_ATR_MULTIPLIER = 1.5   # stop at 1.5× ATR
DEFAULT_TP_ATR_MULTIPLIER = 3.0     # target at 3× ATR
DEFAULT_TRAIL_ACTIVATE = 1.5        # activate trail at 1.5× ATR profit
DEFAULT_TRAIL_DISTANCE = 1.0        # trail by 1.0× ATR


# ── Strategy Engine ────────────────────────────────────────────────────────


class HedgeFundStrategy:
    """
    Institutional trend-following strategy.

    Uses EMA crossovers for trend detection, ATR for volatility-adaptive
    stops and targets, and regime classification for position management.

    Market regimes:
      - TREND_UP:     50-EMA slope > threshold   → long only
      - TREND_DOWN:   50-EMA slope < -threshold  → short only
      - RANGING:      slope near zero             → NO TRADES
      - VOLATILE:     ATR spike                   → half size
    """

    def __init__(
        self,
        ema_short: int = DEFAULT_EMA_SHORT,
        ema_long: int = DEFAULT_EMA_LONG,
        atr_period: int = DEFAULT_ATR_PERIOD,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        vol_period: int = DEFAULT_VOLUME_PERIOD,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        stop_atr: float = DEFAULT_STOP_ATR_MULTIPLIER,
        tp_atr: float = DEFAULT_TP_ATR_MULTIPLIER,
        trail_activate: float = DEFAULT_TRAIL_ACTIVATE,
        trail_distance: float = DEFAULT_TRAIL_DISTANCE,
        min_rrr: float = DEFAULT_MIN_RRR,
        trend_threshold: float = 0.00015,  # min EMA slope to consider trending
    ):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.vol_period = vol_period
        self.risk_per_trade = risk_per_trade
        self.stop_atr = stop_atr
        self.tp_atr = tp_atr
        self.trail_activate = trail_activate
        self.trail_distance = trail_distance
        self.min_rrr = min_rrr
        self.trend_threshold = trend_threshold

        # Performance tracking
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.trade_history: deque = deque(maxlen=500)

        logger.info(
            "HedgeFundStrategy: EMA(%d,%d) ATR(%d) RRR(%.1f:1) stop=%.1f×ATR tp=%.1f×ATR",
            ema_short, ema_long, atr_period, min_rrr, stop_atr, tp_atr
        )

    # ── Technical Primitives ──────────────────────────────────────────────

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        if len(values) < period:
            return values
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _sma(values: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        if len(values) < period:
            return values
        smoothed = np.copy(values)
        for i in range(period - 1, len(values)):
            smoothed[i] = np.mean(values[i - period + 1:i + 1])
        return smoothed

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        if len(close) < 2:
            return np.ones_like(close) * (close[-1] * 0.01) if len(close) > 0 else np.array([1.0])
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[tr[0]], tr])  # pad first
        atr = np.zeros_like(tr)
        atr[:period] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = tr[i] * alpha + atr[i - 1] * (1 - alpha)
        return atr

    def _rsi(self, prices: np.ndarray) -> float:
        """RSI-14."""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = deltas[deltas > 0].sum() if np.any(deltas > 0) else 0
        losses = (-deltas[deltas < 0]).sum() if np.any(deltas < 0) else 0
        avg_gain = gains / self.rsi_period
        avg_loss = losses / self.rsi_period
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _ema_slope(self, prices: np.ndarray, period: int) -> float:
        """Normalized slope of the EMA as a measure of trend strength."""
        if len(prices) < period + 5:
            return 0.0
        ema_vals = self._ema(prices, period)
        recent = ema_vals[-min(period, len(ema_vals)):]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope / (recent.mean() + 1e-10))

    # ── Market State Assessment ───────────────────────────────────────────

    def assess_market(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> MarketState:
        """
        Build a complete picture of current market conditions.

        This is called every cycle and is the single source of truth for
        all trading decisions.
        """
        price = closes[-1]

        # EMAs
        ema_s = self._ema(closes, self.ema_short)[-1]
        ema_l = self._ema(closes, self.ema_long)[-1]

        # Trend slope (normalized)
        trend_slope = self._ema_slope(closes, self.ema_long)

        # ATR
        atr_values = self._atr(highs, lows, closes, self.atr_period)
        atr = float(atr_values[-1]) if len(atr_values) > 0 else price * 0.01
        atr_pct = atr / price * 100

        # RSI
        rsi = self._rsi(closes)

        # Volume ratio
        vol_ratio = 1.0
        if len(volumes) > self.vol_period:
            avg_vol = np.mean(volumes[-self.vol_period:])
            if avg_vol > 0:
                vol_ratio = float(volumes[-1] / avg_vol)

        # Regime classification
        regime, confidence = self._classify_regime(trend_slope, atr_pct, rsi, vol_ratio)

        return MarketState(
            regime=regime,
            trend_slope=trend_slope,
            atr=atr,
            atr_pct=atr_pct,
            ema_short=ema_s,
            ema_long=ema_l,
            price=price,
            volume_ratio=vol_ratio,
            rsi=rsi,
            regime_confidence=confidence,
        )

    def _classify_regime(
        self,
        trend_slope: float,
        atr_pct: float,
        rsi: float,
        vol_ratio: float,
    ) -> Tuple[Regime, float]:
        """
        Classify the market regime and return confidence level.

        Priority:
        1. VOLATILE — if ATR is extreme (top 10% of its range)
        2. TREND_UP / TREND_DOWN — if EMA slope exceeds threshold
        3. RANGING — everything else
        """
        # Volatile regime check
        if atr_pct > 3.0:
            return Regime.VOLATILE, 0.7
        if vol_ratio > 2.0 and atr_pct > 2.0:
            return Regime.VOLATILE, 0.6

        # Trending regime check
        if trend_slope > self.trend_threshold:
            confidence = min(1.0, abs(trend_slope) / (self.trend_threshold * 5))
            return Regime.TREND_UP, confidence
        elif trend_slope < -self.trend_threshold:
            confidence = min(1.0, abs(trend_slope) / (self.trend_threshold * 5))
            return Regime.TREND_DOWN, confidence

        # Default: ranging
        return Regime.RANGING, 0.3

    # ── Signal Generation ────────────────────────────────────────────────

    def generate_signal(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> TradeSignal:
        """
        Generate a trading signal based on market state.

        This is the core decision function. Every trade has:
        - A clear rationale
        - A volatility-adaptive stop price
        - A target price (minimum 2:1 reward-to-risk)
        - Regime context
        """
        state = self.assess_market(closes, highs, lows, volumes)

        # ── NO-TRADE ZONES ───────────────────────────────────────────────

        # Not enough data
        if len(closes) < max(self.ema_long, self.atr_period) + 5:
            return TradeSignal("hold", 0.0, 0.0, 0.0, state.regime.value, "Warm-up: insufficient data")

        # Flat in ranging markets — this is THE discipline most retail traders lack
        if state.regime == Regime.RANGING:
            return TradeSignal(
                "hold", 0.0, 0.0, 0.0,
                state.regime.value,
                "Ranging regime — no edge. Waiting for breakout or trend."
            )

        # Volatile — reduce risk dramatically
        if state.regime == Regime.VOLATILE:
            return TradeSignal(
                "hold", 0.0, 0.0, 0.0,
                state.regime.value,
                "Volatile regime — protecting capital. Waiting for stabilization."
            )

        # ── Volume & Volatility Filters ──────────────────────────────────
        # Low volume = no institutional interest = no edge
        if state.volume_ratio < 0.7:
            return TradeSignal(
                "hold", 0.0, 0.0, 0.0,
                state.regime.value,
                f"Volume too low ({state.volume_ratio:.1f}x) — waiting for institutional participation."
            )

        # Extreme volatility = unpredictable = protect capital
        if state.atr_pct > 2.5:
            return TradeSignal(
                "hold", 0.0, 0.0, 0.0,
                state.regime.value,
                f"ATR {state.atr_pct:.2f}% too wide — risk is unmanageable. Waiting for compression."
            )

        # ── TREND-UP SIGNAL ──────────────────────────────────────────────

        if state.regime == Regime.TREND_UP:
            # Entry conditions for long:
            # 1. Price above short EMA (momentum is with us)
            # 2. RSI not overbought (>70)
            # 3. Volume confirming (optional, prefer above-average)
            if state.price < state.ema_short:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"Trend up but price below EMA{self.ema_short} — waiting for pullback to end."
                )
            if state.rsi > 70:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"RSI {state.rsi:.0f} overbought — waiting for pullback."
                )

            # Calculate stop and target
            stop_price = state.price - state.atr * self.stop_atr
            target_price = state.price + state.atr * self.tp_atr
            risk = (state.price - stop_price) / state.price * 100
            reward = (target_price - state.price) / state.price * 100

            # Check minimum reward-to-risk
            if reward / risk < self.min_rrr and risk > 0:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"RRR {reward/risk:.1f}:1 below minimum {self.min_rrr:.1f}:1."
                )

            confidence = min(1.0, state.regime_confidence * (1.0 + state.volume_ratio * 0.2))
            rationale = (
                f"BUY trend_up | "
                f"RSI={state.rsi:.0f} ATR={state.atr_pct:.2f}% "
                f"RRR={reward/risk:.1f}:1 vol={state.volume_ratio:.1f}x"
            )

            return TradeSignal("buy", confidence, stop_price, target_price, state.regime.value, rationale)

        # ── TREND-DOWN SIGNAL ────────────────────────────────────────────

        if state.regime == Regime.TREND_DOWN:
            # Entry conditions for short:
            if state.price > state.ema_short:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"Trend down but price above EMA{self.ema_short} — waiting for bounce to end."
                )
            if state.rsi < 30:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"RSI {state.rsi:.0f} oversold — waiting for bounce."
                )

            stop_price = state.price + state.atr * self.stop_atr
            target_price = state.price - state.atr * self.tp_atr
            risk = (stop_price - state.price) / state.price * 100
            reward = (state.price - target_price) / state.price * 100

            if reward / risk < self.min_rrr and risk > 0:
                return TradeSignal(
                    "hold", 0.0, 0.0, 0.0,
                    state.regime.value,
                    f"RRR {reward/risk:.1f}:1 below minimum {self.min_rrr:.1f}:1."
                )

            confidence = min(1.0, state.regime_confidence * (1.0 + state.volume_ratio * 0.2))
            rationale = (
                f"SELL trend_down | "
                f"RSI={state.rsi:.0f} ATR={state.atr_pct:.2f}% "
                f"RRR={reward/risk:.1f}:1 vol={state.volume_ratio:.1f}x"
            )

            return TradeSignal("sell", confidence, stop_price, target_price, state.regime.value, rationale)

        # Fallback
        return TradeSignal("hold", 0.0, 0.0, 0.0, state.regime.value, "No signal.")

    # ── Position Management ──────────────────────────────────────────────

    def compute_next_stop(
        self,
        side: str,
        entry: float,
        current_price: float,
        atr: float,
        highest_price: float,
        lowest_price: float,
    ) -> Tuple[float, str]:
        """
        Compute the appropriate stop price for an open position.

        Returns (stop_price, reason).
        Manages trailing stop logic for runners.
        """
        if side == "long":
            initial_stop = entry - atr * self.stop_atr
            profit = (current_price - entry) / atr  # profit in ATR units
            if profit >= self.trail_activate:
                # Trail: stop = highest_price - trail_distance * ATR
                trail_stop = highest_price - atr * self.trail_distance
                return max(initial_stop, trail_stop), "trailing"
            return initial_stop, "initial"
        else:
            initial_stop = entry + atr * self.stop_atr
            profit = (entry - current_price) / atr
            if profit >= self.trail_activate:
                trail_stop = lowest_price + atr * self.trail_distance
                return min(initial_stop, trail_stop), "trailing"
            return initial_stop, "initial"

    # ── Trade Outcome Tracking ───────────────────────────────────────────

    def record_trade_outcome(self, trade: TradeRecord):
        """Record completed trade for performance tracking and risk adjustment."""
        self.trade_history.append(trade)
        self.total_trades += 1

        if trade.pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        logger.info(
            "EXIT %s %s: %+.2f USDT (%+.2f%%) | Win rate: %.0f%% | Regime: %s | Reason: %s",
            trade.side.upper(), trade.regime, trade.pnl, trade.pnl_pct,
            self.win_rate * 100, trade.regime, trade.exit_reason
        )

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.5

    def get_position_sizing_factor(self) -> float:
        """
        Returns a multiplier for position sizing.
        1.0 = normal, 0.0 = halt.
        """
        if self.consecutive_losses >= DEFAULT_MAX_CONSECUTIVE_LOSSES:
            logger.warning(
                "%d consecutive losses — HALTING trading",
                self.consecutive_losses
            )
            return 0.0
        if self.consecutive_losses >= 2:
            return 0.5  # half size after 2 losses
        return 1.0


if __name__ == "__main__":
    """Self-test with generated data."""
    np.random.seed(42)
    # Generate a trend with noise
    trend = np.linspace(0, 2000, 400)  # upward drift
    noise = np.random.randn(400) * 200
    prices = 50000 + trend + noise
    highs = prices + np.abs(np.random.randn(400) * 100)
    lows = prices - np.abs(np.random.randn(400) * 100)
    volumes = np.random.rand(400) * 100 + 50

    strat = HedgeFundStrategy()
    signal = strat.generate_signal(prices, highs, lows, volumes)
    print(f"Signal: {signal.action}")
    print(f"Regime: {signal.regime}")
    print(f"Rationale: {signal.rationale}")
    print(f"Stop: {signal.stop_price:.2f} Target: {signal.target_price:.2f}")
    print(f"Sizing factor: {strat.get_position_sizing_factor():.2f}")
