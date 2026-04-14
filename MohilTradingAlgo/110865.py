import json
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ─────────────────────────────────────────────────────────────────────
#  Logger (Standard Prosperity 4 logger for the visualizer)
# ─────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {}
        for product, obs in observations.conversionObservations.items():
            conv_obs[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# ── ASH_COATED_OSMIUM ─────────────────────────────────────────────────
# Mean-reverts around 10,000. Market spread is ~16 wide (9990-9994 / 10009-10013).
# Strategy: tight MM at ±2 to be best bid/ask and capture maximum flow.
OSMIUM_FAIR = 10_000
OSMIUM_MM_HALF   = 2    # Quote fair ± 2  →  9998 bid / 10002 ask
OSMIUM_TAKE_EDGE = 3    # Take any ask < 9997 or bid > 10003 (clearly mispriced)
OSMIUM_INV_SKEW  = 0.3  # Inventory skew coefficient (per unit of position)
OSMIUM_MAX_SKEW  = 4    # Cap inventory skew

# ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────
# PERFECT linear trend: +1000 per day (+0.001 per timestamp unit).
# Intercept shifts +1000 each day (bootstrapped from first observation).
#
# KEY INSIGHT: holding 80 long all day = +80,000/day from trend alone.
# Strategy: go MAX LONG immediately and hold. Never sell passively.
# Only sell on massive spikes well above fair (>+50) to capture rare windfalls.
PEPPER_SLOPE      = 1.0 / 1000   # +1 per 1000 timestamp units
PEPPER_BUY_EDGE   = 8            # Buy any ask at or below fair + 8
PEPPER_SELL_SPIKE = 50           # Only sell if bid > fair + 50
PEPPER_PASSIVE_BID_OFFSET = 2    # Passive bid at fair + this offset (above midpoint)


# ─────────────────────────────────────────────────────────────────────
#  TRADER
# ─────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}

        trader_state = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        for product in state.order_depths:
            od: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 50)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(od, position, limit)

            elif product == "INTARIAN_PEPPER_ROOT":
                result[product], trader_state = self.trade_pepper(
                    od, position, limit, state.timestamp, trader_state
                )

        trader_data = json.dumps(trader_state)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ═════════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM — Tight Market-Making around 10,000
    #
    #  Previous spread was ±5, putting us at 9995/10005. We were inside
    #  the market spread but not tight enough to capture most flow.
    #  Moving to ±2 makes us the best bid/ask and captures the ~2190
    #  market volume/day at 4 ticks profit per round trip (~10k/day).
    # ═════════════════════════════════════════════════════════════════

    def trade_osmium(self, od: OrderDepth, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        fair = OSMIUM_FAIR

        # ── 1. Aggressive: snap up any clearly mispriced levels ──────
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask < fair - OSMIUM_TAKE_EDGE:
                    qty = min(-od.sell_orders[ask], limit - position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                        position += qty
                        logger.print(f"OSM TAKE BUY {qty}@{ask}")

        if od.buy_orders:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid > fair + OSMIUM_TAKE_EDGE:
                    qty = min(od.buy_orders[bid], limit + position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                        position -= qty
                        logger.print(f"OSM TAKE SELL {qty}@{bid}")

        # ── 2. Tight passive MM with inventory skew ──────────────────
        # Clamp skew so we don't cross fair
        skew = -position * OSMIUM_INV_SKEW
        skew = max(-OSMIUM_MAX_SKEW, min(OSMIUM_MAX_SKEW, skew))

        bid_px = int(round(fair - OSMIUM_MM_HALF + skew))
        ask_px = int(round(fair + OSMIUM_MM_HALF + skew))

        # Hard floor/ceiling: never cross fair
        bid_px = min(bid_px, fair - 1)
        ask_px = max(ask_px, fair + 1)

        bid_qty = limit - position
        ask_qty = limit + position

        if bid_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", bid_px, bid_qty))
        if ask_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", ask_px, -ask_qty))

        return orders

    # ═════════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — Trend-Following (Max Long)
    #
    #  Previous: symmetric MM around trend → tiny round-trip profits,
    #  completely ignoring the +1000/day trend signal.
    #
    #  New: go MAX LONG as fast as possible and hold.
    #  Expected PnL: ~80 units × +1000/day × 3 days = +240,000.
    #  Compare to old ~1,800 total.
    # ═════════════════════════════════════════════════════════════════

    def trade_pepper(
        self,
        od: OrderDepth,
        position: int,
        limit: int,
        timestamp: int,
        trader_state: dict,
    ) -> tuple[List[Order], dict]:
        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None and best_ask is None:
            return orders, trader_state

        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else (best_bid or best_ask)

        # ── Bootstrap fair-value intercept from first observation ────
        # Since the trend is a PERFECT linear function, a single
        # observation is enough: intercept = mid - t * slope
        pepper_key = "pepper_intercept"
        if pepper_key not in trader_state:
            trader_state[pepper_key] = mid - timestamp * PEPPER_SLOPE
            logger.print(f"PEPPER bootstrap intercept={trader_state[pepper_key]:.1f} at t={timestamp}")

        intercept = trader_state[pepper_key]

        # ── Slowly adapt intercept (handles day-boundary shifts) ─────
        fair = intercept + timestamp * PEPPER_SLOPE
        residual = mid - fair
        # Very gentle adaptation — the trend is essentially deterministic
        trader_state[pepper_key] = intercept + 0.005 * residual

        logger.print(f"PEPPER t={timestamp} fair={fair:.1f} mid={mid:.1f} pos={position}")

        # ── 1. Aggressive BUY: take any ask ≤ fair + BUY_EDGE ────────
        # Even buying slightly above fair is fine — the trend recovers
        # that cost in BUY_EDGE * 1000 timestamp units (a tiny fraction
        # of the day).  We always want to be at MAX LONG.
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask <= fair + PEPPER_BUY_EDGE:
                    qty = min(-od.sell_orders[ask], limit - position)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                        position += qty
                        logger.print(f"PEPPER TAKE BUY {qty}@{ask} fair={fair:.1f}")

        # ── 2. Rare spike sell: only if bid is far above fair ────────
        # We never want a short position on a trending-up asset.
        # Only sell a portion if price is wildly above fair (mean
        # reversion / fat tail), and never go net short.
        if od.buy_orders and position > 0:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid > fair + PEPPER_SELL_SPIKE:
                    qty = min(od.buy_orders[bid], position)  # Don't go short
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                        position -= qty
                        logger.print(f"PEPPER SPIKE SELL {qty}@{bid} fair={fair:.1f}")

        # ── 3. Passive BUY: fill remaining capacity at fair+offset ───
        # Quote above the mid so we're the best bid and attract sellers.
        # This fills up the last portion of our limit that aggressive
        # taking didn't catch.
        remaining = limit - position
        if remaining > 0:
            passive_bid = int(round(fair + PEPPER_PASSIVE_BID_OFFSET))
            # Cap at best_ask - 1 to avoid crossing the book
            if best_ask is not None:
                passive_bid = min(passive_bid, best_ask - 1)
            orders.append(Order("INTARIAN_PEPPER_ROOT", passive_bid, remaining))

        # NO passive sell order — we hold our long position for the trend.

        return orders, trader_state
