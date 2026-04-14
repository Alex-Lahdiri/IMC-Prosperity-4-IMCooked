import json
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ─────────────────────────────────────────────────────────────────────
#  Logger (standard Prosperity 4 logger for the visualizer)
# ─────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
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

# Osmium: fixed fair value
OSMIUM_FAIR = 10_000
OSMIUM_AGGRESSIVE_EDGE = 1         # Take mispricing beyond 1 from fair

# Osmium multi-level quoting config (improvement #1)
# [offset_from_fair, fraction_of_remaining_capacity]
OSMIUM_QUOTE_LEVELS = [
    (4, 0.50),   # Level 1: fair ± 4, 50% of remaining capacity
    (5, 0.30),   # Level 2: fair ± 5, 30% of remaining capacity
    (7, 1.00),   # Level 3: fair ± 7, all remaining capacity
]

# Pepper Root: linear trend parameters
PEPPER_SLOPE = 1.0 / 1000          # +1 per 1000 timestamp units
PEPPER_AGGRESSIVE_EDGE = 2
PEPPER_DEFAULT_SPREAD_HALF = 4     # Fallback when book is empty

# Pepper Root: known starting intercepts from historical data
# Day -2 started ~10000, day -1 ~11000, day 0 ~12000 => day 1 ~13000
PEPPER_EXPECTED_INTERCEPT = 13_000  # Improvement #2: trade from tick 1


# ─────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def get_weighted_mid(od: OrderDepth) -> float | None:
    """Improvement #5: Volume-weighted mid-price instead of simple mid."""
    if not od.buy_orders or not od.sell_orders:
        # Fall back to simple mid if one side is empty
        if od.buy_orders and not od.sell_orders:
            return float(max(od.buy_orders.keys()))
        elif od.sell_orders and not od.buy_orders:
            return float(min(od.sell_orders.keys()))
        return None

    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())
    bid_vol = abs(od.buy_orders[best_bid])
    ask_vol = abs(od.sell_orders[best_ask])

    if bid_vol + ask_vol == 0:
        return (best_bid + best_ask) / 2

    # Volume-weighted: price leans toward the side with MORE volume
    # (more volume on bid = price closer to ask, market is bid-heavy)
    wmid = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
    return wmid


def get_dynamic_spread(od: OrderDepth, default_half: int) -> int:
    """Improvement #4: Adapt spread to actual market spread."""
    if not od.buy_orders or not od.sell_orders:
        return default_half

    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())
    market_spread = best_ask - best_bid

    if market_spread <= 0:
        return default_half

    # Quote at roughly 50-60% of market spread per side
    # but never tighter than 3 or wider than 8
    half = max(3, min(8, int(market_spread * 0.55)))
    return half


# ─────────────────────────────────────────────────────────────────────
#  TRADER
# ─────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}

        # ── Restore persisted state ──────────────────────────────
        trader_state = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        # ── Process each product ─────────────────────────────────
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 50)

            if product == "ASH_COATED_OSMIUM":
                orders = self.trade_osmium(order_depth, position, limit)

            elif product == "INTARIAN_PEPPER_ROOT":
                orders, trader_state = self.trade_pepper(
                    order_depth, position, limit, state.timestamp, trader_state
                )

            result[product] = orders

        # ── Persist state across ticks ───────────────────────────
        trader_data = json.dumps(trader_state)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ═════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM — Market-Making with Multi-Level Quotes
    # ═════════════════════════════════════════════════════════════

    def trade_osmium(
        self,
        od: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        fair = OSMIUM_FAIR

        # ── Improvement #3: Aggressive takes FIRST, then passive ──
        # This ensures aggressive fills aren't blocked by passive quote capacity

        # Improvement #6: Position-aware aggressive edge
        # When heavily positioned, require more edge to take further
        pos_ratio = abs(position) / limit  # 0.0 to 1.0
        adjusted_edge = OSMIUM_AGGRESSIVE_EDGE
        if pos_ratio > 0.7:
            adjusted_edge = OSMIUM_AGGRESSIVE_EDGE + 1  # Need 1 more tick of edge when loaded

        # Lift cheap asks (buy below fair - edge)
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                # Only buy if price is cheap enough AND we're not too long
                if ask_price < fair - adjusted_edge and position < limit:
                    ask_vol = od.sell_orders[ask_price]   # negative
                    max_buy = limit - position
                    buy_qty = min(-ask_vol, max_buy)
                    if buy_qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_price, buy_qty))
                        position += buy_qty
                        logger.print(f"OSM TAKE BUY {buy_qty}x @ {ask_price}")
                elif ask_price >= fair - adjusted_edge:
                    break  # No more cheap levels

        # Hit expensive bids (sell above fair + edge)
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + adjusted_edge and position > -limit:
                    bid_vol = od.buy_orders[bid_price]    # positive
                    max_sell = limit + position
                    sell_qty = min(bid_vol, max_sell)
                    if sell_qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_price, -sell_qty))
                        position -= sell_qty
                        logger.print(f"OSM TAKE SELL {sell_qty}x @ {bid_price}")
                elif bid_price <= fair + adjusted_edge:
                    break

        # ── Improvement #1: Multi-level passive quoting ──────────
        # Post 2-3 levels per side at increasing distance from fair
        # with inventory skew applied to all levels

        # Inventory skew: shift all quotes toward flattening position
        skew = -position * 0.5
        skew = max(-3, min(3, skew))

        # Improvement #4: Dynamic spread from order book
        base_spread = get_dynamic_spread(od, 5)

        remaining_bid_capacity = limit - position
        remaining_ask_capacity = limit + position

        for offset, frac in OSMIUM_QUOTE_LEVELS:
            # Compute level price with skew
            bid_price = int(round(fair - offset + skew))
            ask_price = int(round(fair + offset + skew))

            # Never cross fair value
            bid_price = min(bid_price, fair - 1)
            ask_price = max(ask_price, fair + 1)

            # Allocate fraction of remaining capacity
            bid_qty = max(1, int(remaining_bid_capacity * frac))
            ask_qty = max(1, int(remaining_ask_capacity * frac))

            # Clamp to actual remaining
            bid_qty = min(bid_qty, remaining_bid_capacity)
            ask_qty = min(ask_qty, remaining_ask_capacity)

            if bid_qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid_price, bid_qty))
                remaining_bid_capacity -= bid_qty

            if ask_qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask_price, -ask_qty))
                remaining_ask_capacity -= ask_qty

            if remaining_bid_capacity <= 0 and remaining_ask_capacity <= 0:
                break

        return orders

    # ═════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — Trend Following + Adaptive MM
    # ═════════════════════════════════════════════════════════════

    def trade_pepper(
        self,
        od: OrderDepth,
        position: int,
        limit: int,
        timestamp: int,
        trader_state: dict,
    ) -> tuple[List[Order], dict]:
        orders: List[Order] = []

        pepper_key = "pepper_intercept"
        pepper_count_key = "pepper_tick_count"

        # ── Improvement #5: Weighted mid-price ───────────────────
        mid = get_weighted_mid(od)
        if mid is None:
            return orders, trader_state

        # ── Improvement #2: Trade from tick 1 ────────────────────
        # Use expected intercept immediately, refine with EMA
        if pepper_key not in trader_state:
            # First tick: bootstrap from expected value
            # Estimate intercept = mid - timestamp * slope
            estimated_from_mid = mid - timestamp * PEPPER_SLOPE
            # Blend with our prior expectation (weight prior more on first tick)
            trader_state[pepper_key] = PEPPER_EXPECTED_INTERCEPT * 0.7 + estimated_from_mid * 0.3
            trader_state[pepper_count_key] = 0
            logger.print(f"PEPPER init intercept = {trader_state[pepper_key]:.1f}")

        intercept = trader_state[pepper_key]
        tick_count = trader_state.get(pepper_count_key, 0)

        # ── Compute fair value ───────────────────────────────────
        fair = intercept + timestamp * PEPPER_SLOPE

        # ── Adaptive EMA: learn faster early, slower later ───────
        # First 20 ticks: alpha=0.1 (learn fast). After: alpha=0.01 (stable)
        alpha = 0.1 if tick_count < 20 else 0.01
        residual = mid - fair
        trader_state[pepper_key] = intercept + alpha * residual
        trader_state[pepper_count_key] = tick_count + 1

        logger.print(f"PEPPER t={timestamp} fair={fair:.1f} mid={mid:.1f} res={residual:.1f} pos={position} alpha={alpha}")

        # ── Improvement #3: Aggressive takes FIRST ───────────────

        # Improvement #6: Position-aware aggressive edge
        pos_ratio = abs(position) / limit
        adjusted_edge = PEPPER_AGGRESSIVE_EDGE
        if pos_ratio > 0.7:
            adjusted_edge = PEPPER_AGGRESSIVE_EDGE + 1

        # Only take buys if we have room to go longer
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair - adjusted_edge and position < limit:
                    ask_vol = od.sell_orders[ask_price]
                    max_buy = limit - position
                    buy_qty = min(-ask_vol, max_buy)
                    if buy_qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, buy_qty))
                        position += buy_qty
                        logger.print(f"PEPPER TAKE BUY {buy_qty}x @ {ask_price} (fair={fair:.1f})")
                elif ask_price >= fair - adjusted_edge:
                    break

        # Only take sells if we have room to go shorter
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + adjusted_edge and position > -limit:
                    bid_vol = od.buy_orders[bid_price]
                    max_sell = limit + position
                    sell_qty = min(bid_vol, max_sell)
                    if sell_qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, -sell_qty))
                        position -= sell_qty
                        logger.print(f"PEPPER TAKE SELL {sell_qty}x @ {bid_price} (fair={fair:.1f})")
                elif bid_price <= fair + adjusted_edge:
                    break

        # ── Passive MM with dynamic spread + trend bias ──────────

        # Improvement #4: Dynamic spread
        spread_half = get_dynamic_spread(od, PEPPER_DEFAULT_SPREAD_HALF)

        # Trend bias: slight long lean since price trends up
        trend_bias = 1

        # Inventory skew
        inv_skew = -position * 0.4
        inv_skew = max(-3, min(3, inv_skew))

        # Multi-level quoting for Pepper too
        remaining_bid = limit - position
        remaining_ask = limit + position

        levels = [
            (spread_half, 0.50),
            (spread_half + 2, 0.35),
            (spread_half + 4, 1.00),
        ]

        for offset, frac in levels:
            bid_price = int(round(fair - offset + inv_skew + trend_bias))
            ask_price = int(round(fair + offset + inv_skew + trend_bias))

            bid_qty = max(1, int(remaining_bid * frac))
            ask_qty = max(1, int(remaining_ask * frac))

            bid_qty = min(bid_qty, remaining_bid)
            ask_qty = min(ask_qty, remaining_ask)

            if bid_qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, bid_qty))
                remaining_bid -= bid_qty

            if ask_qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, -ask_qty))
                remaining_ask -= ask_qty

            if remaining_bid <= 0 and remaining_ask <= 0:
                break

        return orders, trader_state
