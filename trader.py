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

# Osmium market-making: how wide to quote passively
OSMIUM_SPREAD_HALF = 5             # Will quote fair ± this => 9995 bid / 10005 ask
OSMIUM_AGGRESSIVE_EDGE = 1        # Take mispricing beyond 1 from fair

# Pepper Root: linear trend parameters (from data analysis)
# fair_value(t) = intercept + t * slope   where t = timestamp within the day
# The intercept shifts +1000 each day. We bootstrap it from early ticks.
PEPPER_SLOPE = 1.0 / 1000         # +1 per 1000 timestamp units = +1000 over 1M ts
PEPPER_SPREAD_HALF = 4
PEPPER_AGGRESSIVE_EDGE = 2


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
    #  ASH_COATED_OSMIUM - Pure Market-Making around 10,000
    # ═════════════════════════════════════════════════════════════

    def trade_osmium(
        self,
        od: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        fair = OSMIUM_FAIR

        # ── 1. Aggressive: take any obvious mispricing ───────────
        # Lift cheap asks (buy below fair - edge)
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair - OSMIUM_AGGRESSIVE_EDGE:
                    ask_vol = od.sell_orders[ask_price]          # negative
                    max_buy = limit - position
                    buy_qty = min(-ask_vol, max_buy)
                    if buy_qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_price, buy_qty))
                        position += buy_qty
                        logger.print(f"OSM TAKE BUY {buy_qty}x @ {ask_price}")

        # Hit expensive bids (sell above fair + edge)
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + OSMIUM_AGGRESSIVE_EDGE:
                    bid_vol = od.buy_orders[bid_price]           # positive
                    max_sell = limit + position                   # position can go to -limit
                    sell_qty = min(bid_vol, max_sell)
                    if sell_qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_price, -sell_qty))
                        position -= sell_qty
                        logger.print(f"OSM TAKE SELL {sell_qty}x @ {bid_price}")

        # ── 2. Passive market-making with inventory skew ─────────
        # Shift quotes to lean position back to zero
        # If long, lower both quotes to sell more / buy less
        # If short, raise both quotes to buy more / sell less
        skew = -position * 0.5                                  # 0.5 per unit of inventory
        skew = max(-3, min(3, skew))                            # Cap the skew

        bid_price = int(round(fair - OSMIUM_SPREAD_HALF + skew))
        ask_price = int(round(fair + OSMIUM_SPREAD_HALF + skew))

        # Ensure we don't cross fair
        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)

        bid_qty = limit - position
        ask_qty = limit + position

        if bid_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", ask_price, -ask_qty))

        return orders

    # ═════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT - Linear Trend Following + MM
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

        # ── Bootstrap the daily intercept from early observations ──
        # Strategy: collect the first few mid-prices, fit intercept
        # fair_value(t) = intercept + t * PEPPER_SLOPE
        # => intercept = mid_price - t * PEPPER_SLOPE
        
        pepper_key = "pepper_intercept"
        pepper_prices_key = "pepper_bootstrap_prices"

        # Compute current mid from order book
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid = best_bid
        elif best_ask is not None:
            mid = best_ask
        else:
            return orders, trader_state

        # During first N ticks, collect prices to estimate intercept
        bootstrap_n = 10
        if pepper_key not in trader_state:
            # Accumulate bootstrap observations
            if pepper_prices_key not in trader_state:
                trader_state[pepper_prices_key] = []
            
            trader_state[pepper_prices_key].append({
                "t": timestamp,
                "mid": mid,
            })

            if len(trader_state[pepper_prices_key]) >= bootstrap_n:
                # Estimate intercept: intercept = mid - t * slope
                intercepts = [
                    obs["mid"] - obs["t"] * PEPPER_SLOPE
                    for obs in trader_state[pepper_prices_key]
                ]
                trader_state[pepper_key] = sum(intercepts) / len(intercepts)
                # Clean up bootstrap data
                del trader_state[pepper_prices_key]
                logger.print(f"PEPPER bootstrapped intercept = {trader_state[pepper_key]:.1f}")
            else:
                # Not enough data yet - don't trade, just observe
                return orders, trader_state

        intercept = trader_state[pepper_key]

        # ── Compute fair value ───────────────────────────────────
        fair = intercept + timestamp * PEPPER_SLOPE

        # ── Continuously refine intercept with EMA ───────────────
        # Gently adapt if our estimate drifts
        alpha = 0.01
        residual = mid - fair
        trader_state[pepper_key] = intercept + alpha * residual

        logger.print(f"PEPPER t={timestamp} fair={fair:.1f} mid={mid:.1f} residual={residual:.1f} pos={position}")

        # ── 1. Aggressive: take mispriced levels ─────────────────
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair - PEPPER_AGGRESSIVE_EDGE:
                    ask_vol = od.sell_orders[ask_price]          # negative
                    max_buy = limit - position
                    buy_qty = min(-ask_vol, max_buy)
                    if buy_qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, buy_qty))
                        position += buy_qty
                        logger.print(f"PEPPER TAKE BUY {buy_qty}x @ {ask_price} (fair={fair:.1f})")

        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + PEPPER_AGGRESSIVE_EDGE:
                    bid_vol = od.buy_orders[bid_price]
                    max_sell = limit + position
                    sell_qty = min(bid_vol, max_sell)
                    if sell_qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, -sell_qty))
                        position -= sell_qty
                        logger.print(f"PEPPER TAKE SELL {sell_qty}x @ {bid_price} (fair={fair:.1f})")

        # ── 2. Passive MM with trend-aware skew ──────────────────
        # Since price is trending UP, we want a slight long bias
        # Also apply inventory skew to flatten position
        trend_bias = 1                                          # Slight long lean
        inv_skew = -position * 0.4
        inv_skew = max(-3, min(3, inv_skew))

        bid_price = int(round(fair - PEPPER_SPREAD_HALF + inv_skew + trend_bias))
        ask_price = int(round(fair + PEPPER_SPREAD_HALF + inv_skew + trend_bias))

        bid_qty = limit - position
        ask_qty = limit + position

        if bid_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, -ask_qty))

        return orders, trader_state
