import json
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ---------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# -- OSMIUM: tight MM (from teammate) + position-aware edge (from us) --
OSMIUM_FAIR      = 10_000
OSMIUM_MM_HALF   = 2       # Tight quotes: 9998 / 10002
OSMIUM_TAKE_EDGE = 3       # Take anything beyond ±3 from fair
OSMIUM_INV_SKEW  = 0.3     # Gentle skew
OSMIUM_MAX_SKEW  = 4

# -- PEPPER: max-long trend capture (from teammate) -------------------
PEPPER_SLOPE             = 1.0 / 1000
PEPPER_BUY_EDGE          = 8     # Buy even at fair+8 - trend recovers in 8s
PEPPER_SELL_SPIKE        = 50    # Only sell on massive spikes
PEPPER_PASSIVE_BID_ABOVE = 2     # Bid at fair+2 to attract sellers


# ---------------------------------------------------------------------
#  TRADER
# ---------------------------------------------------------------------

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
            limit = POSITION_LIMITS.get(product, 80)

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

    # =============================================================
    #  ASH_COATED_OSMIUM - Tight MM + Position-Aware Edge
    #
    #  Teammate's ±2 spread (tighter = more fills on live)
    #  + Our position-aware edge (protective when loaded)
    #  + Aggressive-first ordering (from our v3)
    # =============================================================

    def trade_osmium(self, od: OrderDepth, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        fair = OSMIUM_FAIR

        # Position-aware aggressive edge (from our v3)
        edge = OSMIUM_TAKE_EDGE
        if abs(position) > int(limit * 0.7):
            edge += 1

        # -- STEP 1: Aggressive takes first -----------------------
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask < fair - edge and position < limit:
                    qty = min(-od.sell_orders[ask], limit - position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                        position += qty
                        logger.print(f"OSM TAKE BUY {qty}@{ask}")
                else:
                    break

        if od.buy_orders:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid > fair + edge and position > -limit:
                    qty = min(od.buy_orders[bid], limit + position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                        position -= qty
                        logger.print(f"OSM TAKE SELL {qty}@{bid}")
                else:
                    break

        # -- STEP 2: Tight passive MM with remaining capacity -----
        skew = -position * OSMIUM_INV_SKEW
        skew = max(-OSMIUM_MAX_SKEW, min(OSMIUM_MAX_SKEW, skew))

        bid_px = int(round(fair - OSMIUM_MM_HALF + skew))
        ask_px = int(round(fair + OSMIUM_MM_HALF + skew))

        bid_px = min(bid_px, fair - 1)
        ask_px = max(ask_px, fair + 1)

        bid_qty = limit - position
        ask_qty = limit + position

        if bid_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", bid_px, bid_qty))
        if ask_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", ask_px, -ask_qty))

        return orders

    # =============================================================
    #  INTARIAN_PEPPER_ROOT - Max Long Trend Capture
    #
    #  Teammate's core insight: hold max long, trend does the work.
    #  + Our weighted mid for more accurate fair value
    #  + Slightly faster EMA adaptation (0.008 vs 0.005)
    # =============================================================

    def trade_pepper(self, od: OrderDepth, position: int, limit: int,
                     timestamp: int, trader_state: dict) -> tuple[List[Order], dict]:
        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None and best_ask is None:
            return orders, trader_state

        # Weighted mid (from our v3) - more accurate fair value
        if best_bid is not None and best_ask is not None:
            bid_vol = abs(od.buy_orders[best_bid])
            ask_vol = abs(od.sell_orders[best_ask])
            if bid_vol + ask_vol > 0:
                mid = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
            else:
                mid = (best_bid + best_ask) / 2
        else:
            mid = float(best_bid or best_ask)

        # -- Single-tick bootstrap (from teammate) ----------------
        pepper_key = "pepper_intercept"
        if pepper_key not in trader_state:
            trader_state[pepper_key] = mid - timestamp * PEPPER_SLOPE
            logger.print(f"PEPPER bootstrap intercept={trader_state[pepper_key]:.1f}")

        intercept = trader_state[pepper_key]

        # -- Fair value + gentle EMA ------------------------------
        fair = intercept + timestamp * PEPPER_SLOPE
        residual = mid - fair
        trader_state[pepper_key] = intercept + 0.008 * residual

        logger.print(f"PEPPER t={timestamp} fair={fair:.1f} mid={mid:.1f} pos={position}")

        # -- 1. Aggressive BUY: take any ask <= fair + BUY_EDGE ----
        # Even buying above fair is fine - trend recovers the cost
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask <= fair + PEPPER_BUY_EDGE and position < limit:
                    qty = min(-od.sell_orders[ask], limit - position)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                        position += qty
                        logger.print(f"PEPPER BUY {qty}@{ask} fair={fair:.1f}")
                else:
                    break

        # -- 2. Spike sell: only if bid far above fair, never go short --
        if od.buy_orders and position > 0:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid > fair + PEPPER_SELL_SPIKE:
                    qty = min(od.buy_orders[bid], position)  # Never go short
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                        position -= qty
                        logger.print(f"PEPPER SPIKE SELL {qty}@{bid} fair={fair:.1f}")
                else:
                    break

        # -- 3. Passive BUY: fill remaining capacity above fair ---
        # Bid above mid to be best bid and attract sellers
        remaining = limit - position
        if remaining > 0:
            passive_bid = int(round(fair + PEPPER_PASSIVE_BID_ABOVE))
            if best_ask is not None:
                passive_bid = min(passive_bid, best_ask - 1)
            orders.append(Order("INTARIAN_PEPPER_ROOT", passive_bid, remaining))

        # NO passive sell - hold the long for the trend
        return orders, trader_state
