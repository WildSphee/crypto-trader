from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from binance.client import Client

# from binance.enums import *


@dataclass
class SizingConfig:
    # Assumed typical bar move used for a naive Kelly fraction denominator
    typical_bar_return: float = 0.003  # 0.3%
    kelly_cap: float = 0.20  # max fraction of capital to risk
    min_notional_usdt: float = (
        5.0  # Binance min notional (example; check symbol filters)
    )


class PositionManager:
    """
    Handles position state, sizing, and order placement (long-only).
    """

    def __init__(self, client: Client, symbol: str, sizing: SizingConfig) -> None:
        self.client = client
        self.symbol = symbol
        self.sizing = sizing
        self.is_long = False
        self.last_quantity = 0.0

        # Pull symbol filters to round quantities properly
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise RuntimeError(f"Symbol info not found for {symbol}")

        self.step_size = 0.0
        self.min_qty = 0.0
        self.min_notional = self.sizing.min_notional_usdt
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                self.step_size = float(f["stepSize"])
                self.min_qty = float(f["minQty"])
            elif f["filterType"] == "NOTIONAL":
                # Futures uses NOTIONAL filter; spot often doesn't provide this one
                try:
                    self.min_notional = max(
                        self.min_notional,
                        float(f.get("minNotional", self.min_notional)),
                    )
                except Exception:
                    pass

        if self.step_size <= 0:
            # Fallback to a very small step if API doesn't provide one (rare)
            self.step_size = 1e-6

    def _round_qty(self, qty: float) -> float:
        """
        Round quantity to the symbol step size, and ensure >= minQty.
        """
        step = self.step_size
        rounded = math.floor(qty / step) * step
        if rounded < self.min_qty:
            return 0.0
        return float(rounded)

    def _account_balances(self) -> Tuple[float, float]:
        """
        Returns (free_usdt, free_base) balances.
        """
        usdt = float(self.client.get_asset_balance(asset="USDT")["free"])
        # base asset (e.g., BTC in BTCUSDT)
        base_asset = self.symbol.replace("USDT", "")
        base = float(self.client.get_asset_balance(asset=base_asset)["free"])
        return usdt, base

    def compute_quantity_kelly(self, up_prob: float, last_price: float) -> float:
        """
        Kelly fraction for a binary bet with symmetric payoff approximated
        by typical_bar_return (very rough). Clip to [0, kelly_cap].
        """
        edge = up_prob - (1.0 - up_prob)  # = 2*up_prob - 1
        denom = max(self.sizing.typical_bar_return, 1e-6)
        f_star = edge / denom
        f_star = max(0.0, min(f_star, self.sizing.kelly_cap))  # clip

        free_usdt, _ = self._account_balances()
        # ensure we respect min notional
        dollar_alloc = max(self.sizing.min_notional_usdt, free_usdt * f_star)
        qty = dollar_alloc / last_price
        return self._round_qty(qty)

    # ---------- Orders ----------

    def open_long(self, qty: float) -> Optional[dict]:
        if qty <= 0:
            print("Open long skipped: quantity rounded to 0.")
            return None
        try:
            order = self.client.order_market_buy(symbol=self.symbol, quantity=qty)
            self.is_long = True
            self.last_quantity = qty
            print(f"[ORDER] LONG buy {qty} {self.symbol}")
            return order
        except Exception as e:
            print(f"[ERROR] open_long failed: {e}")
            return None

    def close_long(self) -> Optional[dict]:
        if not self.is_long or self.last_quantity <= 0:
            return None
        try:
            order = self.client.order_market_sell(
                symbol=self.symbol, quantity=self.last_quantity
            )
            print(f"[ORDER] Close LONG sell {self.last_quantity} {self.symbol}")
            self.is_long = False
            self.last_quantity = 0.0
            return order
        except Exception as e:
            print(f"[ERROR] close_long failed: {e}")
            return None
