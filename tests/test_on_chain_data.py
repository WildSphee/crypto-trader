import pandas as pd
import pytest

from features.on_chain_data import (
    BTC_METRICS,
    SUPPORTED_INTERVALS,
    get_btc_onchain_smoothed,
)

# ---- Fixtures & helpers ----


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")


@pytest.fixture
def charts_payload_three_days():
    # Provide three days (2025-07-01..03) for any blockchain.info chart endpoint
    base_days = pd.date_range("2025-07-01", "2025-07-03", freq="1D", tz="UTC")

    # Create values increasing per day; each slug will map to its own series but we just need any numeric
    def mk_values(mult=1.0, start=1.0):
        vals = []
        cur = start
        for ts in base_days:
            vals.append({"x": int(ts.timestamp()), "y": cur})
            cur += mult
        return vals

    # Return a function per-slug at runtime, but our mock will reuse this fixture by generating per call.
    return {"_template_days": base_days, "_mk_values": mk_values}


@pytest.fixture
def binance_klines_payload():
    # Minimal valid klines array (open_time, open, high, low, close, volume, ...); only a few rows needed
    def make(symbol, interval, start_ts, end_ts):
        idx = pd.date_range(start_ts, end_ts, tz="UTC", freq=interval)
        rows = []
        price = 50000.0
        for t in idx:
            open_ms = int(t.timestamp() * 1000)
            rows.append(
                [
                    open_ms,
                    str(price),
                    str(price * 1.01),
                    str(price * 0.99),
                    str(price * 1.005),
                    "10.0",
                    open_ms + int(60e3),
                    "0",
                    100,
                    "0",
                    "0",
                    "0",
                ]
            )
            price *= 1.001
        return rows

    return make


def _mock_requests_get(monkeypatch, charts_payload_three_days, binance_klines_payload):
    """
    Patch requests.get to return:
      - blockchain.info/charts/* -> JSON with "values"
      - api.binance.com/api/v3/klines -> klines array
    """
    import requests

    class _FakeResp:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("HTTP error")

    def fake_get(url, params=None, headers=None, timeout=30):
        # blockchain charts
        if "blockchain.info/charts/" in url:
            slug = url.rstrip("/").split("/")[-1]
            # determine growth rate per slug for deterministic but different series
            mult = (hash(slug) % 5 + 1) * 0.5  # 0.5..2.5 step
            values = charts_payload_three_days["_mk_values"](mult=mult, start=1.0)
            payload = {"status": "ok", "values": values}
            return _FakeResp(payload, 200)

        # binance klines
        if "api.binance.com/api/v3/klines" in url:
            interval = params.get("interval")
            # Map to pandas offset alias expected by the fixture generator
            to_pandas = {
                "1m": "1min",
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1H",
                "2h": "2H",
                "4h": "4H",
                "6h": "6H",
                "8h": "8H",
                "12h": "12H",
                "1d": "1D",
            }[interval]
            start_ts = pd.to_datetime(params["startTime"], unit="ms", utc=True)
            end_ts = pd.to_datetime(params["endTime"], unit="ms", utc=True)

            # ✅ Call the callable returned by the fixture directly
            rows = binance_klines_payload(
                symbol=params.get("symbol", "BTCUSDT"),
                interval=to_pandas,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            return _FakeResp(rows, 200)

        return _FakeResp({}, 404)

    monkeypatch.setattr(requests, "get", fake_get)


@pytest.mark.parametrize("interval", sorted(SUPPORTED_INTERVALS))
def test_onchain_grid_and_columns(
    monkeypatch, charts_payload_three_days, binance_klines_payload, interval
):
    _mock_requests_get(monkeypatch, charts_payload_three_days, binance_klines_payload)

    start, end = "2025-07-01", "2025-07-03"
    df = get_btc_onchain_smoothed(
        start=start,
        end=end,
        interval=interval,
        metrics=None,  # use all metrics
        lookback_days=2,
        agg_to_daily="mean",
        include_price=False,  # no price first
    )

    # Ensure all *_smooth columns for chosen metrics exist
    expected_cols = [v + "_smooth" for v in BTC_METRICS.values()]
    for c in expected_cols:
        assert c in df.columns

    # Index tz and bounds
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert pd.Timestamp(start, tz="UTC") <= df.index[0] <= pd.Timestamp(end, tz="UTC")
    assert df.index[-1] <= pd.Timestamp(end, tz="UTC")

    # No empties and some non-NaN values after ffill
    assert not df.empty
    assert df.notna().any().any()


@pytest.mark.parametrize("interval", ["4h", "1d", "1m"])
def test_onchain_include_price_joins(
    monkeypatch, charts_payload_three_days, binance_klines_payload, interval
):
    _mock_requests_get(monkeypatch, charts_payload_three_days, binance_klines_payload)

    start, end = "2025-07-01", "2025-07-03"
    df = get_btc_onchain_smoothed(
        start=start,
        end=end,
        interval=interval,
        include_price=True,
        price_symbol="BTCUSDT",
    )

    # Price columns should be present (left-join, may include NaNs if kline grid sparse)
    for c in ["price_open", "price_high", "price_low", "price_close", "price_volume"]:
        assert c in df.columns


@pytest.mark.parametrize("bad_interval", ["10m", "7h", "2d", "foo"])
def test_onchain_bad_interval_raises(
    monkeypatch, charts_payload_three_days, binance_klines_payload, bad_interval
):
    _mock_requests_get(monkeypatch, charts_payload_three_days, binance_klines_payload)

    with pytest.raises(ValueError):
        get_btc_onchain_smoothed(
            start="2025-07-01",
            end="2025-07-03",
            interval=bad_interval,
        )
