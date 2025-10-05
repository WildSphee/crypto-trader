import pandas as pd
import pytest

from features.fear_index import get_fng_features, SUPPORTED_INTERVALS

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


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, params=None, timeout=30):
        # Only endpoint used: https://api.alternative.me/fng/
        return _FakeResp(self._payload, 200)


@pytest.fixture
def fng_payload_three_days():
    # Build 3 days of Fear/Neutral/Greed spanning 2025-07-01..2025-07-03
    days = [
        ("2025-07-01", 25, "Extreme Fear"),
        ("2025-07-02", 50, "Neutral"),
        ("2025-07-03", 75, "Greed"),
    ]
    records = []
    for d, val, label in days:
        ts = int(pd.Timestamp(d, tz="UTC").timestamp())
        records.append(
            {"timestamp": str(ts), "value": str(val), "value_classification": label}
        )
    return {"name": "fng", "data": records, "metadata": {"error": None}}


def _expected_grid(start, end, interval):
    # Mirror how the function creates the target index
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if interval == "1d":
        return pd.date_range(start=start_ts, end=end_ts, freq="1D", tz="UTC")
    # for sub-daily: forward-fill onto exact frequency
    freq_map = {
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
    }
    return pd.date_range(start=start_ts, end=end_ts, freq=freq_map[interval], tz="UTC")


# ---- Tests ----


@pytest.mark.parametrize("interval", sorted(SUPPORTED_INTERVALS))
def test_fng_basic_grid_and_columns(interval, fng_payload_three_days):
    start, end = "2025-07-01", "2025-07-03"
    fake_sess = _FakeSession(fng_payload_three_days)

    df = get_fng_features(
        start=start,
        end=end,
        interval=interval,
        lookback_days=2,
        past_only=True,
        session=fake_sess,
    )

    # Required columns
    assert list(df.columns) == [
        "fng_score",
        "fng_label",
        "fng_ordinal",
        "fng_ordinal_smooth",
        "fng_ordinal_smooth_int",
    ]

    # Index checks
    assert df.index.tz is not None and str(df.index.tz) == "UTC"

    # Grid alignment and bounds
    exp_idx = _expected_grid(start, end, interval)
    assert df.index[0] == exp_idx[0]
    assert df.index[-1] == exp_idx[-1]

    # Non-empty and forward-filled (values should not be all-NaN)
    assert not df.empty
    assert df["fng_ordinal"].notna().any()


@pytest.mark.parametrize("interval", ["1d", "4h", "1h"])
def test_fng_smoothing_is_past_only(interval, fng_payload_three_days):
    start, end = "2025-07-01", "2025-07-03"
    fake_sess = _FakeSession(fng_payload_three_days)

    df = get_fng_features(
        start=start,
        end=end,
        interval=interval,
        lookback_days=2,
        past_only=True,
        session=fake_sess,
    )

    # Find the first timestamp for 2025-07-01 in the grid:
    first_day = pd.Timestamp("2025-07-01", tz="UTC")
    view = df.loc[df.index >= first_day]
    first_row = view.iloc[0]

    # Since past_only=True and this is the first day, smoothed should be based on prior data only (none),
    # so it's allowed to be NaN.
    assert pd.isna(first_row["fng_ordinal_smooth"]) or first_row[
        "fng_ordinal_smooth"
    ] == pytest.approx(first_row["fng_ordinal"])


def test_fng_invalid_interval_raises(fng_payload_three_days):
    fake_sess = _FakeSession(fng_payload_three_days)
    with pytest.raises(ValueError):
        get_fng_features("2025-07-01", "2025-07-03", interval="10h", session=fake_sess)
