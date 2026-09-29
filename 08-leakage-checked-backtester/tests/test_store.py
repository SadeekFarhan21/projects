"""Partitioned parquet store and its point-in-time read path."""

from datetime import date, datetime

import polars as pl
import pytest

from qrp.data.binance import _to_millis
from qrp.data.store import BarStore
from qrp.panel import Panel


def _bars(symbols=("AAA", "BBB"), start=date(2023, 12, 28), n=10, px=100.0):
    rows = []
    for s in symbols:
        for k in range(n):
            d = pl.date_range(start, start, eager=True)[0]
            d = date.fromordinal(start.toordinal() + k)
            rows.append({"symbol": s, "date": d, "open": px + k, "high": px + k + 1,
                         "low": px + k - 1, "close": px + k, "volume": 10.0,
                         "quote_volume": 1000.0, "trades": 5,
                         "available_at": datetime(d.year, d.month, d.day, 23, 59, 59, 999000)})
    return pl.DataFrame(rows)


def test_roundtrip_and_year_partitions(tmp_path):
    st = BarStore(tmp_path)
    st.ingest(_bars(), "test", ingested_at=datetime(2024, 2, 1))
    years = sorted(p.name for p in (tmp_path / "bars" / "freq=1d").iterdir())
    assert years == ["year=2023", "year=2024"]
    df = st.load()
    assert df.height == 20
    assert df["close"].to_list()[:3] == [100.0, 101.0, 102.0]
    # partition pruning path gives the same rows as a filter
    part = st.load(start=date(2024, 1, 1))
    assert part.height == df.filter(pl.col("date") >= date(2024, 1, 1)).height


def test_restatement_is_point_in_time(tmp_path):
    st = BarStore(tmp_path)
    st.ingest(_bars(), "v1", ingested_at=datetime(2024, 2, 1))
    fixed = _bars().with_columns(
        pl.when((pl.col("symbol") == "AAA") & (pl.col("date") == date(2024, 1, 2)))
        .then(999.0).otherwise(pl.col("close")).alias("close"))
    st.ingest(fixed, "v2", ingested_at=datetime(2024, 3, 1))

    def px(as_of):
        d = st.load(as_of=as_of).filter((pl.col("symbol") == "AAA") &
                                        (pl.col("date") == date(2024, 1, 2)))
        return d["close"].item()

    assert px(datetime(2024, 2, 15)) == 105.0   # before the restatement
    assert px(datetime(2024, 3, 2)) == 999.0    # after it
    assert px(None) == 999.0
    assert st.load().height == 20               # no duplicate versions leak through
    assert st.snapshot_id(datetime(2024, 2, 15)) != st.snapshot_id(None)


def test_bar_not_visible_before_it_closes(tmp_path):
    st = BarStore(tmp_path)
    st.ingest(_bars(), "v1", ingested_at=datetime(2023, 12, 1))  # ingested "early"
    df = st.load(as_of=datetime(2023, 12, 30, 12, 0))
    # bars for 12-28 and 12-29 closed before noon on 12-30; 12-30 itself has not closed
    assert sorted(set(df["date"].to_list())) == [date(2023, 12, 28), date(2023, 12, 29)]


def test_symbols_metadata_is_point_in_time(tmp_path):
    st = BarStore(tmp_path)
    st.ingest(_bars(("AAA",)), "v1", ingested_at=datetime(2024, 1, 20))
    later = _bars(("CCC",), start=date(2024, 1, 5), n=5)
    st.ingest(later, "v2", ingested_at=datetime(2024, 1, 25))
    assert st.symbols(datetime(2024, 1, 21))["symbol"].to_list() == ["AAA"]
    assert st.symbols()["symbol"].to_list() == ["AAA", "CCC"]


def test_ingest_validation(tmp_path):
    st = BarStore(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        st.ingest(pl.concat([_bars(), _bars()]), "dup")
    with pytest.raises(ValueError, match="sanity"):
        st.ingest(_bars().with_columns(pl.lit(-1.0).alias("close")), "bad")
    with pytest.raises(ValueError, match="missing"):
        st.ingest(_bars().drop("close"), "bad")


def test_panel_from_long_keeps_gaps_as_nan(tmp_path):
    df = _bars().filter(~((pl.col("symbol") == "BBB") & (pl.col("date") == date(2024, 1, 1))))
    p = Panel.from_long(df)
    assert p.shape == (10, 2)
    import numpy as np
    assert np.isnan(p["close"][4, 1]) and np.isfinite(p["close"][4, 0])


def test_binance_microsecond_timestamps():
    import numpy as np
    ms = np.array([1735689600000, 1735689600000000])  # 2025-01-01 in ms and in us
    np.testing.assert_array_equal(_to_millis(ms), [1735689600000, 1735689600000])


def test_relisted_ticker_becomes_new_instrument():
    """LUNA-style ticker reuse: the fake 177,000x return across the gap must disappear."""
    import numpy as np
    from qrp.backtest.engine import CostModel, run_backtest
    from qrp.panel import split_relisted
    a = _bars(("LUNA",), start=date(2022, 5, 1), n=5, px=100.0)
    b = _bars(("LUNA",), start=date(2022, 5, 20), n=5, px=5e6)
    df = pl.concat([a, b])
    p_raw = Panel.from_long(df)
    p = Panel.from_long(split_relisted(df, 3))
    assert p.symbols == ["LUNA", "LUNA#2"]
    assert np.isfinite(p["close"][:5, 0]).all() and np.isnan(p["close"][5:, 0]).all()
    W = np.full((p_raw.shape[0], 1), -1.0)  # short the dying coin and keep the target
    raw = run_backtest(W, p_raw["close"], CostModel(0, 0), delay=0)
    assert raw.equity[-1] == 0.0            # the fake relisting jump wipes the book out
    W2 = np.zeros(p.shape)
    W2[:, 0] = -1.0
    fixed = run_backtest(W2, p["close"], CostModel(0, 0), delay=0)
    assert fixed.equity[-1] > 0.9           # short held at the last real price
