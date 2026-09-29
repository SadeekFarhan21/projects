import pandas as pd

from marketpred.data import parse_kline_csv

ROW_MS = "1704067200000,42283.58,44184.1,42180.77,44179.55,27174.3,1704153599999,1169000000.0,1000,1,1,0\n"
ROW_US = "1735689600000000,93576.0,95151.15,92888.0,94591.79,10373.3,1735775999999999,975000000.0,1000,1,1,0\n"


def test_parse_millis_and_micros():
    a = parse_kline_csv(ROW_MS.encode(), "BTCUSDT")
    b = parse_kline_csv(ROW_US.encode(), "BTCUSDT")
    assert a["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert b["date"].iloc[0] == pd.Timestamp("2025-01-01")
    assert a["close"].iloc[0] == 44179.55
    assert b["quote_volume"].iloc[0] == 975000000.0


def test_parse_skips_header_row():
    hdr = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbb,tbq,ignore\n"
    df = parse_kline_csv((hdr + ROW_MS).encode(), "BTCUSDT")
    assert len(df) == 1 and df["symbol"].iloc[0] == "BTCUSDT"
