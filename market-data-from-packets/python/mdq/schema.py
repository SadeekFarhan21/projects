"""Binary record layouts written by mdp_decode and the Parquet schema built from them.

The numpy dtypes below must match the packed C++ structs in
include/mdp/records.hpp byte for byte. to_parquet.py checks each dtype's
itemsize against the record_size that mdp_decode writes into manifest.json.

Conventions for every Parquet table:
  ts        timestamp[ns, UTC]  event time from the IEX message (IEX Trading System clock)
  seq       int64               IEX-TP message sequence number (unique per feed and session)
  symbol    string (dictionary) trimmed Nasdaq Integrated symbol
  *_price   int64               price in 1e-4 dollars (123400 = $12.34), exactly as sent by IEX
  *_size    int64               shares
  single-byte enums are 1-character strings; raw flag bytes are kept as uint8
"""

from __future__ import annotations

import numpy as np

# name -> (numpy dtype of the packed .bin record, description)
DTYPES: dict[str, np.dtype] = {
    "quotes": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("bid_price", "<i8"), ("ask_price", "<i8"),
         ("bid_size", "<u4"), ("ask_size", "<u4"), ("flags", "u1")]
    ),
    "deep_levels": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("price", "<i8"), ("size", "<u4"),
         ("side", "u1"), ("flags", "u1")]
    ),
    "bbo_from_deep": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("bid_price", "<i8"), ("ask_price", "<i8"),
         ("bid_size", "<u4"), ("ask_size", "<u4"), ("bid_levels", "<u2"), ("ask_levels", "<u2")]
    ),
    "trades": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("price", "<i8"), ("trade_id", "<i8"),
         ("size", "<u4"), ("flags", "u1"), ("msg_type", "u1")]
    ),
    "system_events": np.dtype([("ts", "<i8"), ("seq", "<i8"), ("event", "u1")]),
    "security_directory": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("adjusted_poc_price", "<i8"),
         ("round_lot", "<u4"), ("flags", "u1"), ("luld_tier", "u1")]
    ),
    "trading_status": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("reason", "S4"), ("status", "u1")]
    ),
    "security_status": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("msg_type", "u1"), ("status", "u1"),
         ("detail", "u1")]
    ),
    "official_prices": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("price", "<i8"), ("price_type", "u1")]
    ),
    "auctions": np.dtype(
        [("ts", "<i8"), ("seq", "<i8"), ("symbol", "<u8"), ("reference_price", "<i8"),
         ("indicative_clearing_price", "<i8"), ("auction_book_clearing_price", "<i8"),
         ("collar_reference_price", "<i8"), ("lower_auction_collar", "<i8"), ("upper_auction_collar", "<i8"),
         ("paired_shares", "<u4"), ("imbalance_shares", "<u4"), ("scheduled_auction_time", "<u4"),
         ("auction_type", "u1"), ("imbalance_side", "u1"), ("extension_number", "u1")]
    ),
    "unknown_messages": np.dtype([("send_time", "<i8"), ("seq", "<i8"), ("length", "<u2"), ("msg_type", "u1")]),
    "segments": np.dtype(
        [("capture_ts", "<i8"), ("send_time", "<i8"), ("first_seq", "<i8"), ("stream_offset", "<i8"),
         ("channel_id", "<u4"), ("session_id", "<u4"), ("protocol_id", "<u2"), ("message_count", "<u2"),
         ("payload_length", "<u2"), ("skipped", "<u2")]
    ),
}

# Columns stored as 1-character strings (single-byte ASCII enums).
CHAR_COLUMNS = {"side", "msg_type", "event", "status", "detail", "price_type", "auction_type", "imbalance_side"}
# security_status.status is a char for I/O/E but a 0/1 byte for P; keep it as a char of the raw byte
# except that 0 and 1 map to "0" and "1".
TIMESTAMP_COLUMNS = {"ts", "send_time", "capture_ts"}
# Monotone or near-monotone int64 columns: delta encoding beats dictionary encoding.
DELTA_COLUMNS = {"ts", "seq", "trade_id", "send_time", "capture_ts", "first_seq", "stream_offset"}

# Derived boolean flag columns: table -> {column: (source column, mask)}.
FLAG_COLUMNS = {
    "quotes": {"halted": ("flags", 0x80), "pre_post_market": ("flags", 0x40)},
    "trades": {
        "iso": ("flags", 0x80),
        "extended_hours": ("flags", 0x40),
        "odd_lot": ("flags", 0x20),
        "trade_through_exempt": ("flags", 0x10),
        "single_price_cross": ("flags", 0x08),
    },
    "deep_levels": {"event_complete": ("flags", 0x01)},
    "security_directory": {"test_security": ("flags", 0x80), "when_issued": ("flags", 0x40), "etp": ("flags", 0x20)},
}

TABLE_DOCS = {
    "quotes": "TOPS Quote Update: IEX best bid and offer after each change (flags 0x80 halted or unavailable, 0x40 pre or post market).",
    "deep_levels": "DEEP Price Level Update: aggregate displayed size at one price on one side; size 0 removes the level; event_complete marks the end of an atomic book transition.",
    "bbo_from_deep": "Best bid and offer derived from the DEEP book, one row each time the consistent BBO changes at an event boundary (event_complete set). bid_levels and ask_levels are book depth after the event.",
    "trades": "Trade Report (msg_type T) and Trade Break (msg_type B) messages, one row per fill.",
    "system_events": "System Event: O start of messages, S start of system hours, R start of regular hours, M end of regular hours, E end of system hours, C end of messages.",
    "security_directory": "Security Directory for IEX-listed securities: round lot, adjusted previous official close, LULD tier.",
    "trading_status": "Trading Status: H halted, O order acceptance period, P paused, T trading; reason code for halts.",
    "security_status": "Retail Liquidity Indicator (I), Operational Halt (O), Short Sale Price Test (P, with detail) and Security Event (E) messages.",
    "official_prices": "Official opening (Q) and closing (M) prices for IEX-listed securities.",
    "auctions": "Auction Information for IEX-listed securities during auction periods.",
    "unknown_messages": "Messages with an unknown type byte or shorter than the spec minimum (should be empty).",
    "segments": "One row per IEX-TP segment (UDP datagram), including heartbeats: capture time, send time, sequence and stream offset.",
}
