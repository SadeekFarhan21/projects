"""Download Binance daily klines for all USDT spot pairs (about 10 MB of zips)."""
import argparse
import time

from marketpred.data import download

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.perf_counter()
    download(a.start, a.end, a.workers)
    print(f"done in {time.perf_counter() - t0:.1f}s")
