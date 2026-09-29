"""Download MNIST (about 11 MB gzipped) and Tiny Shakespeare (about 1.1 MB) into data/.

Usage: uv run python scripts/download_data.py

Mirrors are tried in order. Files are verified by SHA-256 so a truncated or
tampered download is caught before training silently runs on garbage.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(urls: list[str], dest: Path, expected_sha: str | None = None) -> None:
    if dest.exists() and (expected_sha is None or sha256(dest) == expected_sha):
        print(f"ok (cached) {dest.relative_to(DATA.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            if expected_sha is not None and sha256(tmp) != expected_sha:
                print(f"  checksum mismatch from {url}, trying next mirror")
                tmp.unlink()
                continue
            tmp.rename(dest)
            print(f"ok {dest.relative_to(DATA.parent)} ({dest.stat().st_size / 1e6:.2f} MB)")
            return
        except Exception as e:  # network errors: try the next mirror
            print(f"  failed: {e}")
    sys.exit(f"could not download {dest.name} from any mirror")


def main() -> None:
    for name, sha in MNIST_FILES.items():
        fetch([m + name for m in MNIST_MIRRORS], DATA / "mnist" / name, sha)
    fetch([SHAKESPEARE_URL], DATA / "tinyshakespeare.txt")


if __name__ == "__main__":
    main()
