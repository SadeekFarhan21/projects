"""Download only EuRoC MH_01_easy (ASL format) into data/.

The official ETH host now serves one 12 GB zip per environment. We read just
the MH_01_easy.zip member out of that archive with HTTP range requests
(remotezip), so only about 1.6 GB is transferred, then unpack it.

Usage:  uv run python scripts/download_euroc.py [--seq MH_01_easy]
"""
import argparse
import pathlib
import shutil
import zipfile

from remotezip import RemoteZip

MIRROR = "https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/machine_hall.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MH_01_easy")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inner = out / f"{args.seq}.zip"
    target = out / args.seq
    if (target / "mav0").exists():
        print(f"{target} already present")
        return
    if not inner.exists():
        member = f"machine_hall/{args.seq}/{args.seq}.zip"
        print(f"fetching {member} via range requests")
        with RemoteZip(MIRROR) as z:
            with z.open(member) as src, open(inner.with_suffix(".part"), "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
        inner.with_suffix(".part").rename(inner)
    print("unpacking")
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(inner) as z:
        z.extractall(target)
    print("done", target)


if __name__ == "__main__":
    main()
