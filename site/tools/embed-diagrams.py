#!/usr/bin/env python3
"""Embed Excalidraw renders in the posts.

For each diagram in ALT below:
  diagrams/png/<name>.png  (4-5k px wide render, kept out of source/ so it is not deployed)
  -> source/img/diagrams/<name>.webp  (2400 px wide, what the posts load)
and every <figure data-figure="diagram:<name>"></figure> in source/_posts is
replaced with an <img> figure that opens full size on click.

Re-render a diagram after editing site/diagrams/<name>.excalidraw with the
site-local renderer, which swaps in the site's fonts (see its template):
  <excalidraw skill venv>/bin/python tools/excalidraw/render_excalidraw.py \
    diagrams/<name>.excalidraw --output diagrams/png/<name>.png --scale 2
then run this script again. In the .excalidraw files, fontFamily 2 marks text
(rendered in Quicksand) and fontFamily 3 marks code (rendered in Fragment Mono).
"""
import pathlib
import re
import subprocess
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent
PNG = SITE / "diagrams" / "png"
IMG = SITE / "source" / "img" / "diagrams"
POSTS = SITE / "source" / "_posts"
WIDTH = 2400

ALT = {
    "backtester-pipeline": "Flow of the qrp backtester from the Binance archive through a point-in-time store and wide arrays to a look-ahead audit that stops leaky runs, then purged cross-validation, ridge, three matching backtest kernels, metrics and the run tracker.",
    "crypto-pipeline": "Crypto reversal pipeline from the Binance archive to one forecast that splits into a rank IC path, where H1 is supported, and a dollar backtest path, where H2 is not supported after costs, above a timeline of the development period and the one-time holdout.",
    "peeking-pipeline": "Two independent pipelines that meet only at results: the A/B side reduces event tables to per-arm sufficient statistics for Welch, CUPED and mSPRT tests; the off-policy side turns bandit logs and a cross-fitted reward model into per-round terms for IPS, DR and bootstrap intervals.",
    "sv39-translation": "Sv39 address translation: a 39-bit virtual address split into three 9-bit table indices and a 12-bit page offset, walked through the L2, L1 and L0 page tables to a leaf entry with its physical page number and permission bits.",
    "riscv-memory-map": "Physical memory map after a 128 MiB boot: the test device, PLIC and UART, then OpenSBI, the kernel's text, rodata and data sections with their permissions, guarded boot stacks, the page bitmap, free frames and the device tree blob.",
    "riscv-trap-path": "The kernel's single trap path: kernelvec saves a frame on the interrupted thread's stack, kernel_trap dispatches timer and external interrupts, probe recoveries and fatal exceptions, and kernelvec restores and returns with sret.",
    "riscv-thread-states": "Kernel thread state machine: UNUSED to RUNNABLE on thread_create, RUNNABLE and RUNNING via the scheduler and yield or preemption, RUNNING to SLEEPING and back to RUNNABLE on wakeup, and RUNNING to ZOMBIE to UNUSED on exit and join.",
    "kvdb-write-and-recovery": "kvdb write and recovery paths as a sequence over three files, P-wal, P-journal and P, with the commit point after Wal::append and sync and the atomicity point after the journal header sync marked across all files.",
    "kvdb-slotted-page": "Layout of a 4096-byte kvdb slotted page: a 16-byte header, 2-byte slot offsets growing forward, free space, and cells of key length, value length, key and value growing backward from the end.",
    "redis-event-loop": "The single-threaded event loop of the Redis clone: commands fill an AOF buffer and per-client reply buffers, and before_sleep runs Aof::flush before try_write on every iteration, so no reply leaves before its write is durable.",
    "matching-engine": "Matching engine architecture: an input log feeds the engine's validate, match and rest or cancel steps over its id map, order pool and two book sides; output events tagged with the input sequence number fan out to the output log, market data feed and digest, while a reference matcher must produce identical outputs.",
    "order-book-levels": "Order book layout: bid and ask sides as vectors of price levels sorted worst to best with back() as the best price, and one level's FIFO of order pool slots 17, 4 and 52, doubly linked by slot index.",
    "market-sim": "Market simulator architecture: a scheduler event heap drives the fundamental and the trader agents, which trade through the market and its order book; the market maker receives fills and requotes; the hidden fundamental value is read only by informed traders and analysis stamps.",
    "smolgrad-layers": "smolgrad package layers: user code calls nn, optim and gradcheck; nn goes through functional into the Tensor class, whose forward ops record a closure and whose backward sweeps them in reverse, while optim reads grad and updates data in place outside the graph; numpy sits underneath.",
    "smolgrad-step": "One smolgrad training step: the input flows through Linear, relu, Linear and cross_entropy to a scalar loss, backward runs the recorded closures in reverse to fill the leaf gradients, step updates the weights in place, and zero_grad clears the gradients for the next batch.",
    "kafka-sparse-index": "A sparse index lookup: index entries point to byte positions in the log, and fetching offset 6 binary searches to the floor entry at record 4, skips the headers of records 4 and 5, and returns records from byte 6000.",
    "kafka-broker": "Kafka-style broker architecture: producer and consumer connections each get a broker thread that dispatches to the topics map, group coordinator, offset store and long-poll wakeups; data and committed offsets end up as segment .log and .index files on disk.",
}


def main() -> int:
    missing = [n for n in ALT if not (PNG / f"{n}.png").exists()]
    if missing:
        print("renders missing:", ", ".join(missing))
        return 1

    for name in ALT:
        png, webp = PNG / f"{name}.png", IMG / f"{name}.webp"
        tmp = IMG / f".{name}.resized.png"
        subprocess.run(["sips", "--resampleWidth", str(WIDTH), str(png), "--out", str(tmp)], check=True, capture_output=True)
        subprocess.run(["cwebp", "-quiet", "-q", "88", "-m", "6", str(tmp), "-o", str(webp)], check=True)
        tmp.unlink()
        w, h = (int(x) for x in subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(webp)], capture_output=True, text=True
        ).stdout.split()[-3::2])
        print(f"{name:26} {png.stat().st_size // 1024:5} KB png -> {webp.stat().st_size // 1024:4} KB webp ({w}x{h})")
        ALT[name] = (ALT[name], w, h)

    swapped = 0
    for post in sorted(POSTS.glob("*.md")):
        text = post.read_text()

        def repl(m: re.Match) -> str:
            nonlocal swapped
            name = m.group(1) or m.group(2)
            alt, w, h = ALT[name]
            swapped += 1
            return (
                f'<figure class="excal" data-diagram="{name}">'
                f'<a href="/img/diagrams/{name}.webp" class="excal-link" aria-label="Open the diagram full size">'
                f'<img src="/img/diagrams/{name}.webp" alt="{alt}" width="{w}" height="{h}" loading="lazy" decoding="async">'
                f"</a></figure>"
            )

        # Old placeholders, and figures from an earlier run (their size may have changed).
        new = re.sub(
            r'<figure data-figure="diagram:([a-z0-9-]+)"></figure>'
            r'|<figure class="excal" data-diagram="([a-z0-9-]+)">.*?</figure>',
            repl, text)
        if new != text:
            post.write_text(new)
    print(f"swapped {swapped} figures")
    left = [p.name for p in POSTS.glob("*.md") if 'data-figure="diagram:' in p.read_text()]
    if left:
        print("still unswapped in:", ", ".join(left))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
