#!/usr/bin/env python3
"""End-to-end test of the kvd binary with an independent, stdlib-only RESP client.

redis-cli is not installed on the development machine, so this plays its role:
a second implementation of the protocol (written separately from the C++ one)
talking to the real server process over TCP. It also covers process-level
behaviour the in-process C++ tests cannot: SIGTERM shutdown, kill -9 crash
recovery with appendfsync=always, and a torn AOF tail.

Usage: client_test.py --server build/release/kvd
"""
import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time


class RespError(Exception):
    pass


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b""

    @staticmethod
    def encode(*args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            elif isinstance(a, int):
                a = str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        return b"".join(out)

    def _fill(self):
        chunk = self.sock.recv(65536)
        if not chunk:
            raise ConnectionError("server closed connection")
        self.buf += chunk

    def _line(self):
        while b"\r\n" not in self.buf:
            self._fill()
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def read(self):
        line = self._line()
        t, rest = line[:1], line[1:]
        if t == b"+":
            return rest.decode()
        if t == b"-":
            return RespError(rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n < 0:
                return None
            while len(self.buf) < n + 2:
                self._fill()
            data, self.buf = self.buf[:n], self.buf[n + 2:]
            return data
        if t == b"*":
            return [self.read() for _ in range(int(rest))]
        raise ValueError("bad reply type %r" % line)

    def call(self, *args):
        self.sock.sendall(self.encode(*args))
        return self.read()

    def close(self):
        self.sock.close()


ALL_PROCS = []


class ServerProc:
    def __init__(self, binary, workdir, *extra):
        self.aof = os.path.join(workdir, "test.aof")
        self.proc = subprocess.Popen(
            [binary, "--port", "0", "--aof", self.aof, *extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ALL_PROCS.append(self.proc)
        self.log = []
        deadline = time.time() + 10
        while time.time() < deadline:
            line = self.proc.stderr.readline()
            if not line:
                break
            self.log.append(line)
            m = re.search(r"kvd ready on [\d.]+:(\d+)", line)
            if m:
                self.port = int(m.group(1))
                return
        raise RuntimeError("server did not start: %s" % "".join(self.log))

    def stop(self, sig=signal.SIGTERM):
        self.proc.send_signal(sig)
        rest = self.proc.stderr.read()
        self.log.append(rest)
        return self.proc.wait(timeout=10)


CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def test_commands(c):
    check(c.call("PING") == "PONG", "PING")
    check(c.call("SET", "k", "v") == "OK", "SET")
    check(c.call("GET", "k") == b"v", "GET")
    check(c.call("GET", "missing") is None, "GET missing")
    check(c.call("EXISTS", "k", "missing", "k") == 2, "EXISTS counts duplicates")
    check(c.call("INCR", "n") == 1 and c.call("INCR", "n") == 2, "INCR")
    check(isinstance(c.call("INCR", "k"), RespError), "INCR on non-integer errors")
    check(c.call("EXPIRE", "k", 100) == 1, "EXPIRE")
    check(c.call("TTL", "k") == 100, "TTL")
    check(c.call("TTL", "n") == -1 and c.call("TTL", "nope") == -2, "TTL -1/-2")
    check(sorted(c.call("KEYS", "*")) == [b"k", b"n"], "KEYS *")
    check(c.call("DEL", "k", "n", "nope") == 2, "DEL")
    check(c.call("KEYS", "*") == [], "KEYS empty")
    check(isinstance(c.call("SET", "k"), RespError), "arity error")
    check(isinstance(c.call("FOO"), RespError), "unknown command")
    check(c.call("COMMAND", "DOCS") == [], "COMMAND DOCS (redis-cli handshake)")
    binary = bytes(range(256)) * 4
    check(c.call("SET", b"bin\x00key", binary) == "OK", "binary SET")
    check(c.call("GET", b"bin\x00key") == binary, "binary round trip")
    c.call("DEL", b"bin\x00key")


def test_ttl_realtime(c):
    # Generous margins everywhere: this machine can be very loaded.
    check(c.call("SET", "long", "x", "PX", 5000) == "OK", "SET PX")
    check(c.call("GET", "long") == b"x", "alive before deadline")
    check(0 < c.call("PTTL", "long") <= 5000, "PTTL in range")
    c.call("DEL", "long")
    check(c.call("SET", "short", "x", "PX", 100) == "OK", "SET PX short")
    time.sleep(0.2)
    check(c.call("GET", "short") is None, "gone after deadline (lazy)")
    # Active expiry: keys nobody reads must still disappear.
    c.sock.sendall(b"".join(Client.encode("SET", "a%d" % i, "v", "PX", 50) for i in range(2000)))
    for _ in range(2000):
        c.read()
    deadline = time.time() + 5  # poll: the machine may be heavily loaded
    while c.call("DBSIZE") != 0 and time.time() < deadline:
        time.sleep(0.05)
    check(c.call("DBSIZE") == 0, "active expiry reclaimed untouched keys")


def test_pipelining(c):
    n = 5000
    payload = b"".join(Client.encode("INCR", "pipe") for _ in range(n))
    c.sock.sendall(payload)  # all 5000 in one go, before reading anything
    got = [c.read() for _ in range(n)]
    check(got == list(range(1, n + 1)), "pipelined replies in order")
    # Inline protocol, as typed into nc/telnet.
    c.sock.sendall(b"SET inl 7\r\nGET inl\r\n")
    check(c.read() == "OK" and c.read() == b"7", "inline commands")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        # 1. Protocol and command behaviour.
        s = ServerProc(args.server, d)
        c = Client(s.port)
        test_commands(c)
        test_ttl_realtime(c)
        test_pipelining(c)
        c.call("SET", "persist-me", "yes")
        c.call("SET", "with-ttl", "yes", "EX", 500)
        c.close()
        check(s.stop() == 0, "clean SIGTERM exit")

        # 2. Graceful restart replays the AOF.
        s = ServerProc(args.server, d)
        c = Client(s.port)
        check(c.call("GET", "persist-me") == b"yes", "value survives restart")
        check(c.call("GET", "pipe") == b"5000", "counter survives restart")
        check(480 <= c.call("TTL", "with-ttl") <= 500, "TTL survives restart as absolute deadline")
        n = c.call("DBSIZE")
        check(n == 4, "only live keys come back (got %r)" % n)
        c.close()
        s.stop()

        # 3. Crash (kill -9) with appendfsync=always: every acknowledged write survives.
        s = ServerProc(args.server, d, "--appendfsync", "always")
        c = Client(s.port)
        for i in range(200):
            c.call("SET", "crash%d" % i, i)
        s.stop(signal.SIGKILL)
        c.close()
        s = ServerProc(args.server, d)
        c = Client(s.port)
        check(c.call("GET", "crash199") == b"199", "acked write survives kill -9")
        check(len(c.call("KEYS", "crash*")) == 200, "all 200 acked writes survive kill -9")
        c.close()
        s.stop()

        # 4. Torn tail (simulated crash mid-record) is truncated on load.
        with open(s.aof, "ab") as f:
            f.write(b"*3\r\n$3\r\nSET\r\n$4\r\ntorn")
        s = ServerProc(args.server, d)
        check(any("truncated" in l for l in s.log), "startup reports truncation")
        c = Client(s.port)
        check(c.call("GET", "persist-me") == b"yes", "data intact after torn-tail recovery")
        check(c.call("SET", "after", "1") == "OK", "writes work after recovery")
        c.close()
        s.stop()
        s = ServerProc(args.server, d)
        c = Client(s.port)
        check(c.call("GET", "after") == b"1", "post-recovery append replays cleanly")
        c.close()
        s.stop()

    print("client_test.py: %d checks passed" % CHECKS)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
    finally:
        # Never leave a server behind: an orphan holding our stdout would make
        # ctest wait for its full timeout.
        for p in ALL_PROCS:
            if p.poll() is None:
                p.kill()
                p.wait()
