import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentenv.adapters import APIAdapter, make_adapter


class Stub(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body))
        out = {"choices": [{"message": {"content": '{"tool": "submit", "args": {"answer": 1}}'},
                            "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_api_adapter_against_local_stub():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        a = make_adapter(f"api:test-model@http://127.0.0.1:{srv.server_port}/v1")
        g = a.generate([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0, seed=3)
        assert g.prompt_tokens == 11 and g.completion_tokens == 7 and "submit" in g.text
        path, body = Stub.seen[-1]
        assert path == "/v1/chat/completions" and body["seed"] == 3 and body["model"] == "test-model"
    finally:
        srv.shutdown()


def test_remote_calls_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENTENV_ALLOW_API", raising=False)
    a = APIAdapter("m", base_url="https://api.example.com/v1")
    with pytest.raises(RuntimeError, match="disabled"):
        a.generate([], max_tokens=1, temperature=0, seed=0)
    with pytest.raises(RuntimeError):
        APIAdapter("m").generate([], max_tokens=1, temperature=0, seed=0)
