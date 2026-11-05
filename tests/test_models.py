from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pickframe import models

PAYLOAD = b"fake onnx model " * 4096
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:
        if self.path == "/model.onnx":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.send_header("X-Linked-Etag", f'"{PAYLOAD_SHA}"')
            self.end_headers()
            self.wfile.write(PAYLOAD)
        elif self.path == "/wrong-etag.onnx":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.send_header("X-Linked-Etag", '"' + "0" * 64 + '"')
            self.end_headers()
            self.wfile.write(PAYLOAD)
        elif self.path == "/truncated.onnx":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD) * 2))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch) -> None:
    """The test server is local; never send its requests through a proxy."""
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")


@pytest.fixture
def local_clip(monkeypatch, server: str):
    def use(path: str, sha: str | None = None) -> None:
        model = replace(models.CLIP, url=server + path, sha256=sha)
        monkeypatch.setitem(models.REMOTE_MODELS, "clip", model)

    return use


def test_bundled_models_exist():
    for name in (models.FACE_DETECTOR, models.FACE_LANDMARKS, models.FACE_BLENDSHAPES):
        path = models.bundled(name)
        assert path.is_file()
        assert path.stat().st_size > 100_000
    with pytest.raises(models.ModelError):
        models.bundled("nope.onnx")


def test_models_dir_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PICKFRAME_MODELS_DIR", str(tmp_path / "m"))
    assert models.models_dir() == tmp_path / "m"
    assert models.remote_path("clip").parent == tmp_path / "m"
    monkeypatch.delenv("PICKFRAME_MODELS_DIR")
    assert models.models_dir().name == "models"


def test_download_success_with_linked_etag(local_clip):
    local_clip("/model.onnx")
    seen: list[tuple[int, int | None]] = []
    path = models.download("clip", progress=lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == PAYLOAD
    assert models.is_installed("clip")
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert not path.with_suffix(path.suffix + ".part").exists()
    # already installed: no new download
    assert models.download("clip") == path
    assert models.remove("clip") is True
    assert models.remove("clip") is False


def test_download_pinned_checksum(local_clip):
    local_clip("/model.onnx", sha=PAYLOAD_SHA)
    assert models.download("clip").is_file()
    local_clip("/model.onnx", sha="f" * 64)
    with pytest.raises(models.ModelError, match="checksum mismatch"):
        models.download("clip", force=True)
    assert models.remote_path("clip").read_bytes() == PAYLOAD  # the old file survives


def test_download_ignores_etag_headers(local_clip):
    # Only the pinned SHA-256 is trusted: CDN ETags (e.g. Xet hashes) are not file checksums.
    local_clip("/wrong-etag.onnx")
    assert models.download("clip").read_bytes() == PAYLOAD


def test_download_truncated(local_clip):
    local_clip("/truncated.onnx")
    with pytest.raises(models.ModelError):
        models.download("clip")
    assert not models.is_installed("clip")
    assert not list(models.models_dir().glob("*.part"))


def test_download_http_error(local_clip):
    local_clip("/missing.onnx")
    with pytest.raises(models.ModelError, match="could not download"):
        models.download("clip")


def test_download_cancelled(local_clip):
    local_clip("/model.onnx")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(models.ModelError, match="cancelled"):
        models.download("clip", cancel=cancel)


def test_url_override(monkeypatch, server: str):
    monkeypatch.setenv("PICKFRAME_CLIP_URL", server + "/model.onnx")
    monkeypatch.setitem(
        models.REMOTE_MODELS, "clip", replace(models.CLIP, url="http://invalid.invalid/x")
    )
    assert models.download("clip").read_bytes() == PAYLOAD


def test_session_and_providers(monkeypatch):
    models.onnx_providers.cache_clear()
    assert models.onnx_providers() == ("CPUExecutionProvider",)
    models.onnx_providers.cache_clear()
    monkeypatch.setenv("PICKFRAME_ORT_PROVIDERS", "NotARealProvider")
    assert models.onnx_providers() == ("CPUExecutionProvider",)
    models.onnx_providers.cache_clear()
    monkeypatch.setenv("PICKFRAME_ORT_PROVIDERS", "CPUExecutionProvider")
    assert models.onnx_providers() == ("CPUExecutionProvider",)
    models.onnx_providers.cache_clear()
    sess = models.session(models.bundled(models.FACE_BLENDSHAPES), threads=1)
    assert sess.get_inputs()[0].name == "points"
