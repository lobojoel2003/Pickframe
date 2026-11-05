"""Locate bundled ONNX models and download optional ones.

The face models (about 3.6 MB) ship inside the package, so Pickframe works
offline out of the box. The CLIP image encoder used for semantic grouping is
larger (about 88 MB) and is downloaded once, on request, into the user data
folder.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir

from . import __version__

BUNDLED_DIR = Path(__file__).resolve().parent / "models"

#: Models shipped inside the package.
FACE_DETECTOR = "face_detection_yunet.onnx"
FACE_LANDMARKS = "face_landmarks.onnx"
FACE_BLENDSHAPES = "face_blendshapes.onnx"

ProgressFn = Callable[[int, "int | None"], None]


@dataclass(frozen=True)
class RemoteModel:
    name: str
    filename: str
    url: str
    sha256: str | None
    approx_mb: int
    description: str
    license: str


CLIP = RemoteModel(
    name="clip",
    filename="clip-vit-base-patch32-vision-quantized.onnx",
    url="https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
    sha256=None,
    approx_mb=88,
    description="CLIP ViT-B/32 image encoder (OpenAI), 8-bit ONNX export by Xenova",
    license="MIT",
)

REMOTE_MODELS: dict[str, RemoteModel] = {CLIP.name: CLIP}


class ModelError(RuntimeError):
    """Raised when a model is missing or a download fails."""


def bundled(filename: str) -> Path:
    path = BUNDLED_DIR / filename
    if not path.is_file():
        raise ModelError(f"bundled model {filename} is missing from {BUNDLED_DIR}")
    return path


def models_dir() -> Path:
    override = os.environ.get("PICKFRAME_MODELS_DIR")
    base = (
        Path(override).expanduser()
        if override
        else Path(user_data_dir("pickframe", False)) / "models"
    )
    return base


def remote_path(name: str) -> Path:
    return models_dir() / REMOTE_MODELS[name].filename


def is_installed(name: str) -> bool:
    return remote_path(name).is_file()


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:  # frozen apps and some macOS Pythons ship without system CA certificates
        import certifi

        context.load_verify_locations(certifi.where())
    except Exception:
        pass
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_download_lock = threading.Lock()


def download(
    name: str,
    progress: ProgressFn | None = None,
    force: bool = False,
    cancel: threading.Event | None = None,
) -> Path:
    """Download a remote model, verify its checksum and return its path."""
    model = REMOTE_MODELS[name]
    target = remote_path(name)
    with _download_lock:
        if target.is_file() and not force:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = os.environ.get(f"PICKFRAME_{name.upper()}_URL", model.url)
        request = urllib.request.Request(url, headers={"User-Agent": f"pickframe/{__version__}"})
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else None
                expected = model.sha256
                done = 0
                with partial.open("wb") as out:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise ModelError("download cancelled")
                        chunk = response.read(1 << 18)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            if total is not None and done != total:
                raise ModelError(f"download of {name} was truncated ({done} of {total} bytes)")
            actual = digest.hexdigest()
            if expected and actual != expected:
                raise ModelError(f"checksum mismatch for {name}: expected {expected}, got {actual}")
            os.replace(partial, target)
        except ModelError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise ModelError(f"could not download {name} from {url}: {exc}") from exc
        return target


def remove(name: str) -> bool:
    path = remote_path(name)
    if path.is_file():
        path.unlink()
        return True
    return False


@lru_cache(maxsize=1)
def onnx_providers() -> tuple[str, ...]:
    """CPU everywhere; CoreML / DirectML are opt-in via PICKFRAME_ORT_PROVIDERS."""
    import onnxruntime as ort

    wanted = os.environ.get("PICKFRAME_ORT_PROVIDERS")
    available = ort.get_available_providers()
    if wanted:
        chosen = tuple(p for p in wanted.split(",") if p in available)
        if chosen:
            return (
                (*chosen, "CPUExecutionProvider")
                if "CPUExecutionProvider" not in chosen
                else chosen
            )
    return ("CPUExecutionProvider",)


def session(path: Path, threads: int = 0):
    """Create an ONNX Runtime session with quiet logging."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=list(onnx_providers()))
