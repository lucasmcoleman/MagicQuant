"""
ctypes binding to libggml — encoder dispatch for all quantized schemes.

Discovery order (first match wins):
    1. MAGICQUANT_LIBGGML_DIR env var (explicit override)
    2. Common system llama.cpp build paths
    3. llama-cpp-python's bundled libs (always available since it's a hard dep)

Public API:
    ggml_encode(weights, ggml_type, imatrix=None) -> bytes
    GGML_TYPE_IDS  (mapping name -> numeric ggml type enum, synced from ggml.h)
    LibggmlNotFound  (exception)
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ggml_type enum values from ggml/include/ggml.h.
# Synced as of llama.cpp build at /home/lucas/llama.cpp-build/.
# A startup sanity check (_verify_type_ids) catches drift if the loaded
# libggml renumbers types.
GGML_TYPE_IDS = {
    "F32":     0,
    "F16":     1,
    "Q4_0":    2,
    "Q4_1":    3,
    "Q5_0":    6,
    "Q5_1":    7,
    "Q8_0":    8,
    "Q8_1":    9,
    "Q2_K":    10,
    "Q3_K":    11,
    "Q4_K":    12,
    "Q5_K":    13,
    "Q6_K":    14,
    "Q8_K":    15,
    "IQ2_XXS": 16,
    "IQ2_XS":  17,
    "IQ3_XXS": 18,
    "IQ1_S":   19,
    "IQ4_NL":  20,
    "IQ3_S":   21,
    "IQ2_S":   22,
    "IQ4_XS":  23,
    "IQ1_M":   29,
    "BF16":    30,
    "MXFP4":   39,
}


_SYSTEM_SEARCH_DIRS = [
    "~/llama.cpp/build/bin",
    "~/llama.cpp-build/build/bin",
    "/usr/local/lib",
    "/home/linuxbrew/.linuxbrew/lib",
]


class LibggmlNotFound(RuntimeError):
    """Raised when libggml-base.so / libggml-cpu.so cannot be located."""


def _discover_libggml() -> Tuple[Path, Path]:
    """Return (libggml-base path, libggml-cpu path).

    Discovery order:
      1. MAGICQUANT_LIBGGML_DIR env var (explicit override)
      2. Common system llama.cpp build paths
      3. llama-cpp-python's bundled libs

    Raises:
      LibggmlNotFound if neither library can be found.
    """
    # 1. Explicit env var
    env_dir = os.environ.get("MAGICQUANT_LIBGGML_DIR")
    if env_dir:
        result = _check_dir(Path(env_dir).expanduser())
        if result:
            return result

    # 2. System paths
    for raw in _SYSTEM_SEARCH_DIRS:
        result = _check_dir(Path(raw).expanduser())
        if result:
            return result

    # Optional environment hint
    llamacpp_path = os.environ.get("LLAMACPP_PATH")
    if llamacpp_path:
        result = _check_dir(Path(llamacpp_path).expanduser() / "build" / "bin")
        if result:
            return result

    # 3. llama-cpp-python bundled
    try:
        import llama_cpp  # type: ignore
    except ImportError:
        raise LibggmlNotFound(
            "Could not find libggml-base.so / libggml-cpu.so. "
            "Tried: $MAGICQUANT_LIBGGML_DIR, common system paths, "
            "$LLAMACPP_PATH/build/bin, and llama-cpp-python bundle. "
            "Install llama-cpp-python or set MAGICQUANT_LIBGGML_DIR to a "
            "directory containing both libraries."
        )

    bundle_dir = Path(llama_cpp.__file__).parent / "lib"
    result = _check_dir(bundle_dir)
    if result:
        return result

    raise LibggmlNotFound(
        f"llama-cpp-python is installed but bundled libggml not found at {bundle_dir}. "
        f"Try: pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir"
    )


def _check_dir(d: Path) -> Optional[Tuple[Path, Path]]:
    """Return (base, cpu) library paths if both exist in the directory, else None.

    Tries a few common naming variants (versioned and unversioned).
    """
    if not d.is_dir():
        return None

    base_names = ["libggml-base.so", "libggml-base.so.0", "libggml-base.dylib"]
    cpu_names = ["libggml-cpu.so", "libggml-cpu.so.0", "libggml-cpu.dylib"]

    base = next((d / n for n in base_names if (d / n).exists()), None)
    cpu = next((d / n for n in cpu_names if (d / n).exists()), None)

    # Fallback: glob for versioned variants like libggml-base.so.0.9.7
    if base is None:
        candidates = sorted(d.glob("libggml-base.so*"))
        if candidates:
            base = candidates[0]
    if cpu is None:
        candidates = sorted(d.glob("libggml-cpu.so*"))
        if candidates:
            cpu = candidates[0]

    if base and cpu:
        return base, cpu
    return None
