"""Tests for the fail-fast llama.cpp architecture check (2026-08
multi-build-coexistence field fix): ``binary_supports_arch`` and
``resolve_source_gguf_arch`` in magicquant/utils/llamacpp.py.

Ground truth for the fix: GGUF architecture names are string literals baked
into libllama.so (or a statically-linked binary) -- ``strings <lib> | grep
-c '<arch>'`` is the established probe; ``binary_supports_arch`` reimplements
that as a pure-Python chunked byte scan (no ``strings`` subprocess, no model
load). These tests exercise the scan directly with small synthetic files --
no real llama.cpp binary needed.
"""
import struct
from pathlib import Path

from magicquant.utils.llamacpp import (
    LlamaBinaryArchError,
    binary_supports_arch,
    resolve_source_gguf_arch,
)

ARCH = "muse-glimmer"


def _write_gguf_stub(path, *, arch=None, extra_kv=None):
    """Minimal on-disk GGUF: magic + version + 0 tensors + optional
    general.architecture (and any extra_kv) STRING metadata -- exactly what
    GGUFReader.open() needs to parse a real header, without depending on
    the optional `gguf` package."""
    def _string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kvs = []
    if arch is not None:
        kvs.append(("general.architecture", arch))
    for k, v in (extra_kv or {}).items():
        kvs.append((k, v))

    buf = bytearray()
    buf += struct.pack("<I", 0x46554747)  # "GGUF" magic
    buf += struct.pack("<I", 3)           # version
    buf += struct.pack("<Q", 0)           # tensor_count
    buf += struct.pack("<Q", len(kvs))    # metadata_key_count
    for k, v in kvs:
        buf += _string(k)
        buf += struct.pack("<I", 8)  # GGUF STRING type
        buf += _string(v)
    Path(path).write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# binary_supports_arch: True / False / None
# ---------------------------------------------------------------------------


def test_binary_supports_arch_true_when_literal_present(tmp_path):
    binp = tmp_path / "llama-perplexity"
    binp.write_bytes(b"\x00\x01garbage" + ARCH.encode() + b"trailing\x00" * 10)
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_false_when_literal_absent(tmp_path):
    binp = tmp_path / "llama-perplexity"
    binp.write_bytes(b"\x00\x01garbage-no-match-here\x00" * 100)
    assert binary_supports_arch(str(binp), ARCH) is False


def test_binary_supports_arch_none_when_nothing_to_scan(tmp_path):
    # Neither the binary itself nor its directory exist.
    missing = tmp_path / "nowhere" / "llama-perplexity"
    assert binary_supports_arch(str(missing), ARCH) is None


def test_binary_supports_arch_none_when_dir_exists_but_binary_does_not(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    assert binary_supports_arch(str(bindir / "llama-perplexity"), ARCH) is None


def test_binary_supports_arch_prefers_sibling_libllama_over_binary(tmp_path):
    """A shared-library build carries the arch table in libllama.so*, not
    the thin CLI binary -- the literal is present ONLY in the sibling
    library here, so a True verdict proves the sibling is actually
    consulted (not just the binary)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"thin CLI stub, no arch table here" * 10)
    libp = bindir / "libllama.so.1"
    libp.write_bytes(b"padding" + ARCH.encode() + b"padding")
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_scans_binary_when_no_sibling_lib(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"padding" + ARCH.encode() + b"padding")
    assert binary_supports_arch(str(binp), ARCH) is True


# ---------------------------------------------------------------------------
# Opus review (BLOCKING): all three real llama-perplexity binaries checked
# are dynamically linked with ZERO arch literals in the binary itself (the
# arch table lives entirely in libllama.so) -- a same-directory-only sibling
# search + "fall back to scanning the binary" therefore returned a
# guaranteed-wrong False (not even None) for any layout the sibling glob
# missed, reproduced synthetically below with a standard install-prefix
# layout (bin/llama-perplexity + ../lib/libllama.so). Fixed two ways: (i)
# the sibling search now also covers ../lib, ../lib64, and Windows names;
# (ii) a binary-only scan that comes back empty is trusted as a real
# negative ONLY when the binary does NOT look dynamically linked (no
# b"libllama" DT_NEEDED marker) -- otherwise it's undeterminable (None).
# Library and binary scans are OR'd together (a True from either wins),
# which additionally -- verified by the reviewer to introduce no false
# True -- resolves a stale sibling library sitting beside a static binary
# that itself carries the current literal.
# ---------------------------------------------------------------------------


def test_binary_supports_arch_install_prefix_layout(tmp_path):
    """<prefix>/bin/llama-perplexity + <prefix>/lib/libllama.so -- the
    exact layout the reviewer reproduced the pre-fix False against. The
    binary itself is dynamically linked (carries the DT_NEEDED marker) and
    has no arch literals of its own, matching every real binary checked."""
    prefix = tmp_path / "prefix"
    bindir = prefix / "bin"
    libdir = prefix / "lib"
    bindir.mkdir(parents=True)
    libdir.mkdir(parents=True)
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"ELF...dynamic binary, needs libllama.so..." * 20)
    libp = libdir / "libllama.so"
    libp.write_bytes(b"padding" + ARCH.encode() + b"padding" * 50)
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_install_prefix_lib64_layout(tmp_path):
    prefix = tmp_path / "prefix"
    bindir = prefix / "bin"
    libdir = prefix / "lib64"
    bindir.mkdir(parents=True)
    libdir.mkdir(parents=True)
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"ELF...dynamic binary, needs libllama.so..." * 20)
    libp = libdir / "libllama.so.1"
    libp.write_bytes(b"padding" + ARCH.encode() + b"padding" * 50)
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_dynamic_binary_no_reachable_lib_returns_none(tmp_path):
    """A dynamically-linked binary (carries the b"libllama" DT_NEEDED
    marker) with no reachable library anywhere (not same-dir, not
    ../lib(64)) must return None -- undeterminable -- NEVER the
    guaranteed-wrong False the pre-fix code returned here."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(
        b"ELF header stuff... needs libllama.so.1 ... more binary bytes" * 20
    )
    assert binary_supports_arch(str(binp), ARCH) is None


def test_binary_supports_arch_static_build_no_sibling_returns_true(tmp_path):
    """A true static build carries the arch literals in the binary itself
    and has no DT_NEEDED marker for libllama at all -- a real True, not
    just "undeterminable but happens to look True"."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"statically linked, no dynamic deps" + ARCH.encode())
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_static_build_no_sibling_returns_false_for_other_arch(tmp_path):
    """Sibling to the test above: the same static-build shape, but scanned
    for an arch the binary genuinely doesn't have -- must be a real False
    (the binary is its own arch table and doesn't look dynamically linked),
    not None."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"statically linked, no dynamic deps, only llama arch")
    assert binary_supports_arch(str(binp), ARCH) is False


def test_binary_supports_arch_stale_lib_beside_static_binary_true_via_or(tmp_path):
    """A stale/mismatched sibling library that does NOT contain the current
    arch, sitting beside a STATIC binary that DOES -- the library-only scan
    alone would say False, but OR'ing in the binary scan correctly flips it
    to True (verified by the reviewer not to introduce any false True)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binp = bindir / "llama-perplexity"
    binp.write_bytes(b"statically linked binary, carries " + ARCH.encode())
    stale_libp = bindir / "libllama.so.old"
    stale_libp.write_bytes(b"an old library with a totally different arch table")
    assert binary_supports_arch(str(binp), ARCH) is True


def test_binary_supports_arch_chunk_boundary_straddle(tmp_path):
    """The literal is deliberately positioned to span multiple chunk
    boundaries under a tiny chunk_size -- without carrying the tail of the
    previous chunk forward, a naive per-chunk membership scan would miss
    it entirely."""
    binp = tmp_path / "llama-perplexity"
    literal = ARCH.encode()
    chunk_size = 4
    # "XX" + 12-byte literal + "YY" = 16 bytes; a 4-byte chunking puts the
    # literal's start inside chunk 0 and its end inside chunk 3 -- it
    # crosses THREE chunk boundaries (at offsets 4, 8, 12).
    content = b"XX" + literal + b"YY"
    assert len(content) > chunk_size * 3
    binp.write_bytes(content)
    assert binary_supports_arch(str(binp), ARCH, chunk_size=chunk_size) is True


def test_binary_supports_arch_chunk_boundary_straddle_false_when_absent(tmp_path):
    """Sibling to the straddle test above: tiny chunking must not manufacture
    a false positive when the literal genuinely isn't present."""
    binp = tmp_path / "llama-perplexity"
    binp.write_bytes(b"XX" + b"totally-different-text" + b"YY")
    assert binary_supports_arch(str(binp), ARCH, chunk_size=4) is False


def test_binary_supports_arch_default_chunk_size_still_finds_straddling_literal(tmp_path):
    """Same straddle shape as above but at the REAL default chunk size (4
    MiB) -- pads the file so the literal sits exactly across the boundary
    between the first and second chunk."""
    from magicquant.utils.llamacpp import _ARCH_SCAN_CHUNK_BYTES

    binp = tmp_path / "llama-perplexity"
    literal = ARCH.encode()
    # Start the literal 4 bytes before the chunk boundary so it straddles.
    prefix = b"\x00" * (_ARCH_SCAN_CHUNK_BYTES - 4)
    binp.write_bytes(prefix + literal + b"\x00" * 16)
    assert binary_supports_arch(str(binp), ARCH) is True


# ---------------------------------------------------------------------------
# resolve_source_gguf_arch
# ---------------------------------------------------------------------------


def test_resolve_source_gguf_arch_reads_general_architecture(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf_stub(p, arch=ARCH)
    assert resolve_source_gguf_arch(str(p)) == ARCH


def test_resolve_source_gguf_arch_none_when_no_arch_key(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf_stub(p, arch=None)  # valid GGUF, no general.architecture
    assert resolve_source_gguf_arch(str(p)) is None


def test_resolve_source_gguf_arch_none_for_nonexistent_path(tmp_path):
    assert resolve_source_gguf_arch(str(tmp_path / "nonexistent.gguf")) is None


def test_resolve_source_gguf_arch_none_for_non_gguf_bytes(tmp_path):
    p = tmp_path / "not-a-gguf.gguf"
    p.write_bytes(b"0" * 1024)  # bad magic
    assert resolve_source_gguf_arch(str(p)) is None


def test_resolve_source_gguf_arch_none_for_truncated_gguf(tmp_path):
    p = tmp_path / "truncated.gguf"
    p.write_bytes(b"GGUF")  # magic only, nothing else -- struct.error on read
    assert resolve_source_gguf_arch(str(p)) is None


def test_resolve_source_gguf_arch_none_for_directory(tmp_path):
    d = tmp_path / "safetensors_dir"
    d.mkdir()
    assert resolve_source_gguf_arch(str(d)) is None


# ---------------------------------------------------------------------------
# LlamaBinaryArchError
# ---------------------------------------------------------------------------


def test_llama_binary_arch_error_is_a_runtime_error():
    assert issubclass(LlamaBinaryArchError, RuntimeError)
