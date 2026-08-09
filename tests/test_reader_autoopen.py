"""Regression: GGUFReader must not silently return empty when not explicitly opened.

The parse lives in open(), but __init__ doesn't call it and the public accessors
didn't guard for it — so `GGUFReader(path).get_all_tensors_info()` returned [] and
`read_gguf_file(path)` (whose docstring promises an *opened* reader) returned an
empty one. Accessors now lazily open; open() is idempotent so the `with` /
explicit-open paths still work and never double-parse.
"""

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from magicquant.gguf.reader import GGUFReader, read_gguf_file


@pytest.fixture
def tiny_gguf(tmp_path):
    p = tmp_path / "tiny.gguf"
    w = gguf.GGUFWriter(str(p), arch="llama")
    w.add_uint32("llama.block_count", 1)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((8, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(p)


def test_accessors_lazy_open(tiny_gguf):
    r = GGUFReader(tiny_gguf)  # no .open(), no `with`
    assert len(r.get_all_tensors_info()) == 2
    assert "token_embd.weight" in r.get_tensor_names()
    assert r.get_metadata().get("llama.block_count") == 1


def test_read_gguf_file_returns_opened(tiny_gguf):
    r = read_gguf_file(tiny_gguf)
    assert len(r.get_all_tensors_info()) == 2


def test_open_is_idempotent(tiny_gguf):
    r = GGUFReader(tiny_gguf)
    r.open()
    n1 = len(r.tensors)
    r.open()  # must not append duplicates / re-parse
    assert len(r.tensors) == n1 == 2


def test_context_manager_still_works(tiny_gguf):
    with GGUFReader(tiny_gguf) as r:
        assert len(r.get_all_tensors_info()) == 2
