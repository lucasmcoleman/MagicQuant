"""Block-size / type-size table consistency tests.

The converters' GGML_BLOCK_SIZE / GGML_TYPE_SIZE tables must agree with the
authoritative tables in ggml_binding (the single source of truth). Locks the
IQ4_XS fix: converters previously had block=32/size=18 (IQ4_NL's values) which
disagreed with ggml_binding's correct block=256/size=136, a corruption trap.
"""
from magicquant.quant import converters
from magicquant.quant import ggml_binding


def test_iq4_xs_matches_binding():
    assert converters.GGML_BLOCK_SIZE["IQ4_XS"] == 256
    assert converters.GGML_TYPE_SIZE["IQ4_XS"] == 136
    assert ggml_binding._GGML_BLOCK_SIZE["IQ4_XS"] == 256
    assert ggml_binding._GGML_TYPE_SIZE["IQ4_XS"] == 136


def test_converters_tables_consistent_with_binding():
    """Every type the binding knows must have identical block/type size in
    the converters' tables (no drift)."""
    for name, block in ggml_binding._GGML_BLOCK_SIZE.items():
        assert converters.GGML_BLOCK_SIZE[name] == block, (
            f"{name}: converters block_size {converters.GGML_BLOCK_SIZE.get(name)} "
            f"!= binding {block}"
        )
    for name, size in ggml_binding._GGML_TYPE_SIZE.items():
        assert converters.GGML_TYPE_SIZE[name] == size, (
            f"{name}: converters type_size {converters.GGML_TYPE_SIZE.get(name)} "
            f"!= binding {size}"
        )


def test_iq4_xs_data_size_correct():
    """ggml_tensor_data_size for IQ4_XS uses block=256/size=136."""
    # 256 elements = 1 block = 136 bytes
    assert converters.ggml_tensor_data_size("IQ4_XS", 256) == 136
    # 512 elements = 2 blocks = 272 bytes
    assert converters.ggml_tensor_data_size("IQ4_XS", 512) == 272
