"""v2 budget search's --enable-iq wiring (F9, 2026-08 cleanup).

BudgetInfeasibleError (v2/outcome.py) has always told users to raise
--budget-gb "or enable more aggressive schemes (e.g. --enable-iq)" -- but
until this change V2Config had no enable_iq field at all, so the advice was
dead: the flag parsed on the CLI and was silently discarded by _run_v2_search.
These tests pin the fix at the unit level (_select_schemes / V2Config),
independent of the full run_budget_search characterization suite in
tests/test_v2_search_characterization.py.

Per the verifier notes on findings v2-budget-search/4 and /5:
  * default (enable_iq=False) must be BYTE-IDENTICAL to today's choice set.
  * v2 must NOT reuse v1's unconditional LEGACY_Q4_SCHEME_NAMES drop -- the
    q4nx target profile's Q4_0/Q4_1 must survive regardless of enable_iq.
  * the IQ addition must still respect requires_imatrix per scheme (IQ2_XS/
    IQ2_XXS need an imatrix); v2 additionally excludes the sub-2-bit members
    (IQ1_M, IQ1_S) per docs/redesign.md §9 Non-goals ("Sub-2-bit IQ types").
"""
from magicquant.v2.search import (
    DEFAULT_SCHEMES,
    Q4NX_PROFILE_SCHEMES,
    V2Config,
    _select_schemes,
)

# The six IQ_SCHEME_NAMES members v2 is willing to add (>= 2 bpw); the two
# sub-2-bit members (IQ1_M, IQ1_S) are deliberately never added by v2.
_V2_IQ_ADDITIONS = {"IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS"}
_NO_IMATRIX_IQ_ADDITIONS = {"IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S"}  # requires_imatrix=False
_IMATRIX_ONLY_IQ_ADDITIONS = {"IQ2_XS", "IQ2_XXS"}  # requires_imatrix=True
_SUB_2BIT_IQ = {"IQ1_M", "IQ1_S"}


def _cfg(**overrides):
    kwargs = dict(
        source_model_path="src.gguf",
        output_dir="/tmp/does-not-matter",
        budget_gb=8.0,
    )
    kwargs.update(overrides)
    return V2Config(**kwargs)


def test_enable_iq_defaults_to_false():
    assert _cfg().enable_iq is False


def test_enable_iq_false_choice_set_unchanged_imatrix_active():
    cfg = _cfg(enable_iq=False)
    assert _select_schemes(cfg, imatrix_active=True) == list(DEFAULT_SCHEMES)


def test_enable_iq_false_choice_set_unchanged_no_imatrix():
    """IQ4_NL has requires_imatrix=False (it's IMATRIX_DEPENDENT, not
    requires_imatrix), so the no-imatrix default set is unchanged too."""
    cfg = _cfg(enable_iq=False)
    assert _select_schemes(cfg, imatrix_active=False) == list(DEFAULT_SCHEMES)


def test_enable_iq_true_with_imatrix_adds_all_non_sub_2bit_iq_schemes():
    cfg = _cfg(enable_iq=True)
    kept = _select_schemes(cfg, imatrix_active=True)

    assert set(DEFAULT_SCHEMES).issubset(kept)
    added = set(kept) - set(DEFAULT_SCHEMES)
    assert added == _V2_IQ_ADDITIONS
    assert added.isdisjoint(_SUB_2BIT_IQ)


def test_enable_iq_true_without_imatrix_adds_only_no_imatrix_iq_schemes():
    cfg = _cfg(enable_iq=True)
    kept = _select_schemes(cfg, imatrix_active=False)

    assert set(DEFAULT_SCHEMES).issubset(kept)
    added = set(kept) - set(DEFAULT_SCHEMES)
    assert added == _NO_IMATRIX_IQ_ADDITIONS
    # requires_imatrix members and the sub-2-bit family are both absent.
    assert added.isdisjoint(_IMATRIX_ONLY_IQ_ADDITIONS)
    assert added.isdisjoint(_SUB_2BIT_IQ)


def test_q4nx_profile_unaffected_by_enable_iq_with_imatrix():
    cfg = _cfg(target_profile="q4nx", enable_iq=True)
    kept = _select_schemes(cfg, imatrix_active=True)
    assert kept == list(Q4NX_PROFILE_SCHEMES)
    assert "Q4_0" in kept and "Q4_1" in kept  # the load-bearing NPU-profile check


def test_q4nx_profile_unaffected_by_enable_iq_without_imatrix():
    cfg = _cfg(target_profile="q4nx", enable_iq=True)
    kept = _select_schemes(cfg, imatrix_active=False)
    assert kept == list(Q4NX_PROFILE_SCHEMES)
    assert "Q4_0" in kept and "Q4_1" in kept


def test_explicit_schemes_bypass_enable_iq_addition():
    """cfg.schemes is a user-authority override (search.py:76-77's existing
    contract) -- enable_iq must not silently inject extra names into it."""
    cfg = _cfg(schemes=["BF16", "Q4_K_M"], enable_iq=True)
    kept = _select_schemes(cfg, imatrix_active=True)
    assert kept == ["BF16", "Q4_K_M"]
