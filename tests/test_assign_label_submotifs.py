"""Composite label assignment: max_submotifs = k and the greedy max_submotifs = -1.

Planted test: reference motifs are written into otherwise-empty motifs at known
positions, so the labels every round should find are known exactly.

Run with pytest, or directly: python tests/test_assign_label_submotifs.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# test this checkout, not whichever MotifCompendium is installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MotifCompendium  # noqa: E402

# the module itself (the package re-exports the class under the same name)
mc_module = sys.modules["MotifCompendium.MotifCompendium"]

NAMES = ["CTCF", "GATA", "SP", "NFKB"]
W, L = 10, 80


def _references(seed=3):
    """Four distinct, high-information 10-bp motifs, shape (4, W, 4)."""
    rg = np.random.default_rng(seed)
    refs = np.full((len(NAMES), W, 4), 0.02)
    for m in range(len(NAMES)):
        refs[m, np.arange(W), rg.integers(0, 4, size=W)] = 0.94
    return refs


def _compendium(refs, layouts):
    """One motif per layout: a list of (reference index, start) sites."""
    stack = np.zeros((len(layouts), L, 4))
    for n, sites in enumerate(layouts):
        for m, start in sites:
            stack[n, start : start + W] += refs[m]
    meta = pd.DataFrame({"name": [f"motif{n}" for n in range(len(layouts))]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return MotifCompendium.build(stack, metadata=meta)


def _label(mc, refs, max_submotifs, min_score=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mc.assign_label_from_motifs(
            reference_motifs=refs, labels=NAMES, min_score=min_score,
            max_submotifs=max_submotifs, save_images=False,
        )
    rounds = sorted(int(c[len("match_name"):]) for c in mc.metadata.columns
                    if c.startswith("match_name"))
    names = [[mc.metadata[f"match_name{i}"].iloc[n] for i in rounds] for n in range(len(mc))]
    scores = np.array([[mc.metadata[f"match_score{i}"].iloc[n] for i in rounds]
                       for n in range(len(mc))])
    return rounds, names, scores


LAYOUTS = [
    [(0, 5)],                         # one site
    [(1, 5), (2, 40)],                # two sites
    [(0, 3), (1, 30), (3, 60)],       # three sites
]


def test_this_checkout_is_imported():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(MotifCompendium.__file__).startswith(here)


def test_greedy_finds_every_site_and_stops():
    refs = _references()
    rounds, names, scores = _label(_compendium(refs, LAYOUTS), refs, -1)
    # as many rounds as the most composite motif has sites, no empty extra round
    assert rounds == [0, 1, 2]
    found = [{x for x in row if x} for row in names]
    assert found == [{"CTCF"}, {"GATA", "SP"}, {"CTCF", "GATA", "NFKB"}]
    # each motif's matches come first, then empty labels and zero scores
    for row, sc in zip(names, scores):
        k = sum(bool(x) for x in row)
        assert all(row[:k]) and not any(row[k:])
        assert np.all(sc[:k] > 0) and np.all(sc[k:] == 0)


def test_greedy_agrees_with_enough_fixed_rounds():
    refs = _references()
    _, g_names, g_scores = _label(_compendium(refs, LAYOUTS), refs, -1)
    rounds, f_names, f_scores = _label(_compendium(refs, LAYOUTS), refs, 3)
    assert rounds == [0, 1, 2]
    assert g_names == f_names
    np.testing.assert_allclose(g_scores, f_scores, rtol=1e-6, atol=1e-7)


def test_fixed_rounds_keep_their_columns_even_when_empty():
    refs = _references()
    rounds, names, _ = _label(_compendium(refs, LAYOUTS), refs, 5)
    assert rounds == [0, 1, 2, 3, 4]
    assert not any(row[3] or row[4] for row in names)


def test_greedy_with_no_match_still_writes_round_zero():
    refs = _references()
    rounds, names, scores = _label(_compendium(refs, LAYOUTS), refs, -1, min_score=1.5)
    assert rounds == [0]
    assert not any(row[0] for row in names) and np.all(scores == 0)


def test_greedy_cap_warns(monkeypatch=None):
    refs = _references()
    old = mc_module.GREEDY_MAX_SUBMOTIFS
    mc_module.GREEDY_MAX_SUBMOTIFS = 2
    try:
        mc = _compendium(refs, LAYOUTS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mc.assign_label_from_motifs(reference_motifs=refs, labels=NAMES, min_score=0.5,
                                        max_submotifs=-1, save_images=False)
        assert any("GREEDY_MAX_SUBMOTIFS" in str(w.message) for w in caught)
        assert "match_name1" in mc.metadata.columns and "match_name2" not in mc.metadata.columns
    finally:
        mc_module.GREEDY_MAX_SUBMOTIFS = old


def test_invalid_max_submotifs():
    refs = _references()
    mc = _compendium(refs, LAYOUTS)
    for bad in (0, -2, 1.5):
        try:
            mc.assign_label_from_motifs(reference_motifs=refs, labels=NAMES, min_score=0.5,
                                        max_submotifs=bad, save_images=False)
        except ValueError:
            continue
        raise AssertionError(f"max_submotifs={bad!r} was accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
