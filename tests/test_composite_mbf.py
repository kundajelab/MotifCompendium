"""
Regression + feature checks for MotifCompendium.utils.composite.

Run:
    python tests/test_composite_mbf.py ORIGINAL_COMPOSITE [SNAPSHOT_COMPOSITE [PREVIOUS_COMPOSITE [BASE_COMPOSITE]]]

ORIGINAL_COMPOSITE is the pre-generalization composite.py (old API; 4886214). It gates that the
frozen 3 bp core is still the original algorithm, fed the penalties negated.

SNAPSHOT_COMPOSITE is a composite.py from before the exact state DP and the target padding
(one arrangement per cell; 4af9021). Where the scoring did not change (zero penalties, no
rescaling), results are checked bit for bit against it, feeding it the target padded as the
current code pads it, with and without global_exact.

PREVIOUS_COMPOSITE is a composite.py from before penalties were made positive and the gap
penalty affine, and match_penalty_f was removed (5c9d388). It is fed the penalties negated:
without a skip_penalty_l1 (where the two gap rules agree) results are checked bit for bit, and
its match_penalty_f against match_count_penalty.

BASE_COMPOSITE is the composite.py from before general_penalty_root and the fixed-length
composite were removed and overlap_penalty was renamed overlap_penalty_fraction (b831a8d). Run
with general_penalty_root=0, it must match bit for bit.

Penalties are numbers >= 0, subtracted from the score.
"""

import functools
import importlib.util
import itertools
import math
import os
import sys

import numpy as np


# ------------------------------------------------------------------ helpers
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def api(mod, name):
    """The function name (Bellman-Ford) of mod, or its Smith-Waterman name in the snapshots (old API)."""
    if hasattr(mod, name):
        return getattr(mod, name)
    return getattr(mod, name.replace("bellman_ford", "smith_waterman"))


def make_case(rng, F, M, L, noise=0.05):
    """A finger set, and a motif built by planting those fingers with gaps/overlaps."""
    fingers = rng.random((F * M, 4)) ** 2 + 0.01
    motif = rng.random((L, 4)) * noise
    pos = int(rng.integers(0, max(1, L - M)))
    for f in range(F):
        if pos + M > L:
            break
        if rng.random() < 0.8:  # Sometimes leave a finger unplaced
            motif[pos:pos + M] += fingers[f * M:(f + 1) * M] * (0.5 + rng.random())
        pos = pos + max(1, M + int(rng.integers(-2, 3)))  # Overlap / adjacent / gap
    return motif, fingers


def plant_one_hot(rng, F, M, tries=50):
    """F one-hot fingers laid end to end, with every off-frame window distinct."""
    for _ in range(tries):
        idx = rng.integers(0, 4, size=F * M)
        fingers = np.eye(4)[idx]
        windows = [fingers[i:i + M].tobytes() for i in range(F * M - M + 1)]
        planted = {fingers[f * M:(f + 1) * M].tobytes() for f in range(F)}
        off_frame = [w for i, w in enumerate(windows) if i % M != 0]
        if len(planted) == F and not (planted & set(off_frame)):
            return fingers, fingers.copy()
    raise RuntimeError(f"could not plant distinct one-hot fingers for F={F} M={M}")


def pad(motif, M):
    """The target, padded as the current code pads it (M-1 zeros on either side)."""
    return np.pad(motif, ((M - 1, M - 1), (0, 0)))


NEUTRAL = dict(
    sim_threshold=0.0,
    overlap_penalty_f1=0.02,
    overlap_penalty_f2=0.1,
    skip_penalty_f1=0.1,
    skip_penalty_f2=0.1,
    skip_penalty_l1=0.02,
    skip_penalty_l2=0.1,
)
PENALTY_NAMES = ("overlap_penalty_f1", "overlap_penalty_f2", "skip_penalty_f1", "skip_penalty_f2",
                 "skip_penalty_l1", "skip_penalty_l2")
ZERO = dict(NEUTRAL, overlap_penalty_f1=0.0, overlap_penalty_f2=0.0, skip_penalty_f1=0.0,
            skip_penalty_f2=0.0, skip_penalty_l1=0.0, skip_penalty_l2=0.0)

OUTPUTS = ["mbf_score", "sim_score", "aligned_motif_fs", "mbf_orient",
           "mbf_align_f", "mbf_align_l", "mbf_scale_f", "mbf_actions"]


def differing(out_a, out_b, fields=OUTPUTS):
    """Fields that differ between two run_* outputs (bit for bit)."""
    diffs = []
    for name, a, b in zip(OUTPUTS, out_a[:8], out_b[:8]):
        if name not in fields:
            continue
        if isinstance(a, np.ndarray):
            same = a.shape == b.shape and np.array_equal(a, b)
        else:
            same = a == b
        if not same:
            diffs.append(name)
    return diffs


def differing_from_snapshot(out_new, out_snap, L_padded):
    """Fields that differ from the snapshot run on the padded target: bit for bit, except
    sim_score (to 1e-12; the similarity search sees a differently padded frame), and
    aligned_motif_fs on the snapshot's first L_padded columns (its frame is M-1 wider)."""
    diffs = differing(out_new, out_snap, fields=[f for f in OUTPUTS if f not in ("sim_score", "aligned_motif_fs")])
    if abs(out_new[1] - out_snap[1]) > 1e-12:
        diffs.append("sim_score")
    a, b = out_new[2], out_snap[2]
    if a.shape[1] != L_padded or not np.array_equal(a, b[:, :L_padded]) or b[:, L_padded:].any():
        diffs.append("aligned_motif_fs")
    return diffs


def unroll(fingers, M):
    """Concatenated fingers (F*M, D) as a stack (F, M, D), as the wrappers do."""
    return fingers.reshape(-1, M, fingers.shape[-1])


def call_znf(mod, motif, fingers, **kwargs):
    params = dict(NEUTRAL)
    params.update(kwargs)
    return mod.run_modified_smith_waterman_znf(target_motif=motif, reference_motif=fingers, **params)


def call_general(mod, motif, fingers, M, **kwargs):
    params = dict(NEUTRAL)
    params.update(kwargs)
    # Exercise the unexpanded generalized core; public stack expansion has its own tests.
    params.setdefault("return_matrices", False)
    params.setdefault("label_unsigned", True)
    return api(mod, "_run_modified_bellman_ford")(
        target_motif=motif, subunit_motif_stack=unroll(fingers, M), **params)


def call_core(mod, sim, scale, M, **kwargs):
    """The DP core alone, on given similarities (one orientation)."""
    F, N = sim.shape
    params = dict(NEUTRAL, orientations=("for",))
    params.update(kwargs)
    return api(mod, "calculate_modified_bellman_ford")(
        matrix_l=np.zeros((N + M - 1, 4)), matrix_f=np.zeros((F, M, 4)),
        sim_for=sim, sim_rev=sim, scale_factor=scale, **params)


def prepare(mod, motif, fingers, M):
    """The current preprocessing, so two DP cores can be fed byte-identical inputs."""
    import cupy as cp
    motif = np.abs(motif)
    motif_norm = motif / np.linalg.norm(motif, axis=(0, 1), keepdims=True)
    unrolled = mod._unroll_motif(motif_norm, M, np)
    unrolled_cp = mod._unroll_motif(cp.asarray(motif_norm), M, cp)
    importance = (unrolled * unrolled).sum(axis=(1, 2))
    stack = unroll(np.abs(fingers), M)
    norms = np.linalg.norm(stack, axis=(1, 2), keepdims=True)
    stack_norm = stack / np.where(norms == 0, 1, norms)
    stack_cp = cp.asarray(stack_norm)
    sim_for = mod.compute_aligned_similarity(stack_cp, unrolled_cp)
    sim_rev = mod.compute_aligned_similarity(stack_cp, unrolled_cp[:, ::-1, ::-1])
    return stack_norm, motif_norm, sim_for, sim_rev, importance


def chain_score(chain, sim, scale, M, p):
    """Score of a chain of matches (finger, motif position), by the documented rules: every
    penalty affine (a first unit, then each further one), and subtracted."""
    raw = sum(scale[w] * sim[f, w] for f, w in chain)
    penalty = 0.0
    for (fa, wa), (fb, wb) in zip(chain, chain[1:]):
        d, g = wb - wa, fb - fa - 1
        if d < M:  # Overlap of M - d bp
            penalty += p["overlap_penalty_f1"] + (M - d - 1) * p["overlap_penalty_f2"]
        elif d > M:  # Gap of d - M bp
            penalty += p["skip_penalty_l1"] + (d - M - 1) * p["skip_penalty_l2"]
        if g >= 1:  # Run of g skipped fingers
            penalty += p["skip_penalty_f1"] + (g - 1) * p["skip_penalty_f2"]
    return raw - penalty


def match_chain(actions):
    return [(int(a.split("@")[0][1:]), int(a.split("@")[-1][1:])) for a in actions if "@match@" in a]


# ------------------------------------------------------------------ sections
def negated(p):
    """p with every penalty negated: the convention of PREVIOUS_COMPOSITE."""
    return {k: (-v if k in PENALTY_NAMES else v) for k, v in p.items()}


def test_core_identical(new, orig, n_cases=300, seed=0):
    """0a. The frozen 3 bp core must match the original core exactly, given identical inputs
    (its penalties negated: the original's convention)."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        stack, motif_norm, sim_for, sim_rev, importance = prepare(new, motif, fingers, 3)
        shared = dict(matrix_f=stack, matrix_l=motif_norm, sim_for=sim_for, sim_rev=sim_rev,
                      scale_factor=importance, sim_threshold=0.0,
                      alignment_type="local", initial_score=0.0)
        old = negated(NEUTRAL)  # The original adds its penalties (numbers <= 0)
        out_orig = api(orig, "calculate_modified_bellman_ford")(
            overlap_penalty_f1=old["overlap_penalty_f1"], overlap_penalty_f2=old["overlap_penalty_f2"],
            skip_penalty_l1=old["skip_penalty_l1"], skip_penalty_l2=old["skip_penalty_l2"],
            skip_penalty_f=old["skip_penalty_f1"], **shared)
        out_new = new.calculate_modified_smith_waterman_znf(
            **{k: NEUTRAL[k] for k in PENALTY_NAMES}, **shared)
        if tuple(out_orig[:3]) != tuple(out_new[:3]):
            n_diff += 1
            if n_diff <= 3:
                print(f"  [diff] case {i} F={F} L={L}: {out_orig[0]!r} vs {out_new[0]!r}")
    print(f"0a. frozen core vs original core (identical inputs): {n_cases - n_diff}/{n_cases} identical")
    return n_diff


def test_matches_previous(new, prev, n_cases=40, seed=17):
    """0. Against the composite.py from before the sign change, fed the penalties negated:
    without a skip_penalty_l1 (the two gap rules agree), every path matches bit for bit; and its
    match_penalty_f keeps the alignment and score of match_count_penalty (global_exact). With a
    skip_penalty_l1, a gap of G >= 2 bp now also costs it."""
    rng = np.random.default_rng(seed)
    configs = [ZERO, dict(NEUTRAL, skip_penalty_l1=0.0), dict(NEUTRAL, skip_penalty_l1=0.0, skip_penalty_f2=0.3),
               dict(ZERO, overlap_penalty_f1=0.05, overlap_penalty_f2=0.2, skip_penalty_l2=0.15)]
    n_diff = n_runs = n_cost = n_gap = 0
    for i in range(n_cases):
        M = (3, 4, 2)[i % 3]
        F = int(rng.integers(2, 7))
        motif, fingers = make_case(rng, F, M, int(rng.integers(max(M + 1, 12), 40)))
        for cfg, exact in itertools.product(configs, (True, False)):
            flags = dict(global_exact=exact)
            runs = [("alignment", lambda mod, p: api(mod, "run_modified_bellman_ford_alignment")(
                        target_motif=motif, reference_motif=fingers, finger_length=M, **flags, **p))]
            for label, run in runs:
                a, b = run(new, cfg), run(prev, negated(cfg))
                n_runs += 1
                same = all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a, b))
                if not same:
                    n_diff += 1
                    if n_diff <= 3:
                        print(f"  [diff] {label} M={M} global_exact={exact} {cfg}")
        # The per-match cost: match_penalty_f (in the DP) vs match_count_penalty (by the number of matches)
        sim = rng.random((int(rng.integers(1, 5)), int(rng.integers(2, 9)))) * 1.6 - 0.4
        scale = rng.random(sim.shape[1]) * 0.5
        for cfg, cost in itertools.product((ZERO, dict(NEUTRAL, skip_penalty_l1=0.0)), (0.02, 0.1, 0.5)):
            a = call_core(new, sim, scale, M, match_count_penalty=cost, **cfg)
            b = call_core(prev, sim, scale, M, match_penalty_f=-cost, **negated(cfg))
            n_cost += 1
            if a[2] != b[2] or a[1] != b[1] or abs(a[0] - b[0]) > 1e-12:
                n_diff += 1
                if n_diff <= 3:
                    print(f"  [diff] cost {cost} {cfg}: {a[0]!r} {a[2]} vs {b[0]!r} {b[2]}")
        # With skip_penalty_l1, only alignments with a gap of 2 bp or more can change
        cfg = dict(ZERO, skip_penalty_l1=0.05, skip_penalty_l2=0.1)
        a = call_core(new, sim, scale, M, **cfg)
        b = call_core(prev, sim, scale, M, **negated(cfg))
        chain = match_chain(b[2])
        long_gap = any(wb - wa > M + 1 for (_, wa), (_, wb) in zip(chain, chain[1:]))
        n_gap += long_gap
        if not long_gap and (a[2] != b[2] or a[0] != b[0]):
            n_diff += 1
            print(f"  [diff] skip_penalty_l1 changed an alignment without a 2 bp gap: {a[2]} vs {b[2]}")
    print(f"0. vs previous (penalties negated): {n_runs + n_cost + n_cases - n_diff}/{n_runs + n_cost + n_cases} "
          f"identical ({n_runs} wrapper runs without skip_penalty_l1, {n_cost} match_penalty_f vs "
          f"match_count_penalty, {n_cases} with skip_penalty_l1, {n_gap} of them with a 2+ bp gap)")
    return n_diff


def same_outputs(a, b):
    """Two outputs (nested tuples, lists, dicts, arrays, numbers) equal bit for bit."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_outputs(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same_outputs(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a == b


def test_matches_base(new, base, n_cases=20, seed=22):
    """0c. Against the composite.py from before general_penalty_root and the fixed-length composite
    were removed (run with general_penalty_root=0): the alignment and ZNF wrappers, with their
    matrices, bit for bit. The composite is no longer compared: its match score is the squared
    cosine to the whole motif since 2026-10-30, not the window energy x cosine (section l)."""
    rng = np.random.default_rng(seed)
    n_runs = n_diff = 0
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, L = int(rng.integers(1, 6)), int(rng.integers(M + 2, 20))
        motif = rng.normal(size=(L, 4)) * (rng.random((L, 1)) < 0.8)
        fingers = np.abs(rng.normal(size=(F * M, 4)))
        for p, exact, mc, unsigned in itertools.product(
                (ZERO, NEUTRAL, dict(ZERO, skip_penalty_f2=0.3), dict(ZERO, overlap_penalty_f1=0.05)), (True, False),
                (dict(), dict(match_count_penalty=0.05), dict(match_count_max=2)), (True, False)):
            kw = dict(target_motif=motif, reference_motif=fingers, finger_length=M, global_exact=exact,
                      label_unsigned=unsigned, return_matrices=True, **{k: v for k, v in p.items()}, **mc)
            for fn in ("run_modified_bellman_ford_alignment", "run_modified_smith_waterman_znf"):
                n_runs += 1
                if not same_outputs(api(new, fn)(**kw), api(base, fn)(general_penalty_root=0, **kw)):
                    n_diff += 1
                    if n_diff <= 3:
                        print(f"  [diff] {fn} M={M} {p} global_exact={exact} {mc} unsigned={unsigned}")
    print(f"0c. vs base (general_penalty_root=0): {n_runs - n_diff}/{n_runs} alignment and ZNF runs identical")
    return n_diff


def test_penalty_signs(new):
    """0b. Every penalty is a number >= 0: a negative one (the old convention) is rejected,
    by the core and the wrappers, and match_penalty_f is gone."""
    rng = np.random.default_rng(18)
    motif, fingers = make_case(rng, 3, 3, 20)
    n_bad = 0
    frozen = dict(matrix_l=np.zeros((5, 4)), matrix_f=np.zeros((2, 3, 4)), sim_for=np.ones((2, 3)),
                  sim_rev=np.ones((2, 3)), scale_factor=np.ones(3), sim_threshold=0.0)
    for name in PENALTY_NAMES:
        try:
            new.calculate_modified_smith_waterman_znf(**frozen, **{**{k: 0.0 for k in PENALTY_NAMES}, name: -0.1})
        except ValueError:
            continue
        n_bad += 1
        print(f"  [FAIL] frozen core accepted {name}=-0.1")
    runs = [(name, label, run) for name in PENALTY_NAMES + ("match_count_penalty",) for label, run in (
                ("core", lambda name: call_core(new, np.ones((2, 4)), np.ones(4), 2, **{name: -0.1})),
                ("alignment", lambda name: call_znf(new, motif, fingers, **{name: -0.1})))]
    runs += [(name, "composite", lambda name: new.run_modified_bellman_ford_composite(
                motif, unroll(fingers, 3), **{name: -0.1}))
             for name in ("overlap_penalty_fraction", "match_count_penalty")]
    for name, label, run in runs:
        try:
            run(name)
        except ValueError as exc:
            if "subtracted" not in str(exc):
                n_bad += 1
                print(f"  [FAIL] {label} {name}: {exc}")
            continue
        n_bad += 1
        print(f"  [FAIL] {label} accepted {name}=-0.1")
    for removed in ("match_penalty_f", "general_penalty_root", "overlap_penalty"):
        for label, run in (("alignment", lambda: call_znf(new, motif, fingers, **{removed: 0.1})),
                           ("composite", lambda: new.run_modified_bellman_ford_composite(
                               motif, unroll(fingers, 3), **{removed: 0.1}))):
            if label == "alignment" and removed == "overlap_penalty":
                continue
            try:
                run()
            except TypeError:
                continue
            n_bad += 1
            print(f"  [FAIL] {label} accepted {removed}")
    if (hasattr(new, "run_modified_bellman_ford_composite_fixlen") or hasattr(new, "run_modified_smith_waterman_composite_fixlen")) or hasattr(new, "_match_rescale"):
        n_bad += 1
        print("  [FAIL] run_modified_bellman_ford_composite_fixlen or _match_rescale still defined")
    print(f"0b. penalty signs: negative penalties rejected; match_penalty_f, general_penalty_root, "
          f"overlap_penalty and the fixed-length composite removed: "
          f"{'ok' if n_bad == 0 else f'{n_bad} failures'}")
    return n_bad


def test_degenerate(new):
    """1b. Degenerate inputs must not crash, including a target shorter than a finger."""
    rng = np.random.default_rng(7)
    zero_f = rng.random((9, 4)) + 0.01
    zero_f[3:6] = 0.0
    zero_t = rng.random((20, 4)) + 0.01
    zero_t[:4] = zero_t[-4:] = 0.0  # All-zero windows at either end
    cases = [
        ("single finger", *make_case(rng, 1, 3, 12)),
        ("motif == finger", rng.random((3, 4)) + 0.01, rng.random((3, 4)) + 0.01),
        ("motif shorter than a finger", rng.random((2, 4)) + 0.01, rng.random((6, 4)) + 0.01),
        ("zero finger", rng.random((20, 4)) + 0.01, zero_f),
        ("zero target ends", zero_t, rng.random((9, 4)) + 0.01),
        ("pure noise", rng.random((30, 4)) * 1e-3, rng.random((12, 4)) + 0.01),
    ]
    n_fail = 0
    for name, motif, fingers in cases:
        for config, exact in itertools.product((NEUTRAL, ZERO), (False, True)):
            try:
                out = call_znf(new, motif, fingers, global_exact=exact, **config)
                assert not np.isnan(out[0]) and not np.isnan(out[1]), "NaN score"
            except Exception as exc:
                print(f"  [FAIL] {name} global_exact={exact}: {type(exc).__name__}: {exc}")
                n_fail += 1
        try:
            new.run_modified_bellman_ford_composite(motif, unroll(fingers, 3), subunit_motif_trim=False)
        except Exception as exc:
            print(f"  [FAIL] {name} composite: {type(exc).__name__}: {exc}")
            n_fail += 1
    print(f"1b. degenerate cases: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_matches_snapshot(new, snap, n_cases=40, seed=4):
    """2. Where the scoring did not change (zero penalties, no rescaling), every path
    matches the snapshot run on the padded target, bit for bit."""
    rng = np.random.default_rng(seed)
    n_diff = n_runs = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif3, fingers3 = make_case(rng, F, 3, int(rng.integers(15, 50)))
        motif4, fingers4 = make_case(rng, F, 4, int(rng.integers(20, 60)))
        motif2, fingers2 = make_case(rng, F, 2, int(rng.integers(10, 40)))
        runs = [
            ("znf  M=3", 3, motif3, lambda mod, m, **kw: call_znf(mod, m, fingers3, **kw, **ZERO)),
            ("gen  M=3", 3, motif3, lambda mod, m, **kw: call_general(mod, m, fingers3, 3, **kw, **ZERO)),
            ("gen  M=4", 4, motif4, lambda mod, m, **kw: call_general(mod, m, fingers4, 4, **kw, **ZERO)),
            ("gen  M=2", 2, motif2, lambda mod, m, **kw: call_general(mod, m, fingers2, 2, **kw, **ZERO)),
        ]
        for (label, M, motif, run), exact in itertools.product(runs, (True, False)):
            n_runs += 1
            diffs = differing_from_snapshot(run(new, motif, global_exact=exact), run(snap, pad(motif, M)),
                                            len(motif) + 2 * M - 2)
            if diffs:
                n_diff += 1
                if n_diff <= 3:
                    print(f"  [diff] {label} global_exact={exact}: {diffs}")
    print(f"2. vs snapshot on the padded target (zero penalties, no rescaling), 4 paths x "
          f"global_exact: {n_runs - n_diff}/{n_runs} identical")
    return n_diff


EXACT_CONFIGS = {
    "zero": ZERO,
    "neutral": NEUTRAL,
    "harsh f2": dict(NEUTRAL, skip_penalty_f2=0.3),
    "overlap": dict(ZERO, overlap_penalty_f1=0.05, overlap_penalty_f2=0.2),
    "gap": dict(ZERO, skip_penalty_l1=0.05, skip_penalty_l2=0.15),
    "l1 only": dict(ZERO, skip_penalty_l1=0.1),
    "l2 only": dict(ZERO, skip_penalty_l2=0.1),
    "f-skip": dict(ZERO, skip_penalty_f1=0.05, skip_penalty_f2=0.3),
    "threshold": dict(NEUTRAL, sim_threshold=0.5),
}


def brute_force_cases(n_cases, seed):
    """Tiny random cores, and the best score over every chain of matches, by configuration."""
    rng = np.random.default_rng(seed)
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for name, p in EXACT_CONFIGS.items():
            best = 0.0  # The empty alignment
            for r in range(1, min(F, N) + 1):
                for rows in itertools.combinations(range(F), r):
                    for cols in itertools.combinations(range(N), r):
                        chain = list(zip(rows, cols))
                        if all(sim[f, w] > p["sim_threshold"] for f, w in chain):
                            best = max(best, chain_score(chain, sim, scale, M, p))
            yield name, p, M, sim, scale, best


def core_problems(new, out, sim, scale, M, p):
    """Ways a core's output is inconsistent: its matches must re-score to mbf_score, and its
    state at the optimum must decompose it."""
    F, N = sim.shape
    chain = match_chain(out[2])
    state = out[3]
    problems = []
    if chain and abs(chain_score(chain, sim, scale, M, p) - out[0]) > 1e-12:
        problems.append("matches do not re-score to mbf_score")
    if state["F_total"] != len(chain):
        problems.append(f"F_total {state['F_total']} for {len(chain)} matches")
    if out[0] != (state["H_match"] - state["H_penalty_overlap_f"]
                  - state["H_penalty_skip_f"] - state["H_penalty_skip_l"]):
        problems.append("H != H_match - penalties")
    if min(state["H_penalty_overlap_f"], state["H_penalty_skip_f"], state["H_penalty_skip_l"]) < 0:
        problems.append("a negative penalty sum")
    if out[4].shape != (F + 1, N + 2 * M - 1) or abs(out[4].max() - out[0]) > 1e-12:
        problems.append(f"H_final {out[4].shape}, max {out[4].max()!r}")
    return problems


def test_exact(new, n_cases=90, seed=11):
    """a. With global_exact, the DP is exact: its score is the best over every chain of matches
    (brute force), and its matches re-score to it, for every penalty configuration. Its state at
    the optimum decomposes the score."""
    n_bad = n_runs = 0
    for name, p, M, sim, scale, best in brute_force_cases(n_cases, seed):
        out = call_core(new, sim, scale, M, return_matrices=True, **p)
        problems = core_problems(new, out, sim, scale, M, p)
        if abs(out[0] - best) > 1e-12:
            problems.append(f"score {out[0]!r}, best chain {best!r}")
        n_runs += 1
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] {name} M={M} {sim.shape}: {problems}")
    print(f"a. exact DP: {n_runs - n_bad}/{n_runs} optimal (brute force), re-scored and decomposed, "
          f"{len(EXACT_CONFIGS)} penalty configurations")
    return n_bad


def test_one_state(new, n_cases=90, seed=11):
    """b. Without global_exact (one alignment per cell): never above the best chain, consistent,
    and the exact result whenever no penalty depends on the state (zero penalties). How often it
    misses the best alignment is reported."""
    n_bad = n_runs = 0
    missed, worst = {}, {}
    for name, p, M, sim, scale, best in brute_force_cases(n_cases, seed):
        out = call_core(new, sim, scale, M, global_exact=False, return_matrices=True, **p)
        problems = core_problems(new, out, sim, scale, M, p)
        if out[0] > best + 1e-12:
            problems.append(f"score {out[0]!r} above the best chain {best!r}")
        if name == "zero":
            exact = call_core(new, sim, scale, M, **dict(p))
            if tuple(out[:3]) != tuple(exact[:3]):
                problems.append("differs from global_exact without penalties")
        key = name
        missed[key] = missed.get(key, 0) + (out[0] < best - 1e-12)
        worst[key] = max(worst.get(key, 0.0), best - out[0])
        n_runs += 1
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] {name} M={M} {sim.shape}: {problems}")
    per_config = n_runs // len(missed)
    summary = ", ".join(f"{key} {missed[key]}/{per_config} (max -{worst[key]:.3g})"
                        for key in missed if missed[key])
    print(f"b. one state per cell: {n_runs - n_bad}/{n_runs} consistent, never above the best chain, "
          f"exact without penalties; best alignment missed in: {summary or 'none'}")
    return n_bad


def brute_force_profile(sim, scale, M, p):
    """Best score over chains of exactly N matches, by N (0: the empty alignment; -inf: none)."""
    F, N = sim.shape
    profile = np.full(min(F, N) + 1, -np.inf)
    profile[0] = 0.0
    for r in range(1, min(F, N) + 1):
        for rows in itertools.combinations(range(F), r):
            for cols in itertools.combinations(range(N), r):
                chain = list(zip(rows, cols))
                if all(sim[f, w] > p["sim_threshold"] for f, w in chain):
                    profile[r] = max(profile[r], chain_score(chain, sim, scale, M, p))
    return profile


def test_match_count(new, n_cases=60, seed=14):
    """e. Choosing the number of matches: the profile (best score with exactly N matches) is the
    brute force's with global_exact, and never above it without; the alignment kept has the
    chosen number of matches, its own score, and fewer matches as the penalty grows."""
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    penalties = (0.0, 0.01, 0.05, 0.2, 1.0)
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for (name, p), exact in itertools.product(EXACT_CONFIGS.items(), (True, False)):
            flags = dict(p, global_exact=exact)
            expected = brute_force_profile(sim, scale, M, p)
            reachable = np.isfinite(expected)
            problems = []
            default = call_core(new, sim, scale, M, **flags)
            if default[6] is not None:
                problems.append("a profile without asking")
            profile = call_core(new, sim, scale, M, return_match_profile=True, **flags)[6]
            if profile.shape != expected.shape or np.isfinite(profile[~reachable]).any():
                problems.append(f"profile {profile} for {expected}")
            elif exact and not np.allclose(profile[reachable], expected[reachable], rtol=0, atol=1e-12):
                problems.append(f"profile {profile}, brute force {expected}")
            elif not exact and (profile[reachable] > expected[reachable] + 1e-12).any():
                problems.append(f"profile {profile} above the brute force {expected}")
            if not exact and name == "zero" and not np.array_equal(profile, call_core(
                    new, sim, scale, M, return_match_profile=True, **dict(flags, global_exact=True))[6]):
                problems.append("one state per count differs from global_exact without penalties")
            if exact and abs(profile.max() - default[0]) > 1e-12:
                problems.append(f"best of the profile {profile.max()!r}, best alignment {default[0]!r}")

            counts = []
            for penalty, fraction in itertools.product(penalties, (1.0, 0.9, 0.5)):
                out = call_core(new, sim, scale, M, match_count_penalty=penalty, match_count_fraction=fraction,
                                return_match_profile=True, **flags)
                chosen = new._select_match_count(out[6], penalty, fraction)
                chain = match_chain(out[2])
                if (len(chain) != chosen or abs(out[0] - (out[6][chosen] - penalty * chosen)) > 1e-12
                        or out[3]["F_total"] != chosen):
                    problems.append(f"penalty {penalty} fraction {fraction}: {len(chain)} matches, chose {chosen}")
                elif chain and abs(chain_score(chain, sim, scale, M, p) - penalty * len(chain)
                                   - out[0]) > 1e-12:
                    problems.append("chosen alignment does not re-score to mbf_score")
                if fraction == 1:
                    counts.append(chosen)
            if counts != sorted(counts, reverse=True):
                problems.append(f"more matches with a larger penalty: {counts}")
            if counts[-1] != 0 and profile[1:].max() < 1.0:  # 1 per match: Only the empty alignment
                problems.append(f"penalty 1 kept {counts[-1]} matches")
            n_runs += 1
            if problems:
                n_bad += 1
                if n_bad <= 5:
                    print(f"  [FAIL] {name} global_exact={exact} M={M} {sim.shape}: {problems[:2]}")

    # The rule, by hand
    profile = np.array([0.0, 0.5, 0.8, 0.9, 0.95, -np.inf])
    for penalty, fraction, chosen in ((0.0, 1.0, 4), (0.08, 1.0, 3), (0.12, 1.0, 2), (0.35, 1.0, 1), (0.6, 1.0, 0),
                                      (0.0, 0.9, 3), (0.0, 0.8, 2), (0.0, 0.5, 1), (0.08, 0.5, 1)):
        if new._select_match_count(profile, penalty, fraction) != chosen:
            n_bad += 1
            print(f"  [FAIL] rule: penalty {penalty} fraction {fraction}")
    # A fraction of 1 is the first maximum of the penalized profile (fewest matches), ties included
    for _ in range(200):
        profile = np.round(rng.random(int(rng.integers(1, 9))), 1)  # Rounded: Ties
        profile[0] = 0.0
        profile[rng.random(len(profile)) < 0.2] = -np.inf
        profile[0] = 0.0
        for penalty in (0.0, 0.1, 0.25):
            if new._select_match_count(profile, penalty, 1.0) != int(
                    (profile - penalty * np.arange(len(profile))).argmax()):
                n_bad += 1
                print(f"  [FAIL] fraction 1 on {profile}, penalty {penalty}")
    for fraction in (None, 0.0, 1.5, -0.5):
        try:
            call_core(new, np.ones((2, 4)), np.ones(4), 2, match_count_fraction=fraction)
        except AssertionError:
            continue
        n_bad += 1
        print(f"  [FAIL] accepted match_count_fraction {fraction!r}")

    # Wrappers: One more output, only when asked; the composite keeps fewer subunits as the penalty
    # grows (without a match count penalty, mbf_score is the profile's best: the default is 0.05)
    for j in range(8):
        motif, stack = make_mixed_case(rng, 2, (3, 4)[j % 2], int(rng.integers(15, 30)))
        base = new.run_modified_bellman_ford_composite(motif, stack, match_count_penalty=0.0, target_similarity=None)
        out = new.run_modified_bellman_ford_composite(motif, stack, match_count_penalty=0.0, target_similarity=None,
                                                        return_match_profile=True)
        if (len(base) != 7 or len(out) != 8 or abs(out[7][len(out[3])] - out[0]) > 1e-12
                or abs(out[0] - out[7].max()) > 1e-12 or out[0] != base[0]):
            n_bad += 1
            print(f"  [FAIL] composite outputs {len(base)}, {len(out)}: {out[0]!r} {base[0]!r}")
        sizes = [len(new.run_modified_bellman_ford_composite(
            motif, stack, match_count_penalty=penalty, return_match_profile=True)[3]) for penalty in penalties]
        if sizes != sorted(sizes, reverse=True):
            n_bad += 1
            print(f"  [FAIL] composite: subunits by penalty {sizes}")
        fingers = stack.reshape(-1, 4)
        aligned = call_znf(new, motif, fingers, finger_length=stack.shape[1], return_match_profile=True)
        if len(aligned) != 11 or len(call_znf(new, motif, fingers, finger_length=stack.shape[1])) != 10:
            n_bad += 1
            print(f"  [FAIL] alignment outputs {len(aligned)}")
    try:
        call_core(new, np.ones((2, 4)), np.ones(4), 2, alignment_type="global", match_count_penalty=0.1)
    except AssertionError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] chose a number of matches for a global alignment")
    print(f"e. number of matches: {n_runs - n_bad}/{n_runs} profiles and choices right ({len(EXACT_CONFIGS)} "
          f"penalty configurations x global_exact); rule and wrappers "
          f"{'ok' if n_bad == 0 else 'FAIL'}")
    return n_bad


def test_match_cost(new, n_cases=60, seed=15):
    """f. match_count_penalty as a per-match cost: with global_exact, mbf_score is the best over
    every chain of matches (brute force, the empty one included) of its score minus the cost per
    match; fewer matches as the cost grows; and hand cases."""
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for config in (ZERO, NEUTRAL):
            problems = []
            counts = []
            profile = brute_force_profile(sim, scale, M, config)
            for cost in (0.0, 0.01, 0.05, 0.2, 1.0):
                out = call_core(new, sim, scale, M, match_count_penalty=cost, **config)
                chain = match_chain(out[2])
                counts.append(len(chain))
                best = (profile - cost * np.arange(len(profile))).max()
                if abs(out[0] - best) > 1e-12:
                    problems.append(f"cost {cost}: score {out[0]!r}, best chain {best!r}")
                elif chain and abs(chain_score(chain, sim, scale, M, config) - cost * len(chain) - out[0]) > 1e-12:
                    problems.append(f"cost {cost}: matches do not re-score to mbf_score")
            if counts != sorted(counts, reverse=True):
                problems.append(f"more matches with a larger cost: {counts}")
            n_runs += 1
            if problems:
                n_bad += 1
                if n_bad <= 5:
                    print(f"  [FAIL] M={M} {sim.shape}: {problems[:2]}")

    # By hand: a match is kept exactly if it adds more than the cost
    sim = np.full((2, 4), -1.0)
    sim[0, 0], sim[1, 2] = 0.3, 0.1
    for cost, expected, score in ((0.2, [(0, 0)], 0.1), (0.05, [(0, 0), (1, 2)], 0.3), (0.5, [], 0.0)):
        out = call_core(new, sim, np.ones(4), 2, match_count_penalty=cost, **ZERO)
        if match_chain(out[2]) != expected or abs(out[0] - score) > 1e-12:
            n_bad += 1
            print(f"  [FAIL] cost {cost}: {out[0]!r} {out[2]}")

    # Wrappers: The composite keeps fewer subunits as the cost grows
    for j in range(4):
        motif, stack = make_mixed_case(rng, 2, 3, int(rng.integers(15, 25)))
        sizes = [len(new.run_modified_bellman_ford_composite(motif, stack, match_count_penalty=cost)[3])
                 for cost in (0.0, 0.02, 0.1)]
        if sizes != sorted(sizes, reverse=True):
            n_bad += 1
            print(f"  [FAIL] composite: subunits by cost {sizes}")
    print(f"f. match cost: {n_runs - n_bad}/{n_runs} optimal with the cost (brute force) and shrink with the "
          f"cost; hand cases and wrappers {'ok' if n_bad == 0 else 'FAIL'}")
    return n_bad


def test_match_count_max(new, n_cases=60, seed=19):
    """h. match_count_max limits the number of matches whatever the penalty and fraction: with
    global_exact, mbf_score is the best over every chain of at most that many matches (brute
    force), minus the penalty; the profile stops at it; a limit of min(F, N) or more is no limit;
    and the composite wrapper limits to 2 by default, and to at most 2."""
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(1, 6)), int(rng.integers(2, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for name, p in [(k, EXACT_CONFIGS[k]) for k in ("zero", "neutral", "gap", "f-skip")]:
            expected = brute_force_profile(sim, scale, M, p)
            for cap, cost, exact in itertools.product((1, 2, 3), (0.0, 0.05), (True, False)):
                out = call_core(new, sim, scale, M, match_count_max=cap, match_count_penalty=cost,
                                return_match_profile=True, global_exact=exact, **p)
                chain, profile = match_chain(out[2]), out[6]
                n_runs += 1
                problems = []
                if len(chain) > cap or out[3]["F_total"] != len(chain):
                    problems.append(f"{len(chain)} matches over the limit {cap}")
                if len(profile) != min(F, N, cap) + 1:
                    problems.append(f"profile of {len(profile)} for the limit {cap}")
                capped = expected[:min(F, N, cap) + 1]
                best = (capped - cost * np.arange(len(capped))).max()
                if exact and (abs(out[0] - best) > 1e-12 or not np.allclose(
                        profile[np.isfinite(capped)], capped[np.isfinite(capped)], rtol=0, atol=1e-12)):
                    problems.append(f"score {out[0]!r}, best of at most {cap} {best!r}")
                if not exact and out[0] > best + 1e-12:
                    problems.append(f"score {out[0]!r} above the best of at most {cap} {best!r}")
                if problems:
                    n_bad += 1
                    if n_bad <= 5:
                        print(f"  [FAIL] {name} cap={cap} cost={cost} exact={exact} M={M} "
                              f"{sim.shape}: {problems[:2]}")
            # A limit of min(F, N) or more is no limit
            for exact in (True, False):
                free = call_core(new, sim, scale, M, return_match_profile=True, global_exact=exact, **p)
                big = call_core(new, sim, scale, M, return_match_profile=True, global_exact=exact,
                                match_count_max=min(F, N) + int(rng.integers(0, 3)), **p)
                n_runs += 1
                if free[:3] != big[:3] or not np.array_equal(free[6], big[6]):
                    n_bad += 1
                    print(f"  [FAIL] {name} exact={exact}: a limit of min(F, N) or more changed the result")
    for bad in (0, -1, 1.5, True, "2"):
        try:
            call_core(new, np.ones((2, 4)), np.ones(4), 2, match_count_max=bad, **ZERO)
        except ValueError:
            continue
        n_bad += 1
        print(f"  [FAIL] accepted match_count_max={bad!r}")
    try:
        call_core(new, np.ones((2, 4)), np.ones(4), 2, alignment_type="global", match_count_max=2, **ZERO)
    except AssertionError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] limited the number of matches of a global alignment")

    # Wrappers: The composite limits to 2 by default (here: 5 planted subunits), and rejects more; the
    # alignment wrapper does not limit by default, and its limit holds for its local alignment
    for j in range(6):
        subunits, target, planted = plant_subunits(rng, 2, (3, 4, 5)[j % 3], 5)
        kw = dict(sim_threshold=0.99, subunit_motif_trim=0)
        sizes = [len(new.run_modified_bellman_ford_composite(target, subunits, **kw, **extra)[3])
                 for extra in ({}, dict(match_count_max=1), dict(match_count_max=2))]
        if sizes != [2, 1, 2]:
            n_bad += 1
            print(f"  [FAIL] composite: matches by limit (default, 1, 2) {sizes}")
        for bad in (None, 3, 5):
            try:
                new.run_modified_bellman_ford_composite(target, subunits, match_count_max=bad, **kw)
            except ValueError:
                continue
            n_bad += 1
            print(f"  [FAIL] composite: accepted match_count_max={bad!r}")
        none = len(new.run_modified_bellman_ford_composite(target, subunits, collapse="none", match_count_max=None,
                                                             **kw)[3])
        if none != 5:
            n_bad += 1
            print(f"  [FAIL] composite collapse='none': {none} matches without a limit, expected 5")
    motif, fingers = make_case(rng, 6, 3, 30)
    if differing(call_znf(new, motif, fingers), call_znf(new, motif, fingers, match_count_max=None)):
        n_bad += 1
        print("  [FAIL] alignment wrapper: the default is not no limit")
    print(f"h. match_count_max: {n_runs - n_bad}/{n_runs} within the limit and optimal under it (brute force), "
          f"no-op at min(F, N); validation and wrappers {'ok' if n_bad == 0 else 'FAIL'}")
    return n_bad


def test_signed(new, seed=20):
    """i. label_unsigned=False compares signed motifs on 8 channels (each base's positive and
    negative part apart), so that a negative region only matches a negative subunit; True keeps
    4 channels on absolute values (bit for bit as before: see sections 0 and 2)."""
    from unittest.mock import patch
    import MotifCompendium.utils.motif as utils_motif
    rng = np.random.default_rng(seed)
    n_bad = 0
    # The 8 channels reverse complement as [::-1, ::-1]
    w = rng.normal(size=(20, 5, 4))
    if not np.array_equal(utils_motif.motif_4_to_8(w[:, ::-1, ::-1]), utils_motif.motif_4_to_8(w)[:, ::-1, ::-1]):
        n_bad += 1
        print("  [FAIL] 8 channels do not reverse complement as [::-1, ::-1]")
    # Channels of the similarity: 8 when signed, 4 when unsigned
    motif, stack = make_mixed_case(rng, 2, 3, 20)
    motif = motif * np.where(rng.random(motif.shape) < 0.3, -1, 1)
    for unsigned, channels in ((False, 8), (True, 4)):
        seen = []
        core = new.compute_aligned_similarity
        def record(x, y):
            seen.append((x.shape[-1], y.shape[-1]))
            return core(x, y)
        with patch.object(new, "compute_aligned_similarity", side_effect=record):
            new.run_modified_bellman_ford_composite(motif, stack, label_unsigned=unsigned)
        if not seen or any(c != (channels, channels) for c in seen):
            n_bad += 1
            print(f"  [FAIL] label_unsigned={unsigned}: similarity on channels {seen}")
    # A positive subunit a, and a negated subunit b: signed, only a and -b match
    for M in (3, 4, 5):
        a, b = np.eye(4)[[0] * (M - 1) + [1]], np.eye(4)[[2] + [3] * (M - 1)]
        target = np.concatenate([a, np.full((3, 4), 0.01), -b])
        stack = np.stack([a, b, -b])
        kw = dict(sim_threshold=0.9, allow_reverse_comp=False, match_count_max=2, subunit_motif_trim=0)
        signed = new.run_modified_bellman_ford_composite(target, stack, label_unsigned=False, **kw)
        unsigned = new.run_modified_bellman_ford_composite(target, stack, label_unsigned=True, **kw)
        starts = list(signed[5] - (M - 1))
        if list(signed[3]) != [0, 2] or starts != [0, M + 3]:
            n_bad += 1
            print(f"  [FAIL] M={M} signed: subunits {list(signed[3])} at {starts}, expected [0, 2] at [0, {M + 3}]")
        if list(unsigned[3])[:1] != [0] or len(unsigned[3]) != 2 or unsigned[3][1] not in (1, 2):
            n_bad += 1
            print(f"  [FAIL] M={M} unsigned: subunits {list(unsigned[3])}")
    print(f"i. label_unsigned: 8-channel signed similarity, 4-channel unsigned, signs respected: "
          f"{'ok' if n_bad == 0 else f'{n_bad} failures'}")
    return n_bad


def overlap_cost(prev, cur, lengths, W):
    """Summed weight of the positions of cur that prev (the previous candidate) overlaps."""
    (c0, l0), (c, l) = prev, cur
    end = min(l0 + lengths[c0], l + lengths[c])
    return W[end] - W[l] if end > l else 0.0


def brute_force_chains(scores, matchable, lengths, penalty, max_count, weight):
    """Best score with exactly N matches over every chain of (candidate, start) with strictly
    increasing starts, by N: H_match - penalty * the weight the consecutive candidates overlap."""
    W = np.concatenate([[0.0], np.cumsum(np.concatenate([weight, np.zeros(int(max(lengths)) + 2)]))])
    pairs = [(c, l) for c, l in zip(*np.nonzero(matchable))]
    pairs.sort(key=lambda x: x[1])
    top = len({l for _, l in pairs}) if max_count is None else min(len({l for _, l in pairs}), max_count)
    profile = np.full(top + 1, -np.inf)
    profile[0] = 0.0
    def extend(chain, h, pen):
        n = len(chain)
        if n:
            profile[n] = max(profile[n], h - pen)
        if n == top:
            return
        for c, l in pairs:
            if chain and l <= chain[-1][1]:
                continue
            o = overlap_cost(chain[-1], (c, l), lengths, W) if chain else 0.0
            extend(chain + [(c, l)], h + scores[c, l], pen + penalty * o)
    extend([], 0.0, 0.0)
    return profile


def pool(x_i, x_j, p):
    """Lp pool of two nonnegative arrays, entry by entry: (x_i^p + x_j^p)^(1/p), the maximum for p = inf."""
    return np.maximum(x_i, x_j) if np.isinf(p) else (x_i ** p + x_j ** p) ** (1.0 / p)


def collapse_pool_p(collapse, collapse_p):
    """The order of the Lp pool of a collapse (None: the sum or mean, or lsq_norm)."""
    return {"max_norm": np.inf, "lp_norm": collapse_p}.get(collapse)


def collapse_pair(collapse, collapse_p, v_i, v_j, lo, hi):
    """The collapsed alignment of two scaled placements: their sum; their Lp pool (max_norm,
    lp_norm); halved where both windows cover (mean_norm)."""
    q = collapse_pool_p(collapse, collapse_p)
    if q is not None:
        return pool(v_i, v_j, q)
    r = v_i + v_j
    if hi > lo and collapse == "mean_norm":
        r[lo:hi] /= 2
    return r


def pair_score(target, collapse, collapse_p, b_i, b_j, u_i, u_j, lo, hi):
    """Squared score of a pair of placements (unit vectors u, projections b), before the overlap
    penalty, and the scale of each over its b: the energy the collapsed alignment explains,
    L2-normalized (<t, r>^2 / |r|^2); lsq_norm: the energy of the non-negative least-squares fit of the
    two (scipy's nnls), which is at its least-squares scale already."""
    if collapse == "lsq_norm":
        from scipy.optimize import nnls
        basis = np.stack([u_i.ravel(), u_j.ravel()], axis=1)
        x, _ = nnls(basis, target.ravel())
        r = basis @ x
        return 2 * float(target.ravel() @ r) - float(r @ r), [x[0] / b_i if b_i > 0 else 0.0, x[1] / b_j if b_j > 0 else 0.0]
    r = collapse_pair(collapse, collapse_p, b_i * u_i, b_j * u_j, lo, hi)
    tr, rr = float((target * r).sum()), float((r * r).sum())
    if rr <= 0:
        return -np.inf, [1.0, 1.0]
    return tr * tr / rr, [tr / rr, tr / rr]


def brute_force_collapsed(target, proj, starts, ends, vectors, weight, penalty, collapse="sum_norm", collapse_p=None):
    """Best squared score over every pair of placements with distinct starts (pair_score(), minus
    penalty * the weight of the positions both windows cover), and the pair, by start."""
    W = np.concatenate([[0.0], np.cumsum(weight)])
    value, pair = -np.inf, None
    for i, j in itertools.combinations(range(len(proj)), 2):
        if starts[i] == starts[j]:
            continue
        lo, hi = max(starts[i], starts[j]), min(ends[i], ends[j])
        e, _ = pair_score(target, collapse, collapse_p, proj[i], proj[j], vectors[i], vectors[j], lo, hi)
        v = e - (penalty * (W[hi] - W[lo]) if hi > lo else 0.0)
        if v > value + 1e-12:
            value, pair = v, sorted([i, j], key=lambda k: starts[k])
    return value, pair


COLLAPSES = [("sum_norm", None), ("mean_norm", None), ("none", None), ("max_norm", None), ("lp_norm", 2.0),
             ("lp_norm", 3.5), ("lsq_norm", None)]


def test_composite_score(new, n_cases=24 * len(COLLAPSES), seed=23):
    """l. run_modified_bellman_ford_composite()'s score is the cosine similarity of the motif with
    its collapsed alignment, L2-normalized: each subunit, placed at its start (zero elsewhere), is
    scaled by its cosine to the whole padded, L2-normalized motif (its least-squares weight), and the
    scaled subunits are summed (sum_norm), pooled by their maximum (max_norm) or Lp norm (lp_norm)
    entry by entry, averaged where they overlap (mean_norm), summed with both scales refit by
    non-negative least squares (lsq_norm, scipy's nnls), or not collapsed (none: the squared scores add,
    minus the overlap penalty). Computed here
    from scratch over every alignment (brute force): the profile, the reported matches re-scored,
    and the scales; every collapse with every trim, penalty, threshold and limit. Also: an
    imperfect match scores its cosine; the defaults (no
    overlap penalty, 0.05 per subunit in cosine units, target similarity 0.9, at most 2 subunits);
    target_similarity stops at the fewest subunits reaching it; lp_norm with p = 1 and p = inf is
    sum_norm and max_norm; and match_count_max above 2 and a misused collapse_p are rejected."""
    import inspect
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    for i in range(n_cases):
        F, M, L = int(rng.integers(1, 3)), int(rng.integers(3, 6)), int(rng.integers(5, 10))
        target = rng.normal(size=(L, 4)) * (rng.random((L, 1)) < 0.8)
        stack = np.abs(rng.normal(size=(F, M, 4)))  # No all-zero subunit (trimming one away raises)
        n_modes = len(COLLAPSES)
        collapse, collapse_p = COLLAPSES[i % n_modes]
        trim = (True, 0)[(i // n_modes) % 2]
        penalty = (0.0, 0.5)[(i // (2 * n_modes)) % 2]
        sim_threshold = (0.0, 0.4)[(i // (4 * n_modes)) % 2]
        max_count = ((1, 2, 2) if collapse != "none" else (1, 2, 3))[(i // (8 * n_modes)) % 3]
        out = new.run_modified_bellman_ford_composite(
            target, stack, sim_threshold=sim_threshold, overlap_penalty_fraction=penalty, match_count_max=max_count,
            match_count_penalty=0.0, target_similarity=None, subunit_motif_trim=trim, return_match_profile=True,
            collapse=collapse, collapse_p=collapse_p)

        # From scratch: Candidates (as given, reverse complemented), on the padded, normalized target
        subunits = new._trim_subunits(np.abs(stack), trim)
        candidates = [(s_[::-1, ::-1] if rc else s_) for s_ in subunits for rc in (0, 1)]
        candidates = [c / np.linalg.norm(c) if np.linalg.norm(c) > 0 else c for c in candidates]
        lengths = np.array([len(c) for c in candidates])
        padded = np.pad(np.abs(target), ((M - 1, M - 1), (0, 0)))
        target_norm = padded / np.linalg.norm(padded)
        S = len(target_norm)
        scores, matchable = np.zeros((len(candidates), S)), np.zeros((len(candidates), S), dtype=bool)
        placements, proj, starts, ends, vectors = [], [], [], [], []
        for c, cand in enumerate(candidates):
            for l in range(S - len(cand) + 1):
                window = target_norm[l:l + len(cand)]
                b = float((window * cand).sum())  # <motif, placed subunit>: both have norm 1
                cos_window = b / np.linalg.norm(window) if np.linalg.norm(window) > 0 else 0.0
                scores[c, l], matchable[c, l] = b * b, cos_window > sim_threshold
                if cos_window > sim_threshold:
                    placed = np.zeros_like(target_norm)
                    placed[l:l + len(cand)] = cand
                    placements.append((c, l)); proj.append(b); starts.append(l); ends.append(l + len(cand))
                    vectors.append(placed)
        proj, starts, ends = np.array(proj), np.array(starts), np.array(ends)
        weight = (target_norm * target_norm).sum(axis=1)
        if collapse == "none":
            expected = brute_force_chains(scores, matchable, lengths, penalty, max_count, weight)
            expected = np.where(np.isfinite(expected), np.sqrt(np.maximum(expected, 0.0)), -np.inf)
        else:
            expected = np.full(max_count + 1, -np.inf)
            expected[0] = 0.0
            if len(proj):
                expected[1] = proj.max()
            if len(proj) and max_count == 2:
                value, _ = brute_force_collapsed(target_norm, proj, starts, ends, vectors, weight, penalty, collapse,
                                                 collapse_p)
                expected[2] = np.sqrt(max(value, 0.0)) if np.isfinite(value) else -np.inf
        profile = out[7]
        n_runs += 1
        problems = []
        if profile.shape != expected.shape or not np.allclose(profile, expected, rtol=0, atol=1e-9):
            problems.append(f"profile {np.round(profile, 4)} vs from scratch {np.round(expected, 4)}")
        else:
            # The reported matches re-score to mbf_score; their scales
            matches = [placements.index((2 * int(f) + int(rc), int(h))) for f, rc, h in zip(out[3], out[4], out[5])]
            if not matches:
                rescored, scales = 0.0, []
            elif collapse == "none":
                W = np.concatenate([[0.0], np.cumsum(np.concatenate([weight, np.zeros(int(lengths.max()) + 2)]))])
                chain = [placements[k] for k in matches]
                energy = (sum(proj[k] ** 2 for k in matches)
                          - penalty * sum(overlap_cost(x, y, lengths, W) for x, y in zip(chain, chain[1:])))
                rescored, scales = np.sqrt(max(energy, 0.0)), [proj[k] for k in matches]
            elif len(matches) == 1:
                rescored, scales = proj[matches[0]], [proj[matches[0]]]
            else:
                k_i, k_j = matches
                lo, hi = max(starts[k_i], starts[k_j]), min(ends[k_i], ends[k_j])
                e, f = pair_score(target_norm, collapse, collapse_p, proj[k_i], proj[k_j], vectors[k_i], vectors[k_j], lo, hi)
                pen = penalty * float(weight[lo:hi].sum()) if hi > lo else 0.0
                rescored, scales = np.sqrt(max(e - pen, 0.0)), [proj[k_i] * f[0], proj[k_j] * f[1]]
            if abs(rescored - out[0]) > 1e-9 or abs(out[0] - profile.max()) > 1e-9:
                problems.append(f"matches {matches} re-score to {rescored!r}, mbf_score {out[0]!r}")
            if not np.allclose(out[6], scales, rtol=0, atol=1e-9):
                problems.append(f"scales {out[6]} vs {scales}")
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] {collapse} (p={collapse_p}) F={F} M={M} L={L} trim={trim} p={penalty} thr={sim_threshold} "
                      f"max={max_count}: {problems[:2]}")

    # An imperfect match scores its cosine to the motif
    target = np.eye(4)[[0, 0, 1, 1]] + 0.3
    sub = np.eye(4)[[0, 0, 1, 1]]
    out = new.run_modified_bellman_ford_composite(target, sub[None], subunit_motif_trim=0, return_match_profile=True,
                                                    allow_reverse_comp=False)
    cos = float((target / np.linalg.norm(target) * sub / np.linalg.norm(sub)).sum())
    if abs(out[7][1] - cos) > 1e-12:
        n_bad += 1
        print(f"  [FAIL] one imperfect match: {out[7][1]!r}, cosine {cos!r}")

    # Defaults: no overlap penalty, 0.05 per subunit, target similarity 0.9, at most 2 subunits
    parameters = inspect.signature(new.run_modified_bellman_ford_composite).parameters
    defaults = tuple(parameters[k].default for k in ("overlap_penalty_fraction", "match_count_penalty",
                                                     "target_similarity", "match_count_max"))
    target = np.concatenate([np.eye(4)[[0, 1, 2]], np.eye(4)[[2, 3, 0]]]) + 0.05
    stack = np.stack([np.eye(4)[[0, 1, 2, 2]], np.eye(4)[[2, 2, 3, 0]]])
    same = same_outputs(new.run_modified_bellman_ford_composite(target, stack, subunit_motif_trim=0),
                        new.run_modified_bellman_ford_composite(
                            target, stack, subunit_motif_trim=0, overlap_penalty_fraction=0.0, match_count_penalty=0.05,
                            target_similarity=0.9, match_count_max=2))
    if defaults != (0.0, 0.05, 0.9, 2) or not same:
        n_bad += 1
        print(f"  [FAIL] defaults {defaults!r}, same as passing them: {same}")

    # Target similarity: A subunit explaining most of the motif is kept alone, even if a second raises the cosine
    target = np.concatenate([np.eye(4)[[0, 1, 2, 3, 0, 1]], np.eye(4)[[3]] * 0.4])
    stack = np.stack([np.eye(4)[[0, 1, 2, 3, 0, 1]], np.eye(4)[[3, 3, 3, 3, 3, 3]]])
    kw = dict(subunit_motif_trim=0, allow_reverse_comp=False, match_count_penalty=0.0, return_match_profile=True)
    stop = new.run_modified_bellman_ford_composite(target, stack, target_similarity=0.9, **kw)
    free = new.run_modified_bellman_ford_composite(target, stack, target_similarity=None, **kw)
    if not (stop[7][1] >= 0.9 and stop[7][2] > stop[7][1] and len(stop[3]) == 1 and len(free[3]) == 2):
        n_bad += 1
        print(f"  [FAIL] target similarity: profile {stop[7]}, subunits {len(stop[3])} (0.9) and {len(free[3])} (None)")

    # Validation
    for bad in (None, 0, 3, 5, True):
        try:
            new.run_modified_bellman_ford_composite(target, stack, match_count_max=bad)
        except ValueError:
            continue
        n_bad += 1
        print(f"  [FAIL] accepted match_count_max={bad!r}")
    try:
        new.run_modified_bellman_ford_composite(target, stack, target_similarity="0.9")
    except ValueError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] accepted target_similarity='0.9'")
    try:
        new.run_modified_bellman_ford_composite(target, stack, collapse="max")
    except ValueError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] accepted collapse='max'")
    for kw_bad in (dict(collapse="lp_norm"), dict(collapse="lp_norm", collapse_p=0.5), dict(collapse="lp_norm", collapse_p=True),
                   dict(collapse="lp_norm", collapse_p="2"), dict(collapse="sum_norm", collapse_p=2.0),
                   dict(collapse="max_norm", collapse_p=np.inf)):
        try:
            new.run_modified_bellman_ford_composite(target, stack, **kw_bad)
        except ValueError:
            continue
        n_bad += 1
        print(f"  [FAIL] accepted {kw_bad}")

    # lp_norm: p = 1 is sum_norm and p = inf is max_norm, output for output
    rng = np.random.default_rng(seed + 1)
    for _ in range(8):
        target = rng.normal(size=(int(rng.integers(6, 12)), 4))
        stack = np.abs(rng.normal(size=(3, 5, 4)))
        kw = dict(match_count_penalty=0.0, target_similarity=None, return_match_profile=True)
        for (c_a, p_a), (c_b, p_b) in ((("lp_norm", 1), ("sum_norm", None)), (("lp_norm", np.inf), ("max_norm", None))):
            out_a = new.run_modified_bellman_ford_composite(target, stack, collapse=c_a, collapse_p=p_a, **kw)
            out_b = new.run_modified_bellman_ford_composite(target, stack, collapse=c_b, collapse_p=p_b, **kw)
            if not same_outputs(out_a, out_b):
                n_bad += 1
                print(f"  [FAIL] lp_norm p={p_a} differs from {c_b}")
        # Signed (8 channels): The pooled collapse is scored, finite
        for c_s, p_s in (("max_norm", None), ("lp_norm", 2.0)):
            out_s = new.run_modified_bellman_ford_composite(rng.normal(size=(10, 4)), rng.normal(size=(3, 5, 4)),
                                                              label_unsigned=False, collapse=c_s, collapse_p=p_s, **kw)
            if not (np.isfinite(out_s[0]) and np.isfinite(out_s[1])):
                n_bad += 1
                print(f"  [FAIL] signed {c_s}: mbf_score {out_s[0]!r}, sim_score {out_s[1]!r}")
    for limit in (3, None):
        new.run_modified_bellman_ford_composite(target, stack, collapse="none", match_count_max=limit,
                                                  subunit_motif_trim=0)
    print(f"l. composite score (cosine of the collapsed alignment): {n_runs - n_bad}/{n_runs} profiles from scratch "
          f"(brute force), matches re-scored, scales; imperfect match, defaults, target similarity and validation "
          f"{'ok' if n_bad == 0 else f'FAIL ({n_bad})'}")
    return n_bad


def test_pooled_search(new, n_cases=14, seed=31):
    """m. The pair searches are exact at scale: on random nonnegative targets with up to 400
    placements of random lengths and starts, the best pair scores the same as the best over every
    pair (brute force), with and without an overlap penalty, in small blocks so that the searches
    cross blocks and prune; also when every pair overlaps (no pair without overlap to start from).
    _best_pooled_pair(): lp_norm p = 1.5, 2, 4 and max_norm; _best_collapsed_pair(): lsq_norm and
    sum_norm; _best_mean_pair(): mean_norm. The penalty is on the motif energy of the overlap, or the
    masked one (overlap_cost_prefix: the later-starting placement's own <motif, b u> on the overlap)."""
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    modes = [("lp 1.5", 1.5), ("lp 2", 2.0), ("lp 4", 4.0), ("max_norm", np.inf), ("lsq_norm", None), ("sum_norm", 1.0),
             ("mean_norm", None)]
    for case in range(n_cases):
        P, D = int(rng.integers(20, 50)), 4
        t = np.abs(rng.normal(size=(P, D))) * (rng.random((P, 1)) < 0.85)
        t /= np.linalg.norm(t)
        K = int(rng.integers(40, 400))
        narrow = case % 4 == 3  # Every window covers the middle: every pair overlaps
        lengths = rng.integers(3, 12, size=K)
        if narrow:
            starts = np.array([int(rng.integers(max(0, P // 2 - m + 1), P // 2 + 1)) for m in lengths])
        else:
            starts = np.array([int(rng.integers(0, P - m + 1)) for m in lengths])
        ends = starts + lengths
        unit, proj, U = [], np.zeros(K), np.zeros((K, P, D))
        for k in range(K):
            u = np.abs(rng.normal(size=(lengths[k], D))) * (rng.random((lengths[k], 1)) < 0.9) + 1e-3
            u /= np.linalg.norm(u)
            unit.append(u)
            U[k, starts[k]:ends[k]] = u
            proj[k] = float((t[starts[k]:ends[k]] * u).sum())
        cands = np.arange(K)
        weight = (t * t).sum(axis=1)
        W = np.concatenate([[0.0], np.cumsum(weight)])
        V = U * proj[:, None, None]  # (K, P, D): Scaled placements
        i_all, j_all = np.triu_indices(K, 1)
        keep = starts[i_all] != starts[j_all]
        i_all, j_all = i_all[keep], j_all[keep]
        lo_all, hi_all = np.maximum(starts[i_all], starts[j_all]), np.minimum(ends[i_all], ends[j_all])
        overlap = np.where(hi_all > lo_all, W[np.maximum(hi_all, lo_all)] - W[lo_all], 0.0)
        # Masked: <motif, u> by position of each placement, summed over its first j positions
        prefix = np.zeros((K, int(lengths.max()) + 1))
        for k in range(K):
            prefix[k, 1:lengths[k] + 1] = np.cumsum((t[starts[k]:ends[k]] * unit[k]).sum(axis=1))
            prefix[k, lengths[k] + 1:] = prefix[k, lengths[k]]
        later = np.where(starts[i_all] > starts[j_all], i_all, j_all)
        masked_all = np.where(hi_all > lo_all, proj[later] * prefix[later, np.clip(hi_all - starts[later], 0, None)], 0.0)

        def brute(name, q, penalty, masked=False):
            best = -np.inf
            for c0 in range(0, len(i_all), 20000):
                ii, jj = i_all[c0:c0 + 20000], j_all[c0:c0 + 20000]
                lo, hi = lo_all[c0:c0 + 20000], hi_all[c0:c0 + 20000]
                if name == "lsq_norm":
                    b_i, b_j = proj[ii], proj[jj]
                    g = (U[ii] * U[jj]).sum(axis=(1, 2))
                    det = 1 - g * g
                    x_i, x_j = (b_i - g * b_j) / np.where(det > 1e-12, det, 1.0), (b_j - g * b_i) / np.where(det > 1e-12, det, 1.0)
                    both = (x_i >= 0) & (x_j >= 0) & (det > 1e-12)
                    val = np.where(both, x_i * b_i + x_j * b_j, np.maximum(b_i ** 2, b_j ** 2))
                else:
                    if name == "mean_norm":
                        r = V[ii] + V[jj]
                        both = (np.arange(P)[None, :] >= lo[:, None]) & (np.arange(P)[None, :] < hi[:, None])
                        r = (r * np.where(both, 0.5, 1.0)[:, :, None]).reshape(len(ii), -1)
                    else:
                        r = pool(V[ii], V[jj], q).reshape(len(ii), -1)
                    y, z2 = r @ t.reshape(-1), (r * r).sum(axis=1)
                    val = np.where(y > 0, y * y / np.maximum(z2, 1e-300), -np.inf)
                charge = masked_all[c0:c0 + 20000] if masked else overlap[c0:c0 + 20000]
                best = max(best, float((val - penalty * charge).max()))
            return best

        for name, q in modes:
            for penalty, masked in ((0.0, False), (0.3, False), (0.3, True), (1.0, True)):
                expected = brute(name, q, penalty, masked)
                args = (proj, starts, ends, cands, unit)
                kw = dict(overlap_cost_prefix=penalty * proj[:, None] * prefix) if masked else {}
                if name in ("lsq_norm", "sum_norm"):
                    value, pair, _ = new._best_collapsed_pair(*args, weight, penalty, row_block=16, col_block=64, score=name, **kw)
                elif name == "mean_norm":
                    value, pair, _ = new._best_mean_pair(*args, t, weight, penalty, row_block=16, **kw)
                else:
                    value, pair, _ = new._best_pooled_pair(*args, t, weight, penalty, q, row_block=16, col_block=64,
                                                           pair_block=32, **kw)
                n_runs += 1
                if abs(value - expected) > 1e-9 or (pair is not None and starts[pair[0]] >= starts[pair[1]]):
                    n_bad += 1
                    if n_bad <= 5:
                        print(f"  [FAIL] case {case} K={K} P={P} {name} penalty={penalty} masked={masked} narrow={narrow}: value {value!r} "
                              f"vs brute force {expected!r}, pair {pair}")
    print(f"m. pair searches at scale (lp_norm, max_norm, lsq_norm, sum_norm, mean_norm; energy and masked penalties): "
          f"{n_runs - n_bad}/{n_runs} best pairs exact (brute force over every pair)")
    return n_bad


def masked_cost(prev, cur, starts, ends, proj, q, penalty):
    """The masked overlap penalty of placement cur after prev: penalty * b_cur times cur's own
    <motif, candidate>, summed over the positions of cur that prev covers (q: per position of cur)."""
    j = min(ends[prev], ends[cur]) - starts[cur]
    return penalty * proj[cur] * float(q[cur][:j].sum()) if j > 0 else 0.0


def brute_force_masked(proj, q, starts, ends, penalty, max_count):
    """Best squared score with exactly N matches, by N, and its chain, over every chain of placements with
    strictly increasing starts: the sum of b^2, minus the masked overlap penalty of each placement after the first."""
    order = sorted(range(len(proj)), key=lambda k: starts[k])
    n_starts = len({int(s) for s in starts})
    top = n_starts if max_count is None else min(n_starts, max_count)
    profile, chains = np.full(top + 1, -np.inf), [[] for _ in range(top + 1)]
    profile[0] = 0.0
    def extend(chain, v):
        n = len(chain)
        if n and v > profile[n]:
            profile[n], chains[n] = v, list(chain)
        if n == top:
            return
        for k in order:
            if chain and starts[k] <= starts[chain[-1]]:
                continue
            extend(chain + [k], v + proj[k] ** 2 - (masked_cost(chain[-1], k, starts, ends, proj, q, penalty) if chain else 0.0))
    extend([], 0.0)
    return profile, chains


def test_masked_overlap(new, n_cases=96, n_pair_cases=24 * 5, seed=37):
    """n. The masked overlap penalty (overlap_penalty_mode="masked", the default): an incoming subunit's
    window similarity, over its whole window, still decides whether it is matchable, but its match score
    b^2 = <motif, b u> loses penalty * b * <motif, u> over the positions the previous subunit covers, not
    rescaled. Computed here from scratch (per-position dot products, unsigned and signed on 8 channels),
    with penalties 1 and 0.5, both trims and thresholds: collapse="none" over every chain of up to 3
    subunits (brute force), its profile, the reported matches re-scored, the scales (b), and sim_score,
    the similarity of the reconstruction with each incoming subunit taken out, times the penalty, where
    it covers the previous one; and sum_norm, mean_norm, max_norm, lp_norm and lsq_norm over every pair
    (their collapsed score, minus the later-starting subunit's masked cost), with the matches re-scored
    and the scales. Also: with penalty 0 it is the energy mode, output for output; the defaults (masked,
    penalty 0, sum_norm); and an unknown mode is rejected."""
    import inspect
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    pair_collapses = [("sum_norm", None), ("mean_norm", None), ("max_norm", None), ("lp_norm", 2.0), ("lsq_norm", None)]
    for i in range(n_cases + n_pair_cases):
        chain_case = i < n_cases
        F, M, L = int(rng.integers(1, 4)), int(rng.integers(3, 6)), int(rng.integers(5, 10))
        signed = i % 4 == 3 and chain_case
        target = rng.normal(size=(L, 4)) * (rng.random((L, 1)) < 0.8)
        stack = rng.normal(size=(F, M, 4)) if signed else np.abs(rng.normal(size=(F, M, 4)))
        penalty = (1.0, 0.5)[(i // 4) % 2]
        trim = (True, 0)[(i // 8) % 2]
        sim_threshold = (0.0, 0.4)[(i // 16) % 2]
        if chain_case:
            collapse, collapse_p, max_count = "none", None, (1, 2, 3)[(i // 32) % 3]
        else:
            collapse, collapse_p = pair_collapses[i % len(pair_collapses)]
            max_count = (2, 1)[(i // 64) % 2]
        out = new.run_modified_bellman_ford_composite(
            target, stack, sim_threshold=sim_threshold, overlap_penalty_fraction=penalty, overlap_penalty_mode="masked",
            collapse=collapse, collapse_p=collapse_p, match_count_max=max_count, match_count_penalty=0.0,
            target_similarity=None, subunit_motif_trim=trim, return_match_profile=True, label_unsigned=not signed)

        # From scratch: Candidates (as given, reverse complemented), on the padded, normalized target
        split = (lambda x: np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], axis=-1)) if signed else (lambda x: x)
        subunits = new._trim_subunits(stack if signed else np.abs(stack), trim)
        candidates = [(s_[::-1, ::-1] if rc else s_) for s_ in subunits for rc in (0, 1)]
        candidates = [c / np.linalg.norm(c) if np.linalg.norm(c) > 0 else c for c in candidates]
        padded = np.pad(target if signed else np.abs(target), ((M - 1, M - 1), (0, 0)))
        target_norm = padded / np.linalg.norm(padded)
        placements, proj, starts, ends, q, vectors = [], [], [], [], [], []
        for c, cand in enumerate(candidates):
            for l in range(len(target_norm) - len(cand) + 1):
                window = target_norm[l:l + len(cand)]
                per_position = (split(window) * split(cand)).sum(axis=1)  # <motif, candidate>, by position
                b = float(per_position.sum())
                cos_window = b / np.linalg.norm(window) if np.linalg.norm(window) > 0 else 0.0
                if cos_window > sim_threshold:
                    placements.append((c, l)); proj.append(max(b, 0.0)); starts.append(l); ends.append(l + len(cand))
                    q.append(per_position)
                    placed = np.zeros_like(target_norm)
                    placed[l:l + len(cand)] = cand
                    vectors.append(placed)
        proj, starts, ends = np.array(proj), np.array(starts), np.array(ends)
        later = lambda x, y: (x, y) if starts[x] < starts[y] else (y, x)  # (earlier, later)
        if chain_case:
            expected, _ = brute_force_masked(proj, q, starts, ends, penalty, max_count)
            expected = np.where(np.isfinite(expected), np.sqrt(np.maximum(expected, 0.0)), -np.inf)
        else:
            expected = np.full(max_count + 1, -np.inf)
            expected[0] = 0.0
            if len(proj):
                expected[1] = proj.max()
            if len(proj) and max_count == 2:
                best = -np.inf
                for x, y in itertools.combinations(range(len(proj)), 2):
                    if starts[x] == starts[y]:
                        continue
                    lo, hi = max(starts[x], starts[y]), min(ends[x], ends[y])
                    e, _ = pair_score(target_norm, collapse, collapse_p, proj[x], proj[y], vectors[x], vectors[y], lo, hi)
                    best = max(best, e - masked_cost(*later(x, y), starts, ends, proj, q, penalty))
                expected[2] = np.sqrt(max(best, 0.0)) if np.isfinite(best) else -np.inf
        profile = out[7]
        n_runs += 1
        problems = []
        if profile.shape != expected.shape or not np.allclose(profile, expected, rtol=0, atol=1e-9):
            problems.append(f"profile {np.round(profile, 4)} vs from scratch {np.round(expected, 4)}")
        else:
            matches = [placements.index((2 * int(f) + int(rc), int(h))) for f, rc, h in zip(out[3], out[4], out[5])]
            if chain_case or len(matches) < 2:
                energy = sum(proj[k] ** 2 for k in matches) - sum(
                    masked_cost(a, b, starts, ends, proj, q, penalty) for a, b in zip(matches, matches[1:]))
                scales = [proj[k] for k in matches]
            else:
                k_i, k_j = matches
                lo, hi = max(starts[k_i], starts[k_j]), min(ends[k_i], ends[k_j])
                e, f = pair_score(target_norm, collapse, collapse_p, proj[k_i], proj[k_j], vectors[k_i], vectors[k_j], lo, hi)
                energy = e - masked_cost(*later(k_i, k_j), starts, ends, proj, q, penalty)
                scales = [proj[k_i] * f[0], proj[k_j] * f[1]]
            rescored = np.sqrt(max(energy, 0.0)) if matches else 0.0
            if abs(rescored - out[0]) > 1e-9 or abs(out[0] - profile.max()) > 1e-9:
                problems.append(f"matches {matches} re-score to {rescored!r}, mbf_score {out[0]!r}")
            if not np.allclose(out[6], scales, rtol=0, atol=1e-9):
                problems.append(f"scales {out[6]} vs {scales}")
            if chain_case and matches:  # sim_score: The reconstruction, each incoming subunit masked where it covers the previous one
                recon = np.zeros_like(padded)
                for n_i, k in enumerate(matches):
                    placed = np.zeros_like(padded)
                    placed[starts[k]:ends[k]] = candidates[placements[k][0]] * proj[k]
                    if n_i:
                        hi = min(ends[matches[n_i - 1]], ends[k])
                        if hi > starts[k]:
                            placed[starts[k]:hi] *= 1 - penalty
                    recon += placed
                ic = new.utils_config.get_ic_scale()
                new.utils_config.set_ic_scale(False)
                sim_expected = new.utils_similarity.compute_similarities([recon[np.newaxis], padded[np.newaxis]], [(0, 1)])[0][0]
                new.utils_config.set_ic_scale(ic)
                if abs(float(np.squeeze(out[1])) - float(np.squeeze(sim_expected))) > 1e-9:
                    problems.append(f"sim_score {float(np.squeeze(out[1]))!r} vs the masked reconstruction's "
                                    f"{float(np.squeeze(sim_expected))!r}")
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] masked {collapse} (p={collapse_p}) F={F} M={M} L={L} signed={signed} penalty={penalty} "
                      f"trim={trim} thr={sim_threshold} max={max_count}: {problems[:2]}")

    # Penalty 0: The energy mode, output for output, with every collapse
    for n_i in range(12):
        target, stack = rng.normal(size=(int(rng.integers(6, 12)), 4)), np.abs(rng.normal(size=(3, 5, 4)))
        collapse, collapse_p = ([("none", None)] + pair_collapses)[n_i % 6]
        kw = dict(collapse=collapse, collapse_p=collapse_p, overlap_penalty_fraction=0.0, match_count_penalty=0.0,
                  target_similarity=None, return_match_profile=True)
        if not same_outputs(new.run_modified_bellman_ford_composite(target, stack, overlap_penalty_mode="masked", **kw),
                            new.run_modified_bellman_ford_composite(target, stack, overlap_penalty_mode="energy", **kw)):
            n_bad += 1
            print(f"  [FAIL] masked with penalty 0 differs from energy ({collapse})")

    # Defaults (unpinned): the masked mode, no penalty, sum_norm
    unpinned = getattr(new.run_modified_bellman_ford_composite, "func", new.run_modified_bellman_ford_composite)
    parameters = inspect.signature(unpinned).parameters
    defaults = tuple(parameters[k].default for k in ("overlap_penalty_mode", "overlap_penalty_fraction", "collapse"))
    if defaults != ("masked", 0.0, "sum_norm"):
        n_bad += 1
        print(f"  [FAIL] defaults {defaults!r}")

    # Validation: known modes only
    target, stack = rng.normal(size=(8, 4)), np.abs(rng.normal(size=(2, 4, 4)))
    try:
        new.run_modified_bellman_ford_composite(target, stack, overlap_penalty_mode="mask")
    except ValueError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] accepted overlap_penalty_mode='mask'")
    print(f"n. masked overlap penalty: {n_runs - n_bad}/{n_runs} profiles from scratch (brute force; chains with none, "
          f"pairs with every other collapse), matches re-scored, scales and sim_score; penalty 0, defaults and "
          f"validation {'ok' if n_bad == 0 else f'FAIL ({n_bad})'}")
    return n_bad


def test_composite_varlen(new, seed=21):
    """j. run_modified_bellman_ford_composite(): subunits of any lengths (subunit_motif_trim), ordered by
    position. The chain DP (collapse="none") is the best over every chain (brute force), by number
    of matches, with an overlap penalty (per bp or energy-weighted) and a limit; the pair searches
    (sum_norm, mean_norm) are the best over every pair of placements, with or without an overlap
    penalty, across several blocks; planted composites of mixed lengths, in any
    order, orientation and repetition, are recovered; and signed motifs, trimming and the default
    limit of 2."""
    import MotifCompendium.utils.motif as utils_motif
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    # (a) The chain DP, against brute force
    for i in range(150):
        C, S = int(rng.integers(1, 5)), int(rng.integers(3, 9))
        lengths = rng.integers(1, 5, size=C)
        scores = rng.random((C, S))
        matchable = rng.random((C, S)) < 0.5
        penalty = (0.0, 0.05, 0.3)[i % 3]
        max_count = (None, 1, 2, 3)[(i // 3) % 4]
        weight = np.ones(S) if i % 2 else rng.random(S)  # Per bp, or a random energy profile
        penalty = penalty * (1 if i % 2 else 3)
        profile, trace = new._chain_composite(scores, matchable, lengths, penalty, weight, max_count)
        expected = brute_force_chains(scores, matchable, lengths, penalty, max_count, weight)
        W = np.concatenate([[0.0], np.cumsum(np.concatenate([weight, np.zeros(int(max(lengths)) + 2)]))])
        n_runs += 1
        problems = []
        if profile.shape != expected.shape or not np.allclose(profile, expected, rtol=0, atol=1e-12):
            problems.append(f"profile {np.round(profile, 4)} vs brute force {np.round(expected, 4)}")
        else:
            for n in range(1, len(profile)):
                if not np.isfinite(profile[n]):
                    continue
                chain = trace(n)
                h = sum(scores[c, l] for c, l in chain)
                o = sum(overlap_cost(a, b, lengths, W) for a, b in zip(chain, chain[1:]))
                if (len(chain) != n or any(not matchable[c, l] for c, l in chain)
                        or any(b[1] <= a[1] for a, b in zip(chain, chain[1:]))
                        or abs(h - penalty * o - profile[n]) > 1e-12):
                    problems.append(f"traced chain {chain} does not re-score to {profile[n]!r}")
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] chain C={C} S={S} p={penalty} max={max_count}: {problems[:1]}")
    # (b) The pair searches (sum_norm, mean_norm), against brute force (small blocks)
    for i in range(150):
        K, S, D = int(rng.integers(2, 12)), int(rng.integers(4, 12)), (4, 8)[i % 2]
        target = np.abs(rng.normal(size=(S, D))) + 0.01  # Every placement has b > 0, as every matchable one
        target = target / np.linalg.norm(target)
        starts = rng.integers(0, S, size=K)
        ends = np.minimum(starts + rng.integers(1, 5, size=K), S)
        unit, vectors = [], []
        for k in range(K):
            u = np.abs(rng.normal(size=(ends[k] - starts[k], D))) * (rng.random((ends[k] - starts[k], 1)) < 0.8)
            u[int(rng.integers(len(u)))] += 0.01  # Not all zero
            u = u / np.linalg.norm(u)
            v = np.zeros((S, D))
            v[starts[k]:ends[k]] = u
            unit.append(u)
            vectors.append(v)
        proj = np.array([float((target * v).sum()) for v in vectors])  # b = <target, placed candidate>
        weight = (target * target).sum(axis=1)
        penalty = (0.0, 0.3)[i % 2]
        collapse = ("sum_norm", "mean_norm")[(i // 2) % 2]
        if collapse == "sum_norm":
            value, pair, scale = new._best_collapsed_pair(proj, starts, ends, np.arange(K), unit, weight, penalty,
                                                          row_block=2, col_block=3)
        else:
            value, pair, scale = new._best_mean_pair(proj, starts, ends, np.arange(K), unit, target, weight, penalty,
                                                     row_block=2)
        expected, _ = brute_force_collapsed(target, proj, starts, ends, vectors, weight, penalty, collapse)
        n_runs += 1
        problems = []
        if not (np.isclose(value, expected, rtol=0, atol=1e-12) or (np.isinf(value) and np.isinf(expected))):
            problems.append(f"value {value!r} vs brute force {expected!r}")
        elif pair is not None:
            i0, j0 = pair
            r = proj[i0] * vectors[i0] + proj[j0] * vectors[j0]
            lo, hi = max(starts[i0], starts[j0]), min(ends[i0], ends[j0])
            if collapse == "mean_norm" and hi > lo:
                r[lo:hi] /= 2
            tr, rr = float((target * r).sum()), float((r * r).sum())
            pen = penalty * float(weight[lo:hi].sum()) if hi > lo else 0.0
            if starts[i0] >= starts[j0] or abs(tr * tr / rr - pen - value) > 1e-12 or abs(scale - tr / rr) > 1e-12:
                problems.append(f"pair {pair} does not re-score to {value!r} (scale {scale!r})")
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] {collapse} pair K={K} S={S} D={D} p={penalty}: {problems[:1]}")
    # (c) Trimming: As logo_trimming
    stack = np.zeros((3, 8, 4))
    stack[0, 2:5] = rng.random((3, 4)) + 0.1
    stack[1, :8] = rng.random((8, 4)) * np.array([0.01, 1, 1, 1, 1, 1, 1, 0.01])[:, None]
    stack[2, 1:7] = rng.random((6, 4)) + 0.05
    for trim, expected in ((False, [8, 8, 8]), (0, [3, 8, 6]), (True, None), (0.1, None)):
        got = [len(x) for x in new._trim_subunits(stack, trim)]
        if expected is None:
            threshold = 1 / 8 if trim is True else trim
            expected = [len(utils_motif.trim_motif(stack[f], threshold)) for f in range(3)]
        if got != expected:
            n_bad += 1
            print(f"  [FAIL] subunit_motif_trim={trim!r}: lengths {got}, expected {expected}")
    for bad in (-0.1, 1.5, "1", None):
        try:
            new._trim_subunits(stack, bad)
        except ValueError:
            continue
        n_bad += 1
        print(f"  [FAIL] accepted subunit_motif_trim={bad!r}")
    try:
        new._trim_subunits(np.zeros((1, 4, 4)), 0)
    except ValueError:
        pass
    else:
        n_bad += 1
        print("  [FAIL] accepted a subunit trimmed away entirely")
    # (d) Planted composites of mixed lengths, in any order, orientation and repetition
    for i in range(30):
        M = 8
        F = int(rng.integers(1, 4))
        for _ in range(200):
            subunits = [np.eye(4)[rng.integers(0, 4, size=int(rng.integers(3, M + 1)))] for _ in range(F)]
            planted = [(int(rng.integers(F)), int(rng.random() < 0.5)) for _ in range(int(rng.integers(1, 3)))]
            pieces = [subunits[f][::-1, ::-1] if rc else subunits[f] for f, rc in planted]
            gaps = [int(rng.integers(0, 3)) for _ in planted]
            target = np.concatenate([np.concatenate([np.full((g, 4), 0.0), p]) for g, p in zip(gaps, pieces)])
            # Distinct oriented subunits, each found on the target only where it is planted
            oriented = {(subunits[f][::-1, ::-1] if rc else subunits[f]).tobytes(): (f, rc) for f in range(F) for rc in (0, 1)}
            sites, pos = set(), 0
            for g, x, (f, rc) in zip(gaps, pieces, planted):
                sites.add((f, rc, pos + g))
                pos += g + len(x)
            found = {(f, rc, st) for f in range(F) for rc in (0, 1)
                     for x in [subunits[f][::-1, ::-1] if rc else subunits[f]]
                     for st in range(len(target) - len(x) + 1) if np.array_equal(target[st:st + len(x)], x)}
            if len(oriented) == 2 * F and {(f, st) for f, _, st in found} == {(f, st) for f, _, st in sites}:
                break
        stack = np.zeros((F, M, 4))
        offsets = [int(rng.integers(0, M - len(x) + 1)) for x in subunits]
        for f, (x, o) in enumerate(zip(subunits, offsets)):
            stack[f, o:o + len(x)] = x
        out = new.run_modified_bellman_ford_composite(target, stack, subunit_motif_trim=0, sim_threshold=0.99,
                                                        target_similarity=None)
        starts, pos = [], 0
        for g, x in zip(gaps, pieces):
            starts.append(pos + g)
            pos += g + len(x)
        expected = [(f, rc, st) for (f, rc), st in zip(planted, starts)]
        reported = [(int(f), int(rc), int(h) - (M - 1)) for f, rc, h in zip(out[3], out[4], out[5])]
        n_runs += 1
        # A palindromic subunit may be reported in either orientation
        same = len(reported) == len(expected) and all(
            r[0] == e[0] and r[2] == e[2] and (r[1] == e[1] or np.array_equal(subunits[e[0]], subunits[e[0]][::-1, ::-1]))
            for r, e in zip(reported, expected))
        if not same:
            n_bad += 1
            if n_bad <= 10:
                print(f"  [FAIL] planted {expected}: reported {reported}")
        elif not np.isclose(float(np.ravel(out[1])[0]), 1.0):
            n_bad += 1
            print(f"  [FAIL] planted {expected}: complete reconstruction has sim_score {out[1]}")
    # (e) Signed: A positive subunit a and a negated b only match a and -b
    for M in (3, 4, 5):
        a, b = np.eye(4)[[0] * (M - 1) + [1]], np.eye(4)[[2] + [3] * (M - 1)]
        target = np.concatenate([a, np.full((3, 4), 0.01), -b])
        stack = np.stack([a, b, -b])
        out = new.run_modified_bellman_ford_composite(target, stack, subunit_motif_trim=0, sim_threshold=0.9,
                                                        allow_reverse_comp=False, label_unsigned=False)
        if list(out[3]) != [0, 2] or list(out[5] - (M - 1)) != [0, M + 3]:
            n_bad += 1
            print(f"  [FAIL] signed M={M}: subunits {list(out[3])} at {list(out[5] - (M - 1))}")
    # (f) At most 2 matches by default
    x = np.eye(4)[[0, 1, 2, 3]]
    target = np.concatenate([x] * 5)
    out = new.run_modified_bellman_ford_composite(target, x[None], subunit_motif_trim=0, sim_threshold=0.99)
    if len(out[3]) != 2:
        n_bad += 1
        print(f"  [FAIL] default limit: {len(out[3])} matches")
    print(f"j. variable-length composite: {n_runs - n_bad}/{n_runs} chains and pairs optimal (brute force), planted "
          f"recovered; trimming, signs and the default limit "
          f"{'ok' if n_bad == 0 else f'FAIL ({n_bad})'}")
    return n_bad


def test_storage(new, seed=16):
    """g. What the DP stores: H, by state for global_exact and, without it, with only the state
    fields a penalty depends on; both engines store the same thing without state axes."""
    rng = np.random.default_rng(seed)
    n_bad = 0
    for i in range(12):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(2, 5)), int(rng.integers(4, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for config in (ZERO, NEUTRAL, dict(ZERO, skip_penalty_f2=0.3), dict(ZERO, overlap_penalty_f1=0.1)):
            track_l = any(config[k] != 0 for k in ("overlap_penalty_f1", "overlap_penalty_f2",
                                                   "skip_penalty_l1", "skip_penalty_l2"))
            track_f = config["skip_penalty_f1"] != config["skip_penalty_f2"]
            names = ("H",)
            axes = {name for name, on in (("L_skip", track_l), ("F_skip", track_f)) if on}
            outs = {}
            for exact in (True, False):
                out = call_core(new, sim, scale, M, global_exact=exact, return_matrices=True, **config)
                final = out[5]
                parts = [key for key in final if key.startswith("H")]
                fields = {key for key in final if key in ("L_skip", "F_skip", "F_total")}
                problems = []
                if tuple(parts) != names:
                    problems.append(f"parts {parts}")
                if exact and set(final["axes"]) != axes:
                    problems.append(f"axes {set(final['axes'])}")
                if exact and final[names[0]].shape != (F + 1, N + 2 * M - 1) + tuple(
                        len(final["axes"][a]) for a in final["axes"]):
                    problems.append(f"shape {final[names[0]].shape}")
                if not exact and (fields != axes or final["axes"]
                                  or final[names[0]].shape != (F + 1, N + 2 * M - 1)):
                    problems.append(f"fields {fields}")
                if set(out[3]) != set(_EMPTY) or out[3]["F_total"] != len(match_chain(out[2])):
                    problems.append("mbf_state")
                outs[exact] = out
                if problems:
                    n_bad += 1
                    print(f"  [FAIL] global_exact={exact} {config}: {problems}")
            if not axes and not np.array_equal(outs[True][5]["H"], outs[False][5]["H"]):
                n_bad += 1
                print(f"  [FAIL] engines differ without state axes: {config}")
    print(f"g. storage: H and the states as needed, same in both engines without state axes: "
          f"{'ok' if n_bad == 0 else f'{n_bad} failures'}")
    return n_bad


_EMPTY = ("H_match", "H_penalty_overlap_f", "H_penalty_skip_f", "H_penalty_skip_l", "F_total")


def test_l2_norm_removed(new, seed=12):
    """d. 'l2_norm' is not an overlap penalty any more."""
    rng = np.random.default_rng(seed)
    n_bad = 0
    motif, fingers = make_case(rng, 3, 3, 20)
    for spelling in ("l2_norm", "L2_NORM"):
        for label, run in (
                ("core", lambda: call_core(new, np.ones((2, 4)), np.ones(4), 2, overlap_penalty_f1=spelling)),
                ("alignment", lambda: call_znf(new, motif, fingers, overlap_penalty_f2=spelling)),
                ("composite", lambda: new.run_modified_bellman_ford_composite(
                    motif, unroll(fingers, 3), overlap_penalty_fraction=spelling))):
            try:
                run()
            except ValueError:
                continue
            n_bad += 1
            print(f"  [FAIL] {label} accepted {spelling!r}")
    print(f"d. 'l2_norm' rejected: {'ok' if n_bad == 0 else f'{n_bad} failures'}")
    return n_bad


def test_padding(new):
    """c. A subunit may overlap either end of the target by 1 bp, and the target may be
    shorter than a subunit."""
    n_bad = 0
    for M in (2, 3, 4, 5):
        # A subunit A C..C G: its first and last bp occur nowhere else, and the filler is T
        s = np.eye(4)[[0] + [1] * (M - 2) + [2]]
        for middle in (1, 3, 6):
            # Only the subunit's last bp at the start, and its first bp at the end: A 1 bp overlap
            # has cosine 1/sqrt(M), and every other window 0. No match count penalty: a 1 bp overlap
            # explains too little of the target to pass the default (0.05)
            target = np.concatenate([s[M - 1:], np.eye(4)[[3] * middle], s[:1]])
            L = len(target)
            out = new.run_modified_bellman_ford_composite(
                target, s[None], sim_threshold=0.5 / math.sqrt(M), allow_reverse_comp=False,
                subunit_motif_trim=0, match_count_max=2, match_count_penalty=0.0)
            (_, _, stack, idx, rc, h, _) = out
            if list(h) != [0, L + M - 2] or stack.shape != (2, L + 2 * M - 2, 4):
                n_bad += 1
                print(f"  [FAIL] M={M} L={L}: placed at {list(h)}, stack {stack.shape}")
                continue
            composite = stack.sum(axis=0)[M - 1:M - 1 + L]  # On the target
            if not (composite[0].argmax() == s[M - 1].argmax() and composite[-1].argmax() == s[0].argmax()):
                n_bad += 1
                print(f"  [FAIL] M={M} L={L}: overhanging bp misplaced")
    for M, L in ((3, 1), (3, 2), (5, 3)):
        rng = np.random.default_rng(M * 10 + L)
        out = new.run_modified_bellman_ford_composite(rng.random((L, 4)), rng.random((2, M, 4)))
        aligned = call_znf(new, rng.random((L, 4)), rng.random((2 * M, 4)), finger_length=M)
        if out[2].shape[1:] != (L + 2 * M - 2, 4) or aligned[2].shape != (2, L + 2 * M - 2, 4):
            n_bad += 1
            print(f"  [FAIL] target shorter than M: M={M} L={L}: {out[2].shape}, {aligned[2].shape}")
    print(f"c. padding: 1 bp overlaps at either end, targets shorter than a subunit: "
          f"{'ok' if n_bad == 0 else f'{n_bad} failures'}")
    return n_bad


def track_violations(align_f, align_l, actions, F, M, L):
    """Ways a pair of alignment tracks fails to line up with the placements in actions."""
    problems = []
    if len(align_f) != len(align_l):
        return [f"lengths {len(align_f)} != {len(align_l)}"]
    if [x for x in align_f if x != '-'] != list(range(F * M)):
        problems.append("finger track is not every finger column once, in order")
    if [x for x in align_l if x != '-'] != list(range(L)):
        problems.append("motif track is not every motif position once, in order")
    if any(a == '-' and b == '-' for a, b in zip(align_f, align_l)):
        problems.append("a column with nothing in either track")
    placed = {}
    for action in actions:
        parts = action.split("@")
        if parts[1] == "match":
            placed[int(parts[0][1:])] = int(parts[-1][1:])
    faces = {a: b for a, b in zip(align_f, align_l) if a != '-'}  # Finger column -> what it faces
    covered_earlier = set()
    for f in range(F):
        for k in range(M):
            shown = faces.get(f * M + k)
            if f not in placed:
                if shown != '-':
                    problems.append(f"unmatched finger {f} column {k} faces {shown}")
            elif shown == '-':
                if placed[f] + k not in covered_earlier:
                    problems.append(f"finger {f} column {k} faces '-' but position {placed[f] + k} is not an overlap")
            elif shown != placed[f] + k:
                problems.append(f"finger {f} column {k} faces {shown}, placed at {placed[f] + k}")
        if f in placed:
            covered_earlier.update(range(placed[f], placed[f] + M))
    return problems


def test_alignment_tracks(new, n_cases=200, seed=6):
    """7. The alignment tracks line up with the placements, on every path and penalty mode."""
    rng = np.random.default_rng(seed)
    modes = [dict(), ZERO, dict(ZERO, match_count_max=2), dict(global_exact=False),
             dict(global_exact=False, match_count_penalty=0.05)]
    n_bad = n_runs = n_overlap = 0
    for i in range(n_cases):
        M = (3, 3, 2, 4, 5)[i % 5]
        F = int(rng.integers(1, 11))
        L = int(rng.integers(max(M + 1, 12), 61))
        motif, fingers = make_case(rng, F, M, L)
        for mode in modes:
            out = call_znf(new, motif, fingers, **mode) if M == 3 else call_general(new, motif, fingers, M, **mode)
            n_runs += 1
            n_overlap += any("F-overlap" in a for a in out[7])
            problems = track_violations(out[4], out[5], out[7], F, M, L + 2 * M - 2)  # Padded positions
            if problems:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  [FAIL] M={M} F={F} L={L} {mode}: {problems[:2]}")
    for F, M, L in ((1, 3, 3), (4, 3, 20), (3, 5, 9)):  # No alignment at all
        a_f, a_l = new._alignment_tracks([], F, M, L)
        if track_violations(a_f, a_l, [], F, M, L):
            print(f"  [FAIL] empty alignment F={F} M={M} L={L}")
            n_bad += 1
    print(f"7. alignment tracks: {n_runs - n_bad}/{n_runs} line up with the placements "
          f"({n_overlap} of them contain overlaps); empty alignments ok")
    return n_bad


def test_new_params_noop(new, n_cases=100, seed=1):
    """3. The finger-skip parameters are no-ops at neutral values."""
    rng = np.random.default_rng(seed)
    n_diff = n_changed = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif, fingers = make_case(rng, F, 3, int(rng.integers(15, 50)))
        base = call_znf(new, motif, fingers)
        same = call_znf(new, motif, fingers, skip_penalty_f1=0.1, skip_penalty_f2=0.1)
        if differing(base, same):
            n_diff += 1
        harsh = call_znf(new, motif, fingers, skip_penalty_f2=0.3)
        if harsh[0] != base[0]:
            n_changed += 1
    print(f"3. finger-skip params: {n_cases - n_diff}/{n_cases} neutral-identical; "
          f"{n_changed}/{n_cases} changed under skip_f2=0.3")
    return n_diff


def rc_of(motif):
    return motif[::-1, ::-1]


def plant_subunits(rng, F, M, n, tries=500):
    """n end-to-end placements of F one-hot subunits, each forward or reverse complemented.
    Every orientation of every subunit is distinct, and no off-frame window equals one."""
    for _ in range(tries):
        subunits = np.eye(4)[rng.integers(0, 4, size=(F, M))]
        planted = [(int(rng.integers(F)), int(rng.random() < 0.5)) for _ in range(n)]
        target = np.concatenate([rc_of(subunits[s]) if rc else subunits[s] for s, rc in planted])
        oriented = {(rc_of(subunits[s]) if rc else subunits[s]).tobytes() for s in range(F) for rc in (0, 1)}
        off_frame = {target[i:i + M].tobytes() for i in range(len(target) - M + 1) if i % M}
        if len(oriented) == 2 * F and not (oriented & off_frame):
            return subunits, target, planted
    raise RuntimeError(f"could not plant distinct one-hot subunits for F={F} M={M} n={n}")


def make_mixed_case(rng, F, M, L, noise=0.05):
    """A subunit stack, and a motif planted with random subunits in random orientations."""
    stack = rng.random((F, M, 4)) ** 2 + 0.01
    motif = rng.random((L, 4)) * noise
    pos = int(rng.integers(0, max(1, L - M)))
    while pos + M <= L:
        part = stack[int(rng.integers(F))]
        motif[pos:pos + M] += (rc_of(part) if rng.random() < 0.5 else part) * (0.5 + rng.random())
        pos = pos + max(1, M + int(rng.integers(-2, 3)))  # Overlap / adjacent / gap
    return motif, stack


def test_general_m(new, seed=2):
    """5. Any finger length M > 1 runs, and recovers a planted arrangement."""
    rng = np.random.default_rng(seed)
    n_fail = 0
    for M in [2, 3, 4, 5, 7]:
        for F in [1, 3, 6]:
            L = max(M + 1, int(rng.integers(3 * M, 8 * M)))
            motif, fingers = make_case(rng, F, M, L)
            try:
                out = call_general(new, motif, fingers, M)
            except Exception as exc:
                print(f"  [FAIL] M={M} F={F} L={L}: {type(exc).__name__}: {exc}")
                n_fail += 1
                continue
            if out[2].shape != (F, L + 2 * M - 2, 4):
                print(f"  [FAIL] M={M} F={F} L={L}: aligned shape {out[2].shape}")
                n_fail += 1

    # Planted fingers, at padded positions (the target starts at M-1)
    for M in [2, 3, 4, 5, 7]:
        fingers, motif = plant_one_hot(rng, 4, M)
        matches = [a for a in call_general(new, motif, fingers, M)[7] if a.split("@")[1] == "match"]
        if matches != [f"F{f}@match@L{f * M + M - 1}" for f in range(4)]:
            print(f"  [FAIL] planted adjacent M={M}: got {matches}")
            n_fail += 1

    for M in [2, 3, 4, 5]:
        fingers, _ = plant_one_hot(rng, 3, M)
        motif = np.concatenate([fingers[:2 * M], np.full((1, 4), 0.02), fingers[2 * M:]])
        matches = [a for a in call_general(new, motif, fingers, M)[7] if a.split("@")[1] == "match"]
        if matches != [f"F0@match@L{M - 1}", f"F1@match@L{2 * M - 1}", f"F2@match@L{3 * M}"]:
            print(f"  [FAIL] planted gap M={M}: got {matches}")
            n_fail += 1

    # The alignment wrapper is the generalized core at every M, and the (deprecated) ZNF wrapper
    # is the alignment wrapper
    for M in [2, 3, 3, 3, 4, 5]:
        motif, fingers = make_case(rng, 4, M, int(rng.integers(4 * M, 8 * M)))
        for exact in (True, False):
            flags = dict(global_exact=exact)
            aligned = new.run_modified_bellman_ford_alignment(target_motif=motif, reference_motif=fingers,
                                                                finger_length=M, **flags, **NEUTRAL)
            for label, other in (("generalized path", call_general(new, motif, fingers, M, **flags)),
                                 ("ZNF wrapper", call_znf(new, motif, fingers, finger_length=M, **flags))):
                diffs = differing(aligned, other)
                if diffs:
                    print(f"  [FAIL] alignment wrapper at M={M} differs from the {label}: {diffs}")
                    n_fail += 1
    print(f"5. general M: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_ic_scale(new):
    """k. ic_scale: the composite wrappers IC-scale by default, the alignment wrappers do not; with
    ic_scale, every wrapper is the same as fed the IC-scaled motifs without it (IC-scaled once, the
    global IC-scale setting adding nothing), and the global setting is restored."""
    rng = np.random.default_rng(11)
    n_fail = 0
    motif_ic, config = new.utils_motif.ic_scale, new.utils_config

    def make(L):
        return (rng.dirichlet(np.ones(4) * 0.3, size=L) * rng.choice([-1, 1], size=(L, 1))
                * rng.uniform(0.2, 1, size=(L, 1)))

    wrappers = {
        "composite": (new.run_modified_bellman_ford_composite, True, 3),
        "alignment": (lambda t, f, **k: new.run_modified_bellman_ford_alignment(
            t, f.reshape(-1, 4), finger_length=f.shape[1], **k), False, 3),
        "znf": (lambda t, f, **k: new.run_modified_smith_waterman_znf(
            t, f.reshape(-1, 4), finger_length=f.shape[1], **k), False, 3),
    }
    setting = config.get_ic_scale()
    for name, (fn, default, F) in wrappers.items():
        for trial, unsigned in itertools.product(range(5), (True, False)):
            target, stack = make(int(rng.integers(10, 20))), np.stack([make(4) for _ in range(F)])
            if name in ("alignment", "znf"):
                stack = np.abs(stack)
            kw = dict(label_unsigned=unsigned)
            results = {}
            for global_ic in (False, True):
                config.set_ic_scale(global_ic)
                results[global_ic] = (fn(target, stack, **kw), fn(target, stack, ic_scale=default, **kw),
                                      fn(target, stack, ic_scale=True, **kw),
                                      fn(motif_ic(target), motif_ic(stack), ic_scale=False, **kw))
                if config.get_ic_scale() != global_ic:
                    print(f"  [FAIL] {name}: global IC-scale setting not restored")
                    n_fail += 1
            checks = [("default", results[False][0], results[False][1]),
                      ("IC-scaled once", results[False][2], results[False][3]),
                      ("global setting adds no IC-scaling (default)", results[False][0], results[True][0]),
                      ("global setting adds no IC-scaling (ic_scale)", results[False][2], results[True][2]),
                      ("global setting adds no IC-scaling (no ic_scale)", results[False][3], results[True][3])]
            for label, a, b in checks:
                diffs = differing(a, b)
                if diffs:
                    print(f"  [FAIL] {name} unsigned={unsigned} trial {trial}: {label}: {diffs}")
                    n_fail += 1
    config.set_ic_scale(setting)
    print(f"k. ic_scale: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def pin_ic_scale_off(new):
    '''The tests below check the alignment against composite.py versions without IC-scaling: pin
    the composite wrappers to ic_scale=False (test_ic_scale() checks ic_scale). They were written for
    the overlap penalty on the motif energy: pin overlap_penalty_mode="energy" too (the masked penalty,
    the default, is checked in test_masked_overlap()); a test passing it explicitly overrides the pin.'''
    for name in ("run_modified_bellman_ford_composite",):
        setattr(new, name, functools.partial(getattr(new, name), ic_scale=False, overlap_penalty_mode="energy"))


if __name__ == "__main__":
    new = load_module(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MotifCompendium", "utils", "composite.py"),
        "composite_new")
    failures = test_ic_scale(new)
    pin_ic_scale_off(new)
    if len(sys.argv) > 1:
        orig = load_module(sys.argv[1], "composite_orig")
        failures += test_core_identical(new, orig)
    if len(sys.argv) > 3:
        prev = load_module(sys.argv[3], "composite_prev")
        failures += test_matches_previous(new, prev)
    if len(sys.argv) > 4:
        base = load_module(sys.argv[4], "composite_base")
        failures += test_matches_base(new, base)
    failures += test_penalty_signs(new)
    if len(sys.argv) > 2:
        snap = load_module(sys.argv[2], "composite_snap")
        failures += test_matches_snapshot(new, snap)
    failures += test_degenerate(new)
    failures += test_new_params_noop(new)
    failures += test_exact(new)
    failures += test_one_state(new)
    failures += test_match_count(new)
    failures += test_match_cost(new)
    failures += test_match_count_max(new)
    failures += test_signed(new)
    failures += test_composite_varlen(new)
    failures += test_composite_score(new)
    failures += test_pooled_search(new)
    failures += test_masked_overlap(new)
    failures += test_storage(new)
    failures += test_l2_norm_removed(new)
    failures += test_padding(new)
    failures += test_general_m(new)
    failures += test_alignment_tracks(new)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(1 if failures else 0)
