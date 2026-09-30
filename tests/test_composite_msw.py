"""
Regression + feature checks for MotifCompendium.utils.composite.

Run:
    python tests/test_composite_msw.py ORIGINAL_COMPOSITE [SNAPSHOT_COMPOSITE]

ORIGINAL_COMPOSITE is the pre-generalization composite.py (old API). It gates that the
frozen 3 bp core is still the original algorithm.

SNAPSHOT_COMPOSITE is a composite.py from before the exact state DP and the target padding
(one arrangement per cell, current API otherwise). Where the scoring did not change (zero
penalties, general_penalty_f off), results are checked bit for bit against it, feeding it the
target padded as the current code pads it.
"""

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
    overlap_penalty_f1=-0.02,
    overlap_penalty_f2=-0.1,
    skip_penalty_f1=-0.1,
    skip_penalty_f2=-0.1,
    skip_penalty_l1=-0.02,
    skip_penalty_l2=-0.1,
)
ZERO = dict(NEUTRAL, overlap_penalty_f1=0.0, overlap_penalty_f2=0.0, skip_penalty_f1=0.0,
            skip_penalty_f2=0.0, skip_penalty_l1=0.0, skip_penalty_l2=0.0)

OUTPUTS = ["msw_score", "sim_score", "aligned_motif_fs", "msw_orient",
           "msw_align_f", "msw_align_l", "msw_scale_f", "msw_actions"]


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
    return mod._run_modified_smith_waterman(
        target_motif=motif, subunit_motif_stack=unroll(fingers, M), **params)


def call_core(mod, sim, scale, M, **kwargs):
    """The DP core alone, on given similarities (one orientation)."""
    F, N = sim.shape
    params = dict(NEUTRAL, orientations=("for",))
    params.update(kwargs)
    return mod.calculate_modified_smith_waterman(
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


def chain_score(chain, sim, scale, M, p, general):
    """Score of a chain of matches (finger, motif position), by the documented rules."""
    raw = sum(scale[w] * sim[f, w] for f, w in chain)
    penalty = 0.0
    for (fa, wa), (fb, wb) in zip(chain, chain[1:]):
        d, g = wb - wa, fb - fa - 1
        if d < M:  # Overlap of M - d bp
            penalty += p["overlap_penalty_f1"] + (M - d - 1) * p["overlap_penalty_f2"]
        elif d == M + 1:  # Gap of 1 bp
            penalty += p["skip_penalty_l1"]
        elif d > M + 1:  # Gap of d - M bp
            penalty += (d - M - 1) * p["skip_penalty_l2"]
        if g >= 1:  # Run of g skipped fingers
            penalty += p["skip_penalty_f1"] + (g - 1) * p["skip_penalty_f2"]
    return (raw / math.sqrt(len(chain)) if general else raw) + penalty


def match_chain(actions):
    return [(int(a.split("@")[0][1:]), int(a.split("@")[-1][1:])) for a in actions if "@match@" in a]


# ------------------------------------------------------------------ sections
def test_core_identical(new, orig, n_cases=300, seed=0):
    """0. The frozen 3 bp core must match the original core exactly, given identical inputs."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        stack, motif_norm, sim_for, sim_rev, importance = prepare(new, motif, fingers, 3)
        shared = dict(matrix_f=stack, matrix_l=motif_norm, sim_for=sim_for, sim_rev=sim_rev,
                      scale_factor=importance, sim_threshold=0.0,
                      overlap_penalty_f1=NEUTRAL["overlap_penalty_f1"],
                      overlap_penalty_f2=NEUTRAL["overlap_penalty_f2"],
                      skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
                      skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
                      alignment_type="local", initial_score=0.0)
        out_orig = orig.calculate_modified_smith_waterman(
            skip_penalty_f=NEUTRAL["skip_penalty_f1"], **shared)
        out_new = new.calculate_modified_smith_waterman_znf(
            skip_penalty_f1=NEUTRAL["skip_penalty_f1"], skip_penalty_f2=NEUTRAL["skip_penalty_f2"], **shared)
        if tuple(out_orig[:3]) != tuple(out_new[:3]):
            n_diff += 1
            if n_diff <= 3:
                print(f"  [diff] case {i} F={F} L={L}: {out_orig[0]!r} vs {out_new[0]!r}")
    print(f"0. frozen core vs original core (identical inputs): {n_cases - n_diff}/{n_cases} identical")
    return n_diff


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
        for config, general in itertools.product((NEUTRAL, ZERO), (False, True)):
            try:
                out = call_znf(new, motif, fingers, general_penalty_f=general, **config)
                assert not np.isnan(out[0]) and not np.isnan(out[1]), "NaN score"
                new.run_modified_smith_waterman_composite(motif, unroll(fingers, 3), general_penalty_f=general, **config)
            except Exception as exc:
                print(f"  [FAIL] {name} general={general}: {type(exc).__name__}: {exc}")
                n_fail += 1
    print(f"1b. degenerate cases: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_matches_snapshot(new, snap, n_cases=40, seed=4):
    """2. Where the scoring did not change (zero penalties, general_penalty_f off), every path
    matches the snapshot run on the padded target, bit for bit."""
    rng = np.random.default_rng(seed)
    n_diff = n_runs = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif3, fingers3 = make_case(rng, F, 3, int(rng.integers(15, 50)))
        motif4, fingers4 = make_case(rng, F, 4, int(rng.integers(20, 60)))
        motif2, fingers2 = make_case(rng, F, 2, int(rng.integers(10, 40)))
        runs = [
            ("znf  M=3", 3, motif3, lambda mod, m: call_znf(mod, m, fingers3, **ZERO)),
            ("gen  M=3", 3, motif3, lambda mod, m: call_general(mod, m, fingers3, 3, **ZERO)),
            ("gen  M=4", 4, motif4, lambda mod, m: call_general(mod, m, fingers4, 4, **ZERO)),
            ("gen  M=2 local only", 2, motif2,
             lambda mod, m: call_general(mod, m, fingers2, 2, align_remaining=False, **ZERO)),
        ]
        for label, M, motif, run in runs:
            n_runs += 1
            diffs = differing_from_snapshot(run(new, motif), run(snap, pad(motif, M)), len(motif) + 2 * M - 2)
            if diffs:
                n_diff += 1
                if n_diff <= 3:
                    print(f"  [diff] {label}: {diffs}")

        # The composite: the snapshot's frame is M-1 wider on the left
        stack = rng.random((int(rng.integers(1, 3)), 3, 4)) ** 2 + 0.01
        motif = rng.random((int(rng.integers(12, 30)), 4))
        a = new.run_modified_smith_waterman_composite(motif, stack, general_penalty_f=False, **ZERO)
        b = snap.run_modified_smith_waterman_composite(pad(motif, 3), stack, **ZERO)
        n_runs += 1
        if not (a[0] == b[0] and abs(a[1] - b[1]) < 1e-12 and np.array_equal(a[3], b[3])
                and np.array_equal(a[4], b[4]) and np.array_equal(a[5] + 2, b[5]) and np.array_equal(a[6], b[6])
                and np.array_equal(a[2], b[2][:, 2:2 + len(motif) + 4])):
            n_diff += 1
            print("  [diff] composite")
    print(f"2. vs snapshot on the padded target (zero penalties, general_penalty_f off), 5 paths: "
          f"{n_runs - n_diff}/{n_runs} identical")
    return n_diff


def test_exact(new, n_cases=90, seed=11):
    """a. The DP is exact: its score is the best over every chain of matches (brute force), and
    its matches re-score to it, for every penalty configuration, with and without
    general_penalty_f. Its state at the optimum decomposes the score."""
    configs = {
        "zero": ZERO,
        "neutral": NEUTRAL,
        "harsh f2": dict(NEUTRAL, skip_penalty_f2=-0.3),
        "overlap": dict(ZERO, overlap_penalty_f1=-0.05, overlap_penalty_f2=-0.2),
        "gap": dict(ZERO, skip_penalty_l1=-0.05, skip_penalty_l2=-0.15),
        "l1 only": dict(ZERO, skip_penalty_l1=-0.1),
        "f-skip": dict(ZERO, skip_penalty_f1=-0.05, skip_penalty_f2=-0.3),
        "threshold": dict(NEUTRAL, sim_threshold=0.5),
    }
    rng = np.random.default_rng(seed)
    n_bad = n_runs = 0
    for i in range(n_cases):
        M = (2, 3, 4)[i % 3]
        F, N = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        sim = rng.random((F, N)) * 1.6 - 0.4
        scale = rng.random(N) * 0.5
        for (name, p), general in itertools.product(configs.items(), (False, True)):
            best = 0.0  # The empty alignment
            for r in range(1, min(F, N) + 1):
                for rows in itertools.combinations(range(F), r):
                    for cols in itertools.combinations(range(N), r):
                        chain = list(zip(rows, cols))
                        if all(sim[f, w] > p["sim_threshold"] for f, w in chain):
                            best = max(best, chain_score(chain, sim, scale, M, p, general))
            out = call_core(new, sim, scale, M, general_penalty_f=general, return_matrices=True, **dict(p))
            chain = match_chain(out[2])
            state = out[3]
            rescale = 1 / math.sqrt(state["F_total"]) if general and state["F_total"] > 0 else 1.0
            problems = []
            if abs(out[0] - best) > 1e-12:
                problems.append(f"score {out[0]!r}, best chain {best!r}")
            if chain and abs(chain_score(chain, sim, scale, M, p, general) - out[0]) > 1e-12:
                problems.append("matches do not re-score to msw_score")
            if state["F_total"] != len(chain):
                problems.append(f"F_total {state['F_total']} for {len(chain)} matches")
            if out[0] != (state["H_match"] * rescale + state["H_penalty_overlap_f"]
                          + state["H_penalty_skip_f"] + state["H_penalty_skip_l"]):
                problems.append("H != H_match*r + penalties")
            if out[4].shape != (F + 1, N + 2 * M - 1) or out[4].max() != out[0]:
                problems.append(f"H_final {out[4].shape}, max {out[4].max()!r}")
            n_runs += 1
            if problems:
                n_bad += 1
                if n_bad <= 5:
                    print(f"  [FAIL] {name} general={general} M={M} F={F} N={N}: {problems}")
    print(f"a. exact DP: {n_runs - n_bad}/{n_runs} optimal (brute force), re-scored and decomposed, "
          f"{len(configs)} penalty configurations x general_penalty_f")
    return n_bad


def test_general_penalty(new, seed=12):
    """d. general_penalty_f by hand, the case a one-state DP gets wrong, and 'l2_norm' removed."""
    rng = np.random.default_rng(seed)
    n_bad = 0
    # Two planted one-hot fingers: (m1 + m2)/sqrt(2), and every finger scaled by 1/sqrt(2)
    for M in (2, 3, 4, 5):
        fingers, motif = plant_one_hot(rng, 2, M)
        L = len(motif)
        out = call_general(new, motif, fingers, M, general_penalty_f=True, orientations=("for",), **ZERO)
        if (not np.isclose(out[0], 2 * (M / L) / math.sqrt(2), rtol=0, atol=1e-14)
                or not np.allclose(out[6], M / L / math.sqrt(2), rtol=0, atol=1e-14)
                or not np.isclose(out[1], 1.0)
                or match_chain(out[7]) != [(0, M - 1), (1, 2 * M - 1)]):
            n_bad += 1
            print(f"  [FAIL] 2 planted fingers M={M}: score {out[0]}, scales {out[6]}, sim {out[1]}, {out[7]}")

    # At finger 1's cell, the prefix (0.42 + 1)/sqrt(2) = 1.004 beats 1/sqrt(1) = 1.0, but the
    # best alignment continues the other one: 2/sqrt(2) = 1.414 > 2.42/sqrt(3) = 1.397
    sim = np.full((3, 6), -1.0)
    sim[0, 0], sim[1, 2], sim[2, 4] = 0.42, 1.0, 1.0
    out = call_core(new, sim, np.ones(6), 2, general_penalty_f=True, **ZERO)
    if abs(out[0] - math.sqrt(2)) > 1e-15 or match_chain(out[2]) != [(1, 2), (2, 4)]:
        n_bad += 1
        print(f"  [FAIL] one-state trap: {out[0]!r} {out[2]}")

    # 'l2_norm' is gone
    motif, fingers = make_case(rng, 3, 3, 20)
    for spelling in ("l2_norm", "L2_NORM"):
        for label, run in (
                ("core", lambda: call_core(new, np.ones((2, 4)), np.ones(4), 2, overlap_penalty_f1=spelling)),
                ("alignment", lambda: call_znf(new, motif, fingers, overlap_penalty_f2=spelling)),
                ("composite", lambda: new.run_modified_smith_waterman_composite(
                    motif, unroll(fingers, 3), overlap_penalty_f1=spelling))):
            try:
                run()
            except ValueError:
                continue
            n_bad += 1
            print(f"  [FAIL] {label} accepted {spelling!r}")
    print(f"d. general_penalty_f: hand cases, the one-state trap, 'l2_norm' rejected: "
          f"{'ok' if n_bad == 0 else f'{n_bad} failures'}")
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
            # has cosine 1/sqrt(M), and every other window 0
            target = np.concatenate([s[M - 1:], np.eye(4)[[3] * middle], s[:1]])
            L = len(target)
            out = new.run_modified_smith_waterman_composite(
                target, s[None], sim_threshold=0.5 / math.sqrt(M), max_homo_subunits=2,
                allow_reverse_order=False, allow_reverse_comp=False, general_penalty_f=False, **{
                    k: v for k, v in ZERO.items() if k != "sim_threshold"})
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
        out = new.run_modified_smith_waterman_composite(rng.random((L, 4)), rng.random((2, M, 4)))
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
    modes = [dict(), ZERO, dict(ZERO, general_penalty_f=True)]
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
        same = call_znf(new, motif, fingers, skip_penalty_f1=-0.1, skip_penalty_f2=-0.1)
        if differing(base, same):
            n_diff += 1
        harsh = call_znf(new, motif, fingers, skip_penalty_f2=-0.3)
        if harsh[0] != base[0]:
            n_changed += 1
    print(f"3. finger-skip params: {n_cases - n_diff}/{n_cases} neutral-identical; "
          f"{n_changed}/{n_cases} changed under skip_f2=-0.3")
    return n_diff


def rc_of(motif):
    return motif[::-1, ::-1]


def expand(stack, copies, reverse_order, reverse_comp):
    """The composite wrapper's expanded stack, entry by entry: (stack, subunit, orientation)."""
    F = len(stack)
    orders = [list(range(F))] + ([list(range(F))[::-1]] if reverse_order else [])
    entries = [(s, rc) for _ in range(copies) for order in orders
               for rc in ((0, 1) if reverse_comp else (0,)) for s in order]
    expanded = np.stack([rc_of(stack[s]) if rc else stack[s] for s, rc in entries])
    return expanded, np.array([s for s, _ in entries]), np.array([rc for _, rc in entries])


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


def run_composite(new, target, subunits, **kwargs):
    """The composite wrapper's outputs, and the orientation of the core's winning pass."""
    from unittest.mock import patch
    core, cores = new._run_modified_smith_waterman, []

    def record(**core_kwargs):
        cores.append(core(**core_kwargs))
        return cores[-1]

    with patch.object(new, "_run_modified_smith_waterman", side_effect=record):
        out = new.run_modified_smith_waterman_composite(target, subunits, **kwargs)
    return out, cores[-1][3]


def planted_problems(new, subunits, target, expected, **kwargs):
    """Ways the composite wrapper fails to report exactly the expected (subunit, rc, start) matches."""
    F, M, _ = subunits.shape
    L = len(target)
    out, orient = run_composite(new, target, subunits, sim_threshold=0.99, **kwargs)
    if len(out) != 7:
        return [f"{len(out)} outputs"], orient
    (_, sim, stack, idx, rc, h, scale) = out
    reported = [(int(s), int(r), int(x) - (M - 1)) for s, r, x in zip(idx, rc, h)]
    if reported != list(expected):
        return [f"matched {reported}"], orient
    problems = []
    if not idx.dtype.kind == rc.dtype.kind == h.dtype.kind == "i":
        problems.append("subunit indices / orientations / positions are not integers")
    rescale = 1 / math.sqrt(len(expected)) if kwargs.get("general_penalty_f", True) and expected else 1.0
    if not np.allclose(scale, M / L * rescale, rtol=0, atol=1e-14):
        problems.append(f"scale {scale}")
    placed = np.zeros((len(expected), 2 * M + L - 2, 4))
    for n, (s, r, start) in enumerate(expected):
        placed[n, start + M - 1:start + 2 * M - 1] = (rc_of(subunits[s]) if r else subunits[s]) / math.sqrt(M) * M / L * rescale
    if stack.shape != placed.shape or not np.allclose(stack, placed, rtol=0, atol=1e-14):
        problems.append("aligned subunits")
    if len(expected) * M == L and not np.isclose(sim, 1.0):
        problems.append(f"complete reconstruction has sim_score {sim}")
    return problems, orient


def test_subunit_expansion(new, seed=8):
    """8. The composite wrapper's expanded stack, and planted matches reported by input subunit."""
    from unittest.mock import patch

    class Captured(Exception):
        pass

    stack = np.arange(24).reshape(2, 3, 4)
    saved = stack.copy()
    for copies in (1, 2, np.int64(3)):
        for reverse_order, reverse_comp in itertools.product((False, True), repeat=2):
            with patch.object(new, "_run_modified_smith_waterman", side_effect=Captured) as run:
                try:
                    new.run_modified_smith_waterman_composite(
                        np.ones((10, 4)), stack, max_homo_subunits=copies,
                        allow_reverse_order=reverse_order, allow_reverse_comp=reverse_comp)
                except Captured:
                    pass
                kwargs = run.call_args.kwargs
            orders = [stack] + ([stack[::-1]] if reverse_order else [])
            per_copy = [block for order in orders
                        for block in ((order, order[:, ::-1, ::-1]) if reverse_comp else (order,))]
            expanded = kwargs["subunit_motif_stack"]
            assert expanded.shape == (2 ** (reverse_order + reverse_comp) * copies * 2, 3, 4)
            np.testing.assert_array_equal(expanded, np.concatenate(per_copy * int(copies)))
            assert kwargs["orientations"] == (("for", "rev") if reverse_comp else ("for",))
            assert kwargs["place_as_matched"] is True
            assert kwargs["align_remaining"] is False
            assert kwargs["general_penalty_f"] is True
    np.testing.assert_array_equal(stack, saved)
    for copies in (0, -1, 1.5, True, "2"):
        try:
            new.run_modified_smith_waterman_composite(np.ones((10, 4)), stack, max_homo_subunits=copies)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid max_homo_subunits {copies!r}")

    n_bad = n_runs = 0
    orients = set()

    def check(label, subunits, target, expected, **kwargs):
        nonlocal n_bad, n_runs
        problems, orient = planted_problems(new, subunits, target, expected, **kwargs)
        n_runs += 1
        orients.add(orient)
        if problems:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [FAIL] {label} {kwargs}: {problems[:2]}")

    # One subunit, in the order given: copy limits, and both orders of mixed orientations with
    # one copy ([rc, forward] is only reachable with the orientations swapped: the "rev" pass)
    a = np.eye(4)[[0, 0, 0, 1]]  # AAAC
    for copies, planted, expected in (
            (1, [0, 0], [0]),
            (2, [0, 0], [0, 0]),
            (1, [0, 1], [0, 1]),
            (1, [1, 0], [1, 0]),
            (2, [0, 1, 0], [0, 1, 0]),
            (3, [0, 0, 0], [0, 0, 0])):
        target = np.concatenate([rc_of(a) if r else a for r in planted])
        for general in (True, False):
            check(f"planted={planted}", a[None], target, [(0, r, 4 * i) for i, r in enumerate(expected)],
                  max_homo_subunits=copies, allow_reverse_order=False, general_penalty_f=general)
    # Without reverse complements: Only the forward matches
    check("no reverse complements", a[None], np.concatenate([a, rc_of(a), a]), [(0, 0, 0), (0, 0, 8)],
          max_homo_subunits=2, allow_reverse_order=False, allow_reverse_comp=False)
    # One subunit reversed is the same block again: One more match per copy
    check("one subunit, reversed", a[None], np.concatenate([a, a]), [(0, 0, 0), (0, 0, 4)],
          max_homo_subunits=1, allow_reverse_comp=False)

    # Two subunits: Reverse order, and reverse order + reverse complement (the whole chain's
    # reverse complement), each with one copy
    b = np.eye(4)[[2, 1, 3, 3]]  # GCTT
    pair = np.stack([a, b])
    check("reverse order", pair, np.concatenate([b, a]), [(1, 0, 0), (0, 0, 4)],
          max_homo_subunits=1, allow_reverse_comp=False)
    check("reverse order + complement", pair, np.concatenate([rc_of(b), rc_of(a)]), [(1, 1, 0), (0, 1, 4)],
          max_homo_subunits=1)

    # Several subunits, in random order and orientations, with enough copies to match them all
    rng = np.random.default_rng(seed)
    for i in range(24):
        F, M, n = int(rng.integers(1, 4)), (3, 4, 5)[i % 3], int(rng.integers(1, 5))
        subunits, target, planted = plant_subunits(rng, F, M, n)
        expected = [(s, r, M * j) for j, (s, r) in enumerate(planted)]
        for copies in (n, n + 1):
            check(f"F={F} M={M} planted={planted}", subunits, target, expected, max_homo_subunits=copies)
        check(f"F={F} M={M} planted={planted}", subunits, target, [e for e in expected if e[1] == 0],
              max_homo_subunits=n, allow_reverse_comp=False)
    print(f"8. composite expansion: stack layouts ok; {n_runs - n_bad}/{n_runs} planted alignments reported by "
          f"input subunit, orientation, position and scale (winning passes: {sorted(orients)})")
    return n_bad


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


def cosine(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def input_subunit_problems(out, ref, motif, stack, stack_subunit, stack_rc, M):
    """Ways out (the composite wrapper) is not ref (the core, on the expanded stack), by input subunit."""
    if len(out) != 7:
        return [f"{len(out)} outputs"]
    (score, sim, aligned, idx, rc, h, scale) = out
    L = len(motif)
    orient = ref[3]
    problems = []
    if (score, sim) != (ref[0], ref[1]):
        problems.append("score / sim_score")
    matches = match_chain(ref[7])
    if (list(idx) != [stack_subunit[i] for i, _ in matches]
            or list(rc) != [stack_rc[i] ^ (orient == "rev") for i, _ in matches]
            or list(h) != [l for _, l in matches]  # Padded positions
            or list(scale) != [ref[6][i] for i, _ in matches]):
        return problems + ["subunit index / orientation / position / scale"]
    # Orientation: The subunit as reported has the similarity that the alignment matched on
    target, subunits = pad(np.abs(motif), M), np.abs(stack)
    for n, (i, l) in enumerate(matches):
        window = target[l:l + M]
        reported = rc_of(subunits[idx[n]]) if rc[n] else subunits[idx[n]]
        entry = rc_of(subunits[stack_subunit[i]]) if stack_rc[i] else subunits[stack_subunit[i]]
        matched = cosine(entry, window if orient == "for" else rc_of(window))
        if abs(cosine(reported, window) - matched) > 1e-12:
            problems.append(f"orientation of match {n}")
    placed = np.zeros((len(matches), 2 * M + L - 2, 4))
    for n in range(len(matches)):
        subunit = subunits[idx[n]] / np.linalg.norm(subunits[idx[n]])
        placed[n, h[n]:h[n] + M] = (rc_of(subunit) if rc[n] else subunit) * scale[n]
    if aligned.shape != placed.shape or not np.allclose(aligned, placed, rtol=0, atol=1e-14):
        problems.append("aligned subunits")
    if not np.allclose(aligned.sum(axis=0), ref[2].sum(axis=0), rtol=0, atol=1e-14):
        problems.append("composite differs from the core's")
    return problems


def test_input_subunit_outputs(new, n_cases=48, seed=10):
    """9. Random cases: the composite wrapper's outputs are the core's on the expanded stack, by input subunit."""
    rng = np.random.default_rng(seed)
    configs = [(ZERO, True), (ZERO, False), (NEUTRAL, False), (NEUTRAL, True)]
    flags = list(itertools.product((False, True), repeat=2))
    n_bad = n_runs = n_rev = n_rc = n_repeat = n_sim_up = n_overhang = 0
    for i in range(n_cases):
        M = (3, 4, 2, 5)[i % 4]
        reverse_order, reverse_comp = flags[(i // 4) % 4]
        F, copies = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        motif, stack = make_mixed_case(rng, F, M, int(rng.integers(max(M + 1, 12), 40)))
        expanded, stack_subunit, stack_rc = expand(stack, copies, reverse_order, reverse_comp)
        for cfg, general in configs[:3] if i % 3 else configs:  # Penalties + general_penalty_f: 1 in 3
            out = new.run_modified_smith_waterman_composite(
                motif, stack, max_homo_subunits=copies, general_penalty_f=general,
                allow_reverse_order=reverse_order, allow_reverse_comp=reverse_comp, **cfg)
            core = dict(target_motif=motif, subunit_motif_stack=expanded, return_matrices=False, label_unsigned=True,
                        orientations=("for", "rev") if reverse_comp else ("for",), align_remaining=False,
                        general_penalty_f=general, **cfg)
            ref = new._run_modified_smith_waterman(place_as_matched=True, **core)
            problems = input_subunit_problems(out, ref, motif, stack, stack_subunit, stack_rc, M)
            n_runs += 1
            n_rc += bool(np.any(out[4]))
            n_repeat += len(set(out[3].tolist())) < len(out[3])
            n_overhang += bool(np.any(out[5] < M - 1) or np.any(out[5] > len(motif) - 1))
            if ref[3] == "rev":
                n_rev += 1
                n_sim_up += bool(out[1] > new._run_modified_smith_waterman(place_as_matched=False, **core)[1])
            if problems:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  [FAIL] M={M} F={F} copies={copies} reverse order/comp={reverse_order}/{reverse_comp} "
                          f"general={general} {cfg}: {problems[:2]}")
    print(f"9. composite outputs: {n_runs - n_bad}/{n_runs} match the core on the expanded stack, over all 4 "
          f"reverse order / complement settings ({n_rc} with reverse complements, {n_repeat} with repeated "
          f"subunits, {n_overhang} overhanging an end, {n_rev} won by 'rev'; placing 'rev' matches as matched "
          f"raised sim_score in {n_sim_up}/{n_rev})")
    return n_bad


def test_composite_no_remaining(new, n_cases=20, seed=13):
    """10. The composite does not force its leftover stack entries onto the rest of the motif."""
    rng = np.random.default_rng(seed)
    cfg = dict(NEUTRAL, sim_threshold=0.8)  # A threshold that can stop the local alignment early
    n_bad = n_forced = 0
    for i in range(n_cases):
        M = (3, 4, 5, 6)[i % 4]
        F = int(rng.integers(1, 4))
        motif, stack = make_mixed_case(rng, F, M, int(rng.integers(max(M + 1, 12), 40)))
        far = stack[int(rng.integers(F))] * (0.5 + rng.random())  # A subunit past a noise gap
        far = rc_of(far) if rng.random() < 0.5 else far
        gap = rng.random((int(rng.integers(3, 16)), 4)) * 0.05
        motif = np.concatenate([motif, gap, far, rng.random((2, 4)) * 0.05])
        expanded, stack_subunit, stack_rc = expand(stack, 2, True, True)  # The composite's defaults
        out = new.run_modified_smith_waterman_composite(motif, stack, **cfg)
        core = dict(target_motif=motif, subunit_motif_stack=expanded, return_matrices=False, label_unsigned=True,
                    place_as_matched=True, general_penalty_f=True, **cfg)
        local = new._run_modified_smith_waterman(align_remaining=False, **core)
        forced = new._run_modified_smith_waterman(align_remaining=True, **core)
        n_forced += sum("@match@" in a for a in forced[7]) > sum("@match@" in a for a in local[7])
        if input_subunit_problems(out, local, motif, stack, stack_subunit, stack_rc, M):
            n_bad += 1
    print(f"10. composite without the remaining alignment: {n_cases - n_bad}/{n_cases} are the local alignment "
          f"alone (the remaining alignment would have added matches in {n_forced})")
    return n_bad


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
        for general in (False, True):
            aligned = new.run_modified_smith_waterman_alignment(target_motif=motif, reference_motif=fingers,
                                                                finger_length=M, general_penalty_f=general, **NEUTRAL)
            for label, other in (("generalized path", call_general(new, motif, fingers, M, general_penalty_f=general)),
                                 ("ZNF wrapper", call_znf(new, motif, fingers, finger_length=M,
                                                          general_penalty_f=general))):
                diffs = differing(aligned, other)
                if diffs:
                    print(f"  [FAIL] alignment wrapper at M={M} differs from the {label}: {diffs}")
                    n_fail += 1
    print(f"5. general M: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


if __name__ == "__main__":
    new = load_module(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MotifCompendium", "utils", "composite.py"),
        "composite_new")
    failures = 0
    if len(sys.argv) > 1:
        orig = load_module(sys.argv[1], "composite_orig")
        failures += test_core_identical(new, orig)
    if len(sys.argv) > 2:
        snap = load_module(sys.argv[2], "composite_snap")
        failures += test_matches_snapshot(new, snap)
    failures += test_degenerate(new)
    failures += test_new_params_noop(new)
    failures += test_exact(new)
    failures += test_general_penalty(new)
    failures += test_padding(new)
    failures += test_subunit_expansion(new)
    failures += test_input_subunit_outputs(new)
    failures += test_composite_no_remaining(new)
    failures += test_general_m(new)
    failures += test_alignment_tracks(new)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(1 if failures else 0)
