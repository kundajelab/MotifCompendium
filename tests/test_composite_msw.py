"""
Regression + feature checks for MotifCompendium.utils.composite.

Run:
    python tests/test_composite_msw.py ORIGINAL_COMPOSITE [SNAPSHOT_COMPOSITE]

ORIGINAL_COMPOSITE is the pre-generalization composite.py (old API). It gates that the
frozen 3 bp core is still the original algorithm.

SNAPSHOT_COMPOSITE is a later composite.py with the current API. Changes that must not
alter results (short-circuits, output-neutral fixes) are checked bit for bit against it.
"""

import importlib.util
import itertools
import math
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


NEUTRAL = dict(
    sim_threshold=0.0,
    overlap_penalty_l1=-0.02,
    overlap_penalty_l2=-0.1,
    overlap_penalty_f1=1.0,
    overlap_penalty_f2=1.0,
    skip_penalty_f1=-0.1,
    skip_penalty_f2=-0.1,
    skip_penalty_l1=-0.02,
    skip_penalty_l2=-0.1,
)

OUTPUTS = ["msw_score", "sim_score", "aligned_motif_fs", "msw_orient",
           "msw_align_f", "msw_align_l", "msw_scale_f", "msw_actions"]
# The alignment tracks were rebuilt after the snapshot, deliberately; see test_alignment_tracks
SNAPSHOT_FIELDS = [name for name in OUTPUTS if name not in ("msw_align_f", "msw_align_l")]


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


def call_znf(mod, motif, fingers, **kwargs):
    params = dict(NEUTRAL)
    params.update(kwargs)
    return mod.run_modified_smith_waterman_znf(cwm=motif, b1h_ppm_ic=fingers, finger_length=3, **params)


def call_general(mod, motif, fingers, M, **kwargs):
    params = dict(NEUTRAL)
    params.update(kwargs)
    return mod.run_modified_smith_waterman(
        target_motif=motif, subunit_motif_stack=mod.unroll_fingers(fingers, M), **params)


def call_original(mod, motif, fingers):
    return mod.run_modified_smith_waterman_znf(
        cwm=motif, b1h_ppm_ic=fingers,
        sim_threshold=NEUTRAL["sim_threshold"],
        overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
        overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
        skip_penalty_f=NEUTRAL["skip_penalty_f1"],
        skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
        skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
        finger_length=3,
    )


def prepare(mod, motif, fingers, M):
    """The current preprocessing, so two DP cores can be fed byte-identical inputs."""
    import cupy as cp
    motif = np.abs(motif)
    motif_norm = motif / np.linalg.norm(motif, axis=(0, 1), keepdims=True)
    unrolled = mod._unroll_motif(motif_norm, M, np)
    unrolled_cp = mod._unroll_motif(cp.asarray(motif_norm), M, cp)
    importance = (unrolled * unrolled).sum(axis=(1, 2))
    stack = mod.unroll_fingers(np.abs(fingers), M)
    norms = np.linalg.norm(stack, axis=(1, 2), keepdims=True)
    stack_norm = stack / np.where(norms == 0, 1, norms)
    stack_cp = cp.asarray(stack_norm)
    sim_for = mod.compute_aligned_similarity(stack_cp, unrolled_cp)
    sim_rev = mod.compute_aligned_similarity(stack_cp, unrolled_cp[:, ::-1, ::-1])
    return stack_norm, motif_norm, sim_for, sim_rev, importance


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
                      overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
                      overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
                      skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
                      skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
                      alignment_type="local", initial_score=0.0)
        out_orig = orig.calculate_modified_smith_waterman(
            skip_penalty_f=NEUTRAL["skip_penalty_f1"], **shared)
        out_new = new.calculate_modified_smith_waterman_znf(
            skip_penalty_f1=NEUTRAL["skip_penalty_f1"], skip_penalty_f2=NEUTRAL["skip_penalty_f2"],
            overlap_penalty_f1=1.0, overlap_penalty_f2=1.0, **shared)
        if tuple(out_orig[:3]) != tuple(out_new[:3]):
            n_diff += 1
            if n_diff <= 3:
                print(f"  [diff] case {i} F={F} L={L}: {out_orig[0]!r} vs {out_new[0]!r}")
    print(f"0. frozen core vs original core (identical inputs): {n_cases - n_diff}/{n_cases} identical")
    return n_diff


def test_full_run_vs_original(new, orig, n_cases=300, seed=0):
    """1. End to end vs the original: score equal to 1e-12; action changes are reported."""
    rng = np.random.default_rng(seed)
    n_score = n_tail_only = n_other = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        out_new = call_znf(new, motif, fingers)
        out_orig = call_original(orig, motif, fingers)
        if abs(out_new[0] - out_orig[0]) > 1e-12:
            n_score += 1
        a_new, a_orig = out_new[7], out_orig[7]
        if a_new != a_orig:
            if a_new == a_orig[:len(a_new)]:
                n_tail_only += 1  # The remain-pass fix: garbage tail removed
            else:
                n_other += 1
                if n_other <= 3:
                    print(f"  [diff] case {i} F={F} L={L}\n    orig {a_orig}\n    new  {a_new}")
    print(f"1. full run vs original: score off by >1e-12 in {n_score}/{n_cases}; actions differ "
          f"only in the remain-pass tail in {n_tail_only}, otherwise in {n_other}")
    return n_score + n_other


def test_degenerate(new):
    """1b. Degenerate inputs must not crash."""
    rng = np.random.default_rng(7)
    zero_f = rng.random((9, 4)) + 0.01
    zero_f[3:6] = 0.0
    cases = [
        ("single finger", *make_case(rng, 1, 3, 12)),
        ("motif == finger", rng.random((3, 4)) + 0.01, rng.random((3, 4)) + 0.01),
        ("zero finger", rng.random((20, 4)) + 0.01, zero_f),
        ("pure noise", rng.random((30, 4)) * 1e-3, rng.random((12, 4)) + 0.01),
    ]
    overlaps = [NEUTRAL["overlap_penalty_l1"]] + (["gm"] if hasattr(new, "_parse_overlap_penalty") else [])
    n_fail = 0
    for name, motif, fingers in cases:
        for overlap in overlaps:
            try:
                with np.errstate(invalid="ignore", divide="ignore"):  # A zero finger has zero norm
                    call_znf(new, motif, fingers, overlap_penalty_l1=overlap)
            except Exception as exc:
                print(f"  [FAIL] {name} (l1={overlap!r}): {type(exc).__name__}: {exc}")
                n_fail += 1
    print(f"1b. degenerate cases: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_matches_snapshot(new, snap, n_cases=30, seed=4):
    """2. Every penalty configuration matches the snapshot bit for bit, on all three paths."""
    rng = np.random.default_rng(seed)
    nonzero = {
        "overlap": dict(overlap_penalty_l1=-0.02, overlap_penalty_l2=-0.1),
        "skip_l": dict(skip_penalty_l1=-0.02, skip_penalty_l2=-0.1),
        "skip_f": dict(skip_penalty_f1=-0.1, skip_penalty_f2=-0.3),
    }
    zero = {
        "overlap": dict(overlap_penalty_l1=0.0, overlap_penalty_l2=-0.0),
        "skip_l": dict(skip_penalty_l1=0.0, skip_penalty_l2=0.0),
        "skip_f": dict(skip_penalty_f1=-0.0, skip_penalty_f2=0.0),
    }
    configs = []
    for on in itertools.product((False, True), repeat=3):
        cfg = {}
        for key, active in zip(("overlap", "skip_l", "skip_f"), on):
            cfg.update(nonzero[key] if active else zero[key])
        configs.append(cfg)
    configs.append(dict(nonzero["overlap"], overlap_penalty_f1=0.0, overlap_penalty_f2=0.0))
    configs.append(dict(nonzero["overlap"], overlap_penalty_f1=1.0, overlap_penalty_f2=2.0))

    n_diff = n_runs = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif3, fingers3 = make_case(rng, F, 3, int(rng.integers(15, 50)))
        motif4, fingers4 = make_case(rng, F, 4, int(rng.integers(20, 60)))
        for cfg in configs:
            runs = [
                ("znf  M=3", call_znf(new, motif3, fingers3, **cfg), call_znf(snap, motif3, fingers3, **cfg)),
                ("gen  M=3", call_general(new, motif3, fingers3, 3, **cfg), call_general(snap, motif3, fingers3, 3, **cfg)),
                ("gen  M=4", call_general(new, motif4, fingers4, 4, **cfg), call_general(snap, motif4, fingers4, 4, **cfg)),
            ]
            for label, out_new, out_snap in runs:
                n_runs += 1
                diffs = differing(out_new, out_snap, fields=SNAPSHOT_FIELDS)
                if diffs:
                    n_diff += 1
                    if n_diff <= 3:
                        print(f"  [diff] {label} {cfg}: {diffs}")
    print(f"2. vs snapshot, {len(configs)} penalty configs x 3 paths: {n_runs - n_diff}/{n_runs} bit-identical "
          f"(all outputs but the rebuilt alignment tracks)")
    return n_diff


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
    modes = [dict(), {k: 0.0 for k in ("overlap_penalty_l1", "overlap_penalty_l2", "skip_penalty_l1",
                                         "skip_penalty_l2", "skip_penalty_f1", "skip_penalty_f2")},
             dict(overlap_penalty_l1="gm", overlap_penalty_l2="gm")]
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
            problems = track_violations(out[4], out[5], out[7], F, M, L)
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
    """3. The finger-count parameters are no-ops at neutral values."""
    rng = np.random.default_rng(seed)
    n_diff = n_changed = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif, fingers = make_case(rng, F, 3, int(rng.integers(15, 50)))
        base = call_znf(new, motif, fingers)
        same = call_znf(new, motif, fingers, overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
                        skip_penalty_f1=-0.1, skip_penalty_f2=-0.1)
        if differing(base, same):
            n_diff += 1
        harsh = call_znf(new, motif, fingers, overlap_penalty_f2=2.0, skip_penalty_f2=-0.3)
        if harsh[0] != base[0]:
            n_changed += 1
    print(f"3. finger-count params: {n_cases - n_diff}/{n_cases} neutral-identical; "
          f"{n_changed}/{n_cases} changed under f2=2.0 / skip_f2=-0.3")
    return n_diff


def test_geometric_mean(new):
    """4. 'gm' overlap penalty: hand-computed cases, parsing, and end-to-end runs."""
    n_fail = 0

    def check(label, got, expected):
        nonlocal n_fail
        if len(got) != len(expected) or not all(math.isclose(g, e, rel_tol=0, abs_tol=1e-15) for g, e in zip(got, expected)):
            print(f"  [FAIL] {label}: got {got}, expected {expected}")
            n_fail += 1

    M = 3
    row_norms = np.array([[0.1, 0.4, 0.2], [0.3, 0.5, 0.6], [0.7, 0.8, 0.9]])  # (F, M)
    scale = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    penalties = lambda f, l, l_j, prev, l1, l2, f1=1.0, f2=1.0, i_f=0, i_l=0: new._calculate_overlap_penalties(
        f, l, l_j, prev, M, l1, l2, f1, f2, row_norms, scale, i_f, i_l)

    # 1 bp overlap: finger 1 at l=4 over finger 0 at l=2, overlapping at p=4
    v_cur, v_prev = row_norms[1, 0] * scale[4], row_norms[0, 2] * scale[2]
    check("1 bp gm", penalties(1, 4, 2, ["F0@match@L2"], "gm", -0.1), [-math.sqrt(v_cur * v_prev)])
    check("1 bp gm x f1", penalties(1, 4, 2, ["F0@match@L2"], "gm", -0.1, f1=3.0), [-3.0 * math.sqrt(v_cur * v_prev)])

    # 2 bp overlap at p=3,4, mixed: l1 = gm, l2 fixed
    p3 = -math.sqrt(row_norms[1, 0] * scale[3] * row_norms[0, 1] * scale[2])
    check("2 bp gm + fixed", penalties(1, 3, 2, ["F0@match@L2"], "gm", -0.1), [p3, -0.1])
    p4 = -math.sqrt(row_norms[1, 1] * scale[3] * row_norms[0, 2] * scale[2])
    check("2 bp fixed + gm", penalties(1, 3, 2, ["F0@match@L2"], -0.02, "gm"), [-0.02, p4])

    # 3 fingers at p=4: finger 0 at 2, finger 1 at 3, finger 2 at 4
    prev = ["F1@match@L3", "F0@match@L2"]
    v = [row_norms[2, 0] * scale[4], row_norms[1, 1] * scale[3], row_norms[0, 2] * scale[2]]
    p4_three = -(v[0] * v[1] * v[2]) ** (1 / 3) * 5.0  # 3 fingers -> f2
    p5_two = -math.sqrt(row_norms[2, 1] * scale[4] * row_norms[1, 2] * scale[3]) * 2.0  # 2 fingers -> f1
    check("3-finger gm", penalties(2, 4, 3, prev, "gm", "gm", f1=2.0, f2=5.0), [p4_three, p5_two])

    # Absolute action coordinates in a remain pass map back to local ones
    check("offsets", penalties(1, 4, 2, ["F5@match@L12"], "gm", -0.1, i_f=5 - 0, i_l=10),
          [-math.sqrt(row_norms[1, 0] * scale[4] * row_norms[0, 2] * scale[2])])

    # Parsing
    for spelling in ("gm", "GM", "geometric_mean", "Geometric_Mean"):
        if new._parse_overlap_penalty(spelling, "x") != "gm":
            print(f"  [FAIL] parse {spelling!r}")
            n_fail += 1
    try:
        new._parse_overlap_penalty("foo", "overlap_penalty_l1")
        print("  [FAIL] 'foo' did not raise")
        n_fail += 1
    except ValueError:
        pass

    # End to end: both cores run, spellings agree, penalties never exceed 0
    rng = np.random.default_rng(9)
    n_changed = 0
    for _ in range(40):
        F = int(rng.integers(2, 9))
        motif, fingers = make_case(rng, F, 3, int(rng.integers(15, 50)))
        gm = call_znf(new, motif, fingers, overlap_penalty_l1="gm", overlap_penalty_l2="gm")
        spelled = call_znf(new, motif, fingers, overlap_penalty_l1="geometric_mean", overlap_penalty_l2="GM")
        if differing(gm, spelled):
            print("  [FAIL] 'geometric_mean' and 'gm' disagree")
            n_fail += 1
        call_general(new, motif, fingers, 3, overlap_penalty_l1="gm", overlap_penalty_l2=-0.1)
        if gm[0] != call_znf(new, motif, fingers)[0]:
            n_changed += 1
        off = call_znf(new, motif, fingers, overlap_penalty_l1="gm", overlap_penalty_l2="gm",
                       overlap_penalty_f1=0.0, overlap_penalty_f2=0.0)
        zero = call_znf(new, motif, fingers, overlap_penalty_l1=0.0, overlap_penalty_l2=0.0)
        if differing(off, zero):
            print("  [FAIL] gm with zero multipliers differs from no overlap penalty")
            n_fail += 1
    print(f"4. geometric mean: {'ok' if n_fail == 0 else f'{n_fail} failures'} "
          f"({n_changed}/40 end-to-end scores changed vs fixed penalties)")
    return n_fail


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
            if out[2].shape != (F, L + M - 1, 4):
                print(f"  [FAIL] M={M} F={F} L={L}: aligned shape {out[2].shape}")
                n_fail += 1

    for M in [2, 3, 4, 5, 7]:
        fingers, motif = plant_one_hot(rng, 4, M)
        matches = [a for a in call_general(new, motif, fingers, M)[7] if a.split("@")[1] == "match"]
        if matches != [f"F{f}@match@L{f * M}" for f in range(4)]:
            print(f"  [FAIL] planted adjacent M={M}: got {matches}")
            n_fail += 1

    for M in [2, 3, 4, 5]:
        fingers, _ = plant_one_hot(rng, 3, M)
        motif = np.concatenate([fingers[:2 * M], np.full((1, 4), 0.02), fingers[2 * M:]])
        matches = [a for a in call_general(new, motif, fingers, M)[7] if a.split("@")[1] == "match"]
        if matches != ["F0@match@L0", f"F1@match@L{M}", f"F2@match@L{2 * M + 1}"]:
            print(f"  [FAIL] planted gap M={M}: got {matches}")
            n_fail += 1

    # The ZNF wrapper forwards other finger lengths to the generalized core
    for M in [2, 4, 5]:
        motif, fingers = make_case(rng, 4, M, int(rng.integers(4 * M, 8 * M)))
        forwarded = new.run_modified_smith_waterman_znf(cwm=motif, b1h_ppm_ic=fingers, finger_length=M, **NEUTRAL)
        diffs = differing(forwarded, call_general(new, motif, fingers, M))
        if diffs:
            print(f"  [FAIL] ZNF wrapper at M={M} differs from the generalized path: {diffs}")
            n_fail += 1
    print(f"5. general M: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_general_vs_frozen(new, n_cases=100, seed=3):
    """6. At M=3 the generalized core mostly agrees with the frozen one (informational)."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        motif, fingers = make_case(rng, F, 3, int(rng.integers(15, 50)))
        if differing(call_znf(new, motif, fingers), call_general(new, motif, fingers, 3)):
            n_diff += 1
    print(f"6. generalized vs frozen at M=3: {n_cases - n_diff}/{n_cases} identical "
          f"({n_diff} differ, expected: the unified diagonal rule)")


if __name__ == "__main__":
    new = load_module(
        "/oak/stanford/groups/akundaje/cmyun/software/motifcompendium/MotifCompendium/utils/composite.py",
        "composite_new")
    failures = 0
    if len(sys.argv) > 1:
        orig = load_module(sys.argv[1], "composite_orig")
        failures += test_core_identical(new, orig)
        failures += test_full_run_vs_original(new, orig)
    if len(sys.argv) > 2:
        snap = load_module(sys.argv[2], "composite_snap")
        failures += test_matches_snapshot(new, snap)
    failures += test_degenerate(new)
    failures += test_new_params_noop(new)
    if hasattr(new, "_parse_overlap_penalty"):
        failures += test_geometric_mean(new)
    failures += test_general_m(new)
    if hasattr(new, "_alignment_tracks"):
        failures += test_alignment_tracks(new)
    test_general_vs_frozen(new)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(1 if failures else 0)
