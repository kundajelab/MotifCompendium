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
    overlap_penalty_f1=-0.02,
    overlap_penalty_f2=-0.1,
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


def call_frozen(mod, motif, fingers, **kwargs):
    """The frozen 3 bp core, end to end. No wrapper uses it any more, so swap it in for the generalized core."""
    from unittest.mock import patch
    with patch.object(mod, "calculate_modified_smith_waterman", mod.calculate_modified_smith_waterman_znf):
        return call_znf(mod, motif, fingers, **kwargs)


def call_original(mod, motif, fingers):
    return mod.run_modified_smith_waterman_znf(
        cwm=motif, b1h_ppm_ic=fingers,
        sim_threshold=NEUTRAL["sim_threshold"],
        overlap_penalty_f1=NEUTRAL["overlap_penalty_f1"],
        overlap_penalty_f2=NEUTRAL["overlap_penalty_f2"],
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
    stack = unroll(np.abs(fingers), M)
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


def test_full_run_vs_original(new, orig, n_cases=300, seed=0):
    """1. The frozen core end to end vs the original: score equal to 1e-12; action changes are reported."""
    rng = np.random.default_rng(seed)
    n_score = n_tail_only = n_other = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        out_new = call_frozen(new, motif, fingers)
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
    overlaps = [NEUTRAL["overlap_penalty_f1"]] + (["l2_norm"] if hasattr(new, "_parse_overlap_penalty") else [])
    n_fail = 0
    for name, motif, fingers in cases:
        for overlap in overlaps:
            try:
                with np.errstate(invalid="ignore", divide="ignore"):  # A zero finger has zero norm
                    call_znf(new, motif, fingers, overlap_penalty_f1=overlap)
            except Exception as exc:
                print(f"  [FAIL] {name} (f1={overlap!r}): {type(exc).__name__}: {exc}")
                n_fail += 1
    print(f"1b. degenerate cases: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_matches_snapshot(new, snap, n_cases=30, seed=4):
    """2. Every penalty configuration matches the snapshot bit for bit, on all three paths."""
    rng = np.random.default_rng(seed)
    nonzero = {
        "overlap": dict(overlap_penalty_f1=-0.02, overlap_penalty_f2=-0.1),
        "skip_l": dict(skip_penalty_l1=-0.02, skip_penalty_l2=-0.1),
        "skip_f": dict(skip_penalty_f1=-0.1, skip_penalty_f2=-0.3),
    }
    zero = {
        "overlap": dict(overlap_penalty_f1=0.0, overlap_penalty_f2=-0.0),
        "skip_l": dict(skip_penalty_l1=0.0, skip_penalty_l2=0.0),
        "skip_f": dict(skip_penalty_f1=-0.0, skip_penalty_f2=0.0),
    }
    configs = []
    for on in itertools.product((False, True), repeat=3):
        cfg = {}
        for key, active in zip(("overlap", "skip_l", "skip_f"), on):
            cfg.update(nonzero[key] if active else zero[key])
        configs.append(cfg)

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
    modes = [dict(), {k: 0.0 for k in ("overlap_penalty_f1", "overlap_penalty_f2", "skip_penalty_l1",
                                         "skip_penalty_l2", "skip_penalty_f1", "skip_penalty_f2")},
             dict(overlap_penalty_f1="l2_norm", overlap_penalty_f2="l2_norm")]
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


def test_l2_norm(new):
    """Hand-computed normalization, including replacement of earlier scaling."""
    np.testing.assert_allclose(new._match_position_scores(
        np.ones((3, 4), dtype=int), np.ones((1, 3, 4), dtype=int), np.ones(1), "for"), 1 / 3)
    np.testing.assert_array_equal(new._match_position_scores(
        np.ones((3, 4)), np.zeros((1, 3, 4)), np.ones(1), "rev"), np.zeros((1, 1, 3)))
    scores = np.zeros((4, 10, 4))
    scores[0, 2] = [0.1, 0.2, 0.3, 0.4]
    scores[1, 3] = [0.5, 0.6, 0.7, 0.8]
    scores[2, 4] = [0.9, 1.0, 1.1, 1.2]

    def penalties(f, l, previous, M=3, f1="l2_norm", f2="l2_norm", offsets=(0, 0)):
        return new._calculate_overlap_penalties(
            f, l, previous[0][1],
            [f"F{fi + offsets[0]}@match@L{li + offsets[1]}" for fi, li in previous],
            M, f1, f2, scores, *offsets)

    # No overlap; then two fingers with two overlapping positions.
    assert penalties(1, 5, [(0, 2)]) == []
    first = penalties(1, 3, [(0, 2)])
    np.testing.assert_allclose(first, [0.7 / math.sqrt(2) - 0.7, 0.9 / math.sqrt(2) - 0.9])
    np.testing.assert_allclose(penalties(1, 3, [(0, 2)], offsets=(5, 10)), first)
    np.testing.assert_allclose(penalties(1, 3, [(0, 2)], f2=-0.1), [first[0], -0.1])
    np.testing.assert_allclose(penalties(1, 3, [(0, 2)], f1=-0.02), [-0.02, first[1]])

    # At p=4, the first two contributions have already been divided by sqrt(2).
    third = penalties(2, 4, [(1, 3), (0, 2)])
    np.testing.assert_allclose(third, [1.8 / math.sqrt(3) - 0.9 / math.sqrt(2) - 0.9,
                                     1.7 / math.sqrt(2) - 1.7])
    np.testing.assert_allclose(0.9 + first[1] + 0.9 + third[0], 1.8 / math.sqrt(3))
    # A prior numeric rule has not normalized the old contributions.
    mixed = penalties(2, 4, [(1, 3), (0, 2)], f2=-0.1)
    np.testing.assert_allclose(mixed, [1.8 / math.sqrt(3) - 1.8, -0.1])

    for spelling in ("l2_norm", "L2_NORM", "L2_Norm"):
        assert new._parse_overlap_penalty(spelling, "x") == "l2_norm"
    for invalid in ("gm", "geometric_mean", "foo"):
        try:
            new._parse_overlap_penalty(invalid, "overlap_penalty_f1")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid mode {invalid!r}")

    # Force known placements in both recurrences. Starting at zero also checks that
    # actual overlapping positions are normalized even if display gaps are truncated.
    for M, starts in ((3, [0]), (3, [0, 3]), (3, [0, 2]), (3, [0, 1, 2]), (4, [0, 1, 2, 3])):
        F, L = len(starts), starts[-1] + M
        motif = np.ones((L, 4)) / math.sqrt(4 * L)
        fingers = np.ones((F, M, 4)) / math.sqrt(4 * M)
        sim = np.zeros((F, L - M + 1))
        sim[np.arange(F), starts] = 1.
        scale = np.full(L - M + 1, M / L)
        counts = np.zeros(L)
        for start in starts:
            counts[start:start + M] += 1
        expected = np.sqrt(counts).sum() / L
        cores = [new.calculate_modified_smith_waterman]
        if M == 3:
            cores.append(new.calculate_modified_smith_waterman_znf)
        for core in cores:
            for orient in ("for", "rev"):
                for alignment_type, offsets in (("local", (0, 0)), ("global", (5, 10))):
                    params = dict(NEUTRAL, overlap_penalty_f1="l2_norm", overlap_penalty_f2="l2_norm")
                    out = core(matrix_l=motif, matrix_f=fingers, sim_for=sim, sim_rev=sim,
                               scale_factor=scale, orientations=(orient,), alignment_type=alignment_type,
                               initial_f_idx=offsets[0], initial_l_idx=offsets[1], **params)
                    np.testing.assert_allclose(out[0], expected, atol=1e-14)
                    matches = [a for a in out[2] if "@match@" in a]
                    assert matches == [f"F{f + offsets[0]}@match@L{l + offsets[1]}" for f, l in enumerate(starts)]

    # Positional contributions sum to the actual cosine match score in either orientation.
    rng = np.random.default_rng(9)
    motif, fingers = make_case(rng, 4, 3, 20)
    stack, motif_norm, sim_for, sim_rev, importance = prepare(new, motif, fingers, 3)
    for orient, sim in (("for", sim_for), ("rev", sim_rev)):
        np.testing.assert_allclose(new._match_position_scores(motif_norm, stack, importance, orient).sum(axis=2),
                                   sim * importance, atol=1e-14)
    for _ in range(15):
        motif, fingers = make_case(rng, 4, 3, 20)
        for call in (call_znf, lambda mod, m, f, **kw: call_general(mod, m, f, 3, **kw)):
            normalized = call(new, motif, fingers, overlap_penalty_f1="l2_norm", overlap_penalty_f2="l2_norm")
            spelled = call(new, motif, fingers, overlap_penalty_f1="L2_NORM", overlap_penalty_f2="L2_NORM")
            assert not differing(normalized, spelled)
    print("4. L2 normalization: hand calculations, 1-4 fingers, both cores/orientations, offsets, mixed rules ok")
    return 0


ZERO = dict(NEUTRAL, overlap_penalty_f1=0.0, overlap_penalty_f2=0.0, skip_penalty_f1=0.0,
            skip_penalty_f2=0.0, skip_penalty_l1=0.0, skip_penalty_l2=0.0)


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
    if not np.allclose(scale, M / L, rtol=0, atol=1e-14):
        problems.append(f"scale {scale}")
    placed = np.zeros((len(expected), 2 * M + L - 2, 4))
    for n, (s, r, start) in enumerate(expected):
        placed[n, start + M - 1:start + 2 * M - 1] = (rc_of(subunits[s]) if r else subunits[s]) / math.sqrt(M) * M / L
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
        check(f"planted={planted}", a[None], target, [(0, r, 4 * i) for i, r in enumerate(expected)],
              max_homo_subunits=copies, allow_reverse_order=False)
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
    matches = [(int(p[0][1:]), int(p[-1][1:])) for p in (a.split("@") for a in ref[7]) if p[1] == "match"]
    if (list(idx) != [stack_subunit[i] for i, _ in matches]
            or list(rc) != [stack_rc[i] ^ (orient == "rev") for i, _ in matches]
            or list(h) != [l + M - 1 for _, l in matches]
            or list(scale) != [ref[6][i] for i, _ in matches]):
        return problems + ["subunit index / orientation / position / scale"]
    # Orientation: The subunit as reported has the similarity that the alignment matched on
    target, subunits = np.abs(motif), np.abs(stack)
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
    composite = aligned.sum(axis=0)
    if composite[:M - 1].any() or not np.allclose(composite[M - 1:], ref[2].sum(axis=0), rtol=0, atol=1e-14):
        problems.append("composite differs from the core's")
    return problems


def test_input_subunit_outputs(new, n_cases=48, seed=10):
    """9. Random cases: the composite wrapper's outputs are the core's on the expanded stack, by input subunit."""
    rng = np.random.default_rng(seed)
    configs = [ZERO, NEUTRAL, dict(NEUTRAL, overlap_penalty_f1="l2_norm", overlap_penalty_f2="l2_norm")]
    flags = list(itertools.product((False, True), repeat=2))
    n_bad = n_runs = n_rev = n_rc = n_repeat = n_sim_up = 0
    for i in range(n_cases):
        M = (3, 4, 2, 5)[i % 4]
        reverse_order, reverse_comp = flags[(i // 4) % 4]
        F, copies = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        motif, stack = make_mixed_case(rng, F, M, int(rng.integers(max(M + 1, 12), 40)))
        expanded, stack_subunit, stack_rc = expand(stack, copies, reverse_order, reverse_comp)
        for cfg in configs:
            out = new.run_modified_smith_waterman_composite(
                motif, stack, max_homo_subunits=copies,
                allow_reverse_order=reverse_order, allow_reverse_comp=reverse_comp, **cfg)
            core = dict(target_motif=motif, subunit_motif_stack=expanded, return_matrices=False, label_unsigned=True,
                        orientations=("for", "rev") if reverse_comp else ("for",), align_remaining=False, **cfg)
            ref = new._run_modified_smith_waterman(place_as_matched=True, **core)
            problems = input_subunit_problems(out, ref, motif, stack, stack_subunit, stack_rc, M)
            n_runs += 1
            n_rc += bool(np.any(out[4]))
            n_repeat += len(set(out[3].tolist())) < len(out[3])
            if ref[3] == "rev":
                n_rev += 1
                n_sim_up += bool(out[1] > new._run_modified_smith_waterman(place_as_matched=False, **core)[1])
            if problems:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  [FAIL] M={M} F={F} copies={copies} reverse order/comp={reverse_order}/{reverse_comp} "
                          f"{cfg}: {problems[:2]}")
    print(f"9. composite outputs: {n_runs - n_bad}/{n_runs} match the core on the expanded stack, over all 4 "
          f"reverse order / complement settings ({n_rc} with reverse complements, {n_repeat} with repeated "
          f"subunits, {n_rev} won by 'rev'; placing 'rev' matches as matched raised sim_score in {n_sim_up}/{n_rev})")
    return n_bad


def test_composite_no_remaining(new, n_cases=40, seed=13):
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
                    place_as_matched=True, **cfg)
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

    # The alignment wrapper is the generalized core at every M, and the (deprecated) ZNF wrapper
    # is the alignment wrapper
    for M in [2, 3, 3, 3, 4, 5]:
        motif, fingers = make_case(rng, 4, M, int(rng.integers(4 * M, 8 * M)))
        aligned = new.run_modified_smith_waterman_alignment(target_motif=motif, reference_motif=fingers,
                                                            finger_length=M, **NEUTRAL)
        for label, other in (("generalized path", call_general(new, motif, fingers, M)),
                             ("ZNF wrapper", call_znf(new, motif, fingers, finger_length=M))):
            diffs = differing(aligned, other)
            if diffs:
                print(f"  [FAIL] alignment wrapper at M={M} differs from the {label}: {diffs}")
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
        if differing(call_frozen(new, motif, fingers), call_general(new, motif, fingers, 3)):
            n_diff += 1
    print(f"6. generalized vs frozen at M=3: {n_cases - n_diff}/{n_cases} identical ({n_diff} differ, expected: "
          f"the frozen core charges an adjacent finger a gap penalty, to the finger before)")


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
        failures += test_l2_norm(new)
    failures += test_subunit_expansion(new)
    failures += test_input_subunit_outputs(new)
    failures += test_composite_no_remaining(new)
    failures += test_general_m(new)
    if hasattr(new, "_alignment_tracks"):
        failures += test_alignment_tracks(new)
    test_general_vs_frozen(new)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(1 if failures else 0)
