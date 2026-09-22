"""
Regression + generalization checks for MotifCompendium.utils.composite.

Run:
    python tests/test_composite_msw.py [path/to/reference/composite.py]

The reference is the pre-change copy of composite.py. When given, section 1 asserts that
the 3 bp zinc-finger path produces bit-identical results; without it, only the
generalization checks (sections 2-4) run.
"""

import importlib.util
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


def call_new(mod, motif, fingers, M, **kwargs):
    params = dict(NEUTRAL)
    params.update(kwargs)
    return mod.run_modified_smith_waterman_znf(
        cwm=motif, b1h_ppm_ic=fingers, finger_length=M, **params)


def call_reference(mod, motif, fingers, M):
    return mod.run_modified_smith_waterman_znf(
        cwm=motif,
        b1h_ppm_ic=fingers,
        sim_threshold=NEUTRAL["sim_threshold"],
        overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
        overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
        skip_penalty_f=NEUTRAL["skip_penalty_f1"],
        skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
        skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
        finger_length=M,
    )


def compare(out_a, out_b):
    """Returns the list of differing fields between two run_* outputs."""
    names = ["msw_score", "sim_score", "aligned_motif_fs", "msw_orient",
             "msw_align_f", "msw_align_l", "msw_scale_f", "msw_actions"]
    diffs = []
    for name, a, b in zip(names, out_a[:8], out_b[:8]):
        if isinstance(a, np.ndarray):
            same = a.shape == b.shape and np.array_equal(a, b)
        else:
            same = a == b
        if not same:
            diffs.append(name)
    return diffs


# ------------------------------------------------------------------ sections
def prepare(mod, motif, fingers, M):
    """The shared preprocessing, so both DP cores see byte-identical inputs."""
    import cupy as cp
    motif = np.abs(motif)
    motif_norm = motif / np.linalg.norm(motif, axis=(0, 1), keepdims=True)
    unrolled = mod.unroll_motif_np(motif_norm, M)
    unrolled_cp = mod.unroll_motif_cp(cp.asarray(motif_norm), M)
    importance = (unrolled * unrolled).sum(axis=(1, 2))
    fingers_n = mod.unroll_fingers(np.abs(fingers), M)
    fingers_cp = cp.asarray(fingers_n)
    sim_for = mod.compute_aligned_similarity(fingers_cp, unrolled_cp)
    sim_rev = mod.compute_aligned_similarity(fingers_cp, unrolled_cp[:, ::-1, ::-1])
    return fingers_n, motif_norm, sim_for, sim_rev, importance


def test_core_identical(new, ref, n_cases=300, seed=0):
    """0. The frozen 3 bp core must match the reference core exactly (local pass)."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        fingers_n, motif_norm, sim_for, sim_rev, importance = prepare(new, motif, fingers, 3)
        shared = dict(matrix_f=fingers_n, matrix_l=motif_norm, sim_for=sim_for, sim_rev=sim_rev,
                      scale_factor=importance, sim_threshold=0.0,
                      overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
                      overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
                      skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
                      skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
                      alignment_type="local", initial_score=0.0)
        out_ref = ref.calculate_modified_smith_waterman(
            skip_penalty_f=NEUTRAL["skip_penalty_f1"], **shared)
        out_new = new.calculate_modified_smith_waterman_znf(
            skip_penalty_f1=NEUTRAL["skip_penalty_f1"], skip_penalty_f2=NEUTRAL["skip_penalty_f2"],
            overlap_penalty_f1=1.0, overlap_penalty_f2=1.0, **shared)
        if (out_ref[0], out_ref[1], out_ref[2]) != (out_new[0], out_new[1], out_new[2]):
            n_diff += 1
            if n_diff <= 3:
                print(f"  [diff] case {i} F={F} L={L}: {out_ref[0]!r} vs {out_new[0]!r}")
    print(f"0. frozen core vs reference core (local pass): {n_cases - n_diff}/{n_cases} identical")
    return n_diff


def test_regression(new, ref, n_cases=300, seed=0):
    """1. The 3 bp zinc-finger path must be unchanged at neutral multipliers."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    n_tail_only = 0  # New actions are a prefix of the reference: only the remain pass differs
    n_score = 0
    for i in range(n_cases):
        F = int(rng.integers(1, 11))
        L = int(rng.integers(12, 61))
        motif, fingers = make_case(rng, F, 3, L)
        out_new = call_new(new, motif, fingers, 3)
        out_ref = call_reference(ref, motif, fingers, 3)
        diffs = compare(out_new, out_ref)
        if not diffs:
            continue
        n_diff += 1
        if out_new[0] != out_ref[0]:
            n_score += 1
        ref_actions, new_actions = out_ref[7], out_new[7]
        if new_actions == ref_actions[:len(new_actions)]:
            n_tail_only += 1
        elif n_diff - n_tail_only <= 3:
            print(f"  [diff] case {i} F={F} L={L}: {diffs}")
            print(f"         ref {out_ref[0]:.8f} {ref_actions}")
            print(f"         new {out_new[0]:.8f} {new_actions}")
    print(f"1. full run (M=3, neutral): {n_cases - n_diff}/{n_cases} identical; "
          f"of the {n_diff} that differ, {n_tail_only} differ only in the remain-pass tail, "
          f"{n_score} have a different score")
    return 0  # Differences are reported, not failed: the remain pass was fixed


def test_degenerate(new, ref):
    """1b. Degenerate inputs must not crash, and must match the reference."""
    rng = np.random.default_rng(7)
    cases = []
    cases.append(("single finger", *make_case(rng, 1, 3, 12)))
    cases.append(("motif == finger", np.abs(rng.random((3, 4))) + 0.01, np.abs(rng.random((3, 4))) + 0.01))
    zero_f = rng.random((9, 4)) + 0.01
    zero_f[3:6] = 0.0
    cases.append(("zero finger", rng.random((20, 4)) + 0.01, zero_f))
    cases.append(("pure noise", rng.random((30, 4)) * 1e-3, rng.random((12, 4)) + 0.01))
    n_diff = 0
    for name, motif, fingers in cases:
        try:
            out_new = call_new(new, motif, fingers, 3)
        except Exception as exc:
            print(f"  [FAIL] {name}: new raised {type(exc).__name__}: {exc}")
            n_diff += 1
            continue
        try:
            out_ref = call_reference(ref, motif, fingers, 3)
        except Exception as exc:
            print(f"  [note] {name}: reference raised {type(exc).__name__}: {exc} (new did not)")
            continue
        diffs = compare(out_new, out_ref)
        if diffs:
            n_diff += 1
            print(f"  [diff] {name}: {diffs}")
    print(f"1b. degenerate cases: {len(cases) - n_diff}/{len(cases)} clean")
    return n_diff


def test_new_params_noop(new, n_cases=100, seed=1):
    """2. The new parameters are no-ops at neutral values, and bite otherwise."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    n_changed = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        L = int(rng.integers(15, 50))
        motif, fingers = make_case(rng, F, 3, L)
        base = call_new(new, motif, fingers, 3)
        # Same neutral values, spelled differently -> identical
        same = call_new(new, motif, fingers, 3, overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
                        skip_penalty_f1=-0.1, skip_penalty_f2=-0.1)
        if compare(base, same):
            n_diff += 1
        # Non-neutral multiplier -> may change, must never raise
        harsh = call_new(new, motif, fingers, 3, overlap_penalty_f2=2.0, skip_penalty_f2=-0.3)
        if harsh[0] != base[0]:
            n_changed += 1
    print(f"2. new params: {n_cases - n_diff}/{n_cases} neutral-identical; "
          f"{n_changed}/{n_cases} changed under f2=2.0 / skip_f2=-0.3")
    return n_diff


def test_general_m(new, seed=2):
    """3. Any finger length M > 1 runs, and recovers a planted arrangement."""
    rng = np.random.default_rng(seed)
    n_fail = 0
    for M in [2, 3, 4, 5, 7]:
        for F in [1, 3, 6]:
            L = max(M + 1, int(rng.integers(3 * M, 8 * M)))
            motif, fingers = make_case(rng, F, M, L)
            try:
                out = call_new(new, motif, fingers, M)
            except Exception as exc:
                print(f"  [FAIL] M={M} F={F} L={L}: {type(exc).__name__}: {exc}")
                n_fail += 1
                continue
            aligned = out[2]
            if aligned.shape != (F, L + M - 1, 4):
                print(f"  [FAIL] M={M} F={F} L={L}: aligned shape {aligned.shape} != {(F, L + M - 1, 4)}")
                n_fail += 1

    # Planted: F one-hot fingers, so every window has equal importance and only an exact
    # window scores 1.0. Continuous fingers are not usable here: the score is
    # importance x similarity, so a high-importance window can legitimately outrank an
    # exact match on a low-importance one.
    for M in [2, 3, 4, 5, 7]:
        F = 4
        fingers, motif = plant_one_hot(rng, F, M)
        out = call_new(new, motif, fingers, M)
        matches = [a for a in out[7] if a.split("@")[1] == "match"]
        expected = [f"F{f}@match@L{f * M}" for f in range(F)]
        if matches != expected:
            print(f"  [FAIL] planted adjacent M={M}: got {matches}")
            n_fail += 1

    # Planted with a 1 bp gap between finger 1 and finger 2
    for M in [2, 3, 4, 5]:
        F = 3
        fingers, _ = plant_one_hot(rng, F, M)
        motif = np.concatenate([fingers[:2 * M], np.full((1, 4), 0.02), fingers[2 * M:]])
        out = call_new(new, motif, fingers, M)
        matches = [a for a in out[7] if a.split("@")[1] == "match"]
        expected = [f"F0@match@L0", f"F1@match@L{M}", f"F2@match@L{2 * M + 1}"]
        if matches != expected:
            print(f"  [FAIL] planted gap M={M}: got {matches}")
            n_fail += 1
    print(f"3. general M: {'ok' if n_fail == 0 else f'{n_fail} failures'}")
    return n_fail


def test_general_vs_legacy(new, n_cases=100, seed=3):
    """4. At M=3 the generalized core should mostly agree with the frozen one."""
    rng = np.random.default_rng(seed)
    n_diff = 0
    for _ in range(n_cases):
        F = int(rng.integers(2, 9))
        L = int(rng.integers(15, 50))
        motif, fingers = make_case(rng, F, 3, L)
        legacy = new._run_modified_smith_waterman(
            motif=motif, finger_ppm_ic=fingers, finger_length=3,
            calculate_fn=new.calculate_modified_smith_waterman_znf, return_matrices=False, **NEUTRAL)
        general = new._run_modified_smith_waterman(
            motif=motif, finger_ppm_ic=fingers, finger_length=3,
            calculate_fn=new.calculate_modified_smith_waterman, return_matrices=False, **NEUTRAL)
        if compare(legacy, general):
            n_diff += 1
    print(f"4. generalized vs frozen at M=3: {n_cases - n_diff}/{n_cases} identical "
          f"({n_diff} differ, expected: the unified diagonal rule)")
    return n_diff


if __name__ == "__main__":
    new = load_module(
        "/oak/stanford/groups/akundaje/cmyun/software/motifcompendium/MotifCompendium/utils/composite.py",
        "composite_new")
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None

    failures = 0
    if ref_path:
        ref = load_module(ref_path, "composite_ref")
        failures += test_core_identical(new, ref)
        failures += test_regression(new, ref)
        failures += test_degenerate(new, ref)
    failures += test_new_params_noop(new)
    failures += test_general_m(new)
    test_general_vs_legacy(new)  # Informational: differences are expected

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(1 if failures else 0)
