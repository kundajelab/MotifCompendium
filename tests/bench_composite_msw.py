"""
Benchmark the modified Smith-Waterman DP core.

Run:
    python tests/bench_composite_msw.py [path/to/reference/composite.py]

Times the DP core alone (inputs pre-computed, so the GPU similarity step is excluded)
and the full end-to-end call. With a reference given, reports the speedup.
"""

import sys
import time

import numpy as np

from test_composite_msw import NEUTRAL, load_module, make_case, prepare


def make_workload(mod, n_cases=40, M=3, seed=11):
    rng = np.random.default_rng(seed)
    workload = []
    for _ in range(n_cases):
        F = int(rng.integers(4, 11))
        L = int(rng.integers(30, 61))
        motif, fingers = make_case(rng, F, M, L)
        workload.append((motif, fingers) + prepare(mod, motif, fingers, M))
    return workload


def time_core(fn, workload, legacy):
    shared_extra = (dict(skip_penalty_f=NEUTRAL["skip_penalty_f1"]) if legacy else
                    dict(skip_penalty_f1=NEUTRAL["skip_penalty_f1"],
                         skip_penalty_f2=NEUTRAL["skip_penalty_f2"],
                         overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
                         return_matrices=False))
    start = time.perf_counter()
    for _, _, fingers_n, motif_norm, sim_for, sim_rev, importance in workload:
        fn(matrix_f=fingers_n, matrix_l=motif_norm, sim_for=sim_for, sim_rev=sim_rev,
           scale_factor=importance, sim_threshold=0.0,
           overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
           overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
           skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
           skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
           alignment_type="local", initial_score=0.0, **shared_extra)
    return time.perf_counter() - start


def time_end_to_end(mod, workload, legacy):
    extra = ({} if legacy else dict(overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
                                    skip_penalty_f2=NEUTRAL["skip_penalty_f2"]))
    key = "skip_penalty_f" if legacy else "skip_penalty_f1"
    start = time.perf_counter()
    for motif, fingers, *_ in workload:
        mod.run_modified_smith_waterman_znf(
            cwm=motif, b1h_ppm_ic=fingers, sim_threshold=0.0,
            overlap_penalty_l1=NEUTRAL["overlap_penalty_l1"],
            overlap_penalty_l2=NEUTRAL["overlap_penalty_l2"],
            skip_penalty_l1=NEUTRAL["skip_penalty_l1"],
            skip_penalty_l2=NEUTRAL["skip_penalty_l2"],
            finger_length=3, **{key: NEUTRAL["skip_penalty_f1"]}, **extra)
    return time.perf_counter() - start


if __name__ == "__main__":
    new = load_module(
        "/oak/stanford/groups/akundaje/cmyun/software/motifcompendium/MotifCompendium/utils/composite.py",
        "composite_new")
    workload = make_workload(new)
    n = len(workload)

    t_new_core = time_core(new.calculate_modified_smith_waterman_znf, workload, legacy=False)
    t_new_gen = time_core(new.calculate_modified_smith_waterman, workload, legacy=False)
    t_new_e2e = time_end_to_end(new, workload, legacy=False)
    print(f"new  frozen core      {t_new_core:7.3f}s  ({1000 * t_new_core / n:6.1f} ms/case)")
    print(f"new  generalized core {t_new_gen:7.3f}s  ({1000 * t_new_gen / n:6.1f} ms/case)")
    print(f"new  end to end       {t_new_e2e:7.3f}s  ({1000 * t_new_e2e / n:6.1f} ms/case)")

    if len(sys.argv) > 1:
        ref = load_module(sys.argv[1], "composite_ref")
        t_ref_core = time_core(ref.calculate_modified_smith_waterman, workload, legacy=True)
        t_ref_e2e = time_end_to_end(ref, workload, legacy=True)
        print(f"ref  core             {t_ref_core:7.3f}s  ({1000 * t_ref_core / n:6.1f} ms/case)")
        print(f"ref  end to end       {t_ref_e2e:7.3f}s  ({1000 * t_ref_e2e / n:6.1f} ms/case)")
        print(f"\nspeedup: core {t_ref_core / t_new_core:.2f}x, end to end {t_ref_e2e / t_new_e2e:.2f}x")
