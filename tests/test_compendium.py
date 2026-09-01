import numpy as np
import pandas as pd

import MotifCompendium


def test_combine():
    mc = dummy_random_compendium()
    mc_0 = mc[:5]
    mc_1 = mc[5:]
    combined = MotifCompendium.combine([mc_0, mc_1])
    mc["source_compendium"] = ["mc_0"] * 5 + ["mc_1"] * 5
    assert(combined == mc)


############################
# Testing helper functions #
############################
def dummy_random_motifs():
    return np.random.rand(10, 30, 4)


def dummy_random_compendium():
    return MotifCompendium.build(dummy_random_motifs())