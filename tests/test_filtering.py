import numpy as np
import pandas as pd

import MotifCompendium
# import MotifCompendium.analysis.filtering as analysis_filtering


def test_basic_filtering():
    mc = dummy_random_compendium()
    mc["posneg"] = np.random.choice(["pos", "neg"], size=len(mc))
    # analysis_filtering.calculate_filters(mc)


############################
# Testing helper functions #
############################
def dummy_random_motifs():
    return np.random.rand(10, 30, 4)


def dummy_random_compendium():
    return MotifCompendium.build(dummy_random_motifs())
