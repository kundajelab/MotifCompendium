import pandas as pd
import pooch

from .MotifCompendium import load as load_MotifCompendium
from .MotifCompendium import MotifCompendium as MotifCompendiumClass

__all__ = ["load_tutorial_data", "HDMA_compendium"]


def silent_pooch_fetching(func):
    """Decorator to silence pooch fetching messages."""

    def wrapper(*args, **kwargs):
        logger = pooch.get_logger()
        was_disabled = logger.disabled
        logger.disabled = True
        result = func(*args, **kwargs)
        logger.disabled = was_disabled
        return result

    return wrapper


#################
# TUTORIAL DATA #
#################
tutorial_doi = "doi:10.5281/zenodo.17353762"
tutorial_data = pooch.create(
    path=pooch.os_cache("MotifCompendium/data/tutorial_data"),
    base_url=tutorial_doi,
    version="0",
    registry=None,
)
tutorial_data.load_registry_from_doi()


@silent_pooch_fetching
def load_tutorial_data() -> dict[str, pd.DataFrame]:
    """Load tutorial data.

    Returns:
        A dictionary from tutorial data file name to tutorial data file path.

    Note:
        All tutorial data files are downloaded to the local cache directory.
    """
    data_objs = {
        "cardiomyocyte_modisco": "cardiomyocyte_modisco.h5",
        "endothelial_modisco": "endothelial_modisco.h5",
        "JASPAR_pfms": "JASPAR2024_CORE_non-redundant_pfms_meme.txt",
        "HOCOMOCO_pfms": "H14CORE_pfms.txt",
    }  # NOTE: Currently on dataV0
    # data_objs = {
    #     "cardiomyocyte_modisco": "cardiomyocyte_modisco.h5",
    #     "endothelial_modisco": "endothelial_modisco.h5",
    #     "reference_mc": "reference_compendium.mc",
    #     "reference_pfms": "h13_motifs.pfm",
    #     "tutorial_mc": "tutorial_compendium.mc",
    # } # NOTE: Currently on dataV1
    data_files = {x: tutorial_data.fetch(y) for x, y in data_objs.items()}
    return data_files


########
# HDMA #
########
@silent_pooch_fetching
def HDMA_compendium() -> MotifCompendiumClass:
    """Load HDMA MotifCompendium.

    Returns:
        A MotifCompendium containing motifs from the HDMA paper.

    Note:
        Paper link: https://www.biorxiv.org/content/10.1101/2025.04.30.651381.
    """
    hdma_file = tutorial_data.fetch("HDMA.mc")
    return load_MotifCompendium(hdma_file)
