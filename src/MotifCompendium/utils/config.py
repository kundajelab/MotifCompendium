####################
# COMPUTE SETTINGS #
####################
_MAX_CPUS = 1
_USE_GPU = False
_IC_SCALED_SIMILARITY = True
_MAX_CHUNK = -1
_PROGRESS_BAR = False
_FAST_PLOTTING = False


def get_max_cpus() -> int:
    """Get the maximum number of CPUs to use."""
    return _MAX_CPUS


def get_use_gpu() -> bool:
    """Get whether to use GPU acceleration."""
    return _USE_GPU


def get_ic_scaled_similarity() -> bool:
    """Whether or not to perform IC scaling on motifs before computing similarity."""
    return _IC_SCALED_SIMILARITY


def get_max_chunk() -> int:
    """Get similarity calculation chunk size."""
    return _MAX_CHUNK


def get_progress_bar() -> bool:
    """Whether or not to show progress bars."""
    return _PROGRESS_BAR


def get_fast_plotting() -> bool:
    """Whether or not to use fast plotting instead of logomaker."""
    return _FAST_PLOTTING


def set_max_cpus(max_cpus: int) -> None:
    """Set the maximum number of CPUs to use."""
    if not isinstance(max_cpus, int):
        raise TypeError("Max CPUs must be an integer.")
    if max_cpus < 0:
        raise ValueError("Max CPUs must be >= 1.")
    global _MAX_CPUS
    _MAX_CPUS = max_cpus


def set_use_gpu(use_gpu: bool) -> None:
    """Set whether to use GPU acceleration."""
    if not isinstance(use_gpu, bool):
        raise TypeError("Use GPU must be a boolean.")
    global _USE_GPU
    _USE_GPU = use_gpu


def set_ic_scaled_similarity(ic_scaled_similarity: bool) -> None:
    """Set whether or not to perform IC scaling on motifs before computing similarity."""
    if not isinstance(ic_scaled_similarity, bool):
        raise TypeError("IC scaled similarity must be a boolean.")
    global _IC_SCALED_SIMILARITY
    _IC_SCALED_SIMILARITY = ic_scaled_similarity


def set_max_chunk(max_chunk: int) -> None:
    """Set similarity calculation chunk size."""
    if not isinstance(max_chunk, int):
        raise TypeError("Max chunk must be an integer.")
    if (max_chunk < 1) and (max_chunk != -1):
        raise ValueError("Max chunk must be >= 1 (or -1 if no chunking).")
    global _MAX_CHUNK
    _MAX_CHUNK = max_chunk


def set_progress_bar(progress_bar: bool) -> None:
    """Set whether or not to show progress bars."""
    if not isinstance(progress_bar, bool):
        raise TypeError("Progress bar must be a boolean.")
    global _PROGRESS_BAR
    _PROGRESS_BAR = progress_bar


def set_fast_plotting(fast_plotting: bool) -> None:
    """Set whether or not to use fast plotting instead of logomaker."""
    if not isinstance(fast_plotting, bool):
        raise TypeError("Fast plotting must be a boolean.")
    global _FAST_PLOTTING
    _FAST_PLOTTING = fast_plotting
