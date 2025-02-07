####################
# COMPUTE SETTINGS #
####################
_MAX_CHUNK = -1
_MAX_CPUS = 1
_USE_GPU = False


def get_max_chunk() -> int:
    return _MAX_CHUNK

def get_max_cpus() -> int:
    return _MAX_CPUS

def get_use_gpu() -> bool:
    return _USE_GPU

def set_max_chunk(max_chunk: int) -> None:
    if not isinstance(max_chunk, int):
        raise TypeError("Max chunk must be an integer.")
    if (max_chunk < 1) and (max_chunk != -1):
        raise ValueError("Max chunk must be greater than 1 (or -1 if no chunking).")
    global _MAX_CHUNK
    _MAX_CHUNK = max_chunk

def set_max_cpus(max_cpus: int) -> None:
    if not isinstance(max_cpus, int):
        raise TypeError("Max CPUs must be an integer.")
    if max_cpus < 1:
        raise ValueError("Max CPUs must be greater than 1.")
    global _MAX_CPUS
    _MAX_CPUS = max_cpus

def set_use_gpu(use_gpu: bool) -> None:
    if not isinstance(use_gpu, bool):
        raise TypeError("Use GPU must be a boolean.")
    global _USE_GPU
    _USE_GPU = use_gpu