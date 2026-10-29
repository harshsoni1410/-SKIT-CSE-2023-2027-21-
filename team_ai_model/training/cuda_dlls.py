"""
Let TensorFlow 2.10 find the CUDA / cuDNN DLLs that come from pip (requirements-gpu.txt).

TF 2.10 on Windows loads cudart64_110.dll, cublas64_11.dll, cudnn64_8.dll, ... through the
normal DLL search path. The nvidia-*-cu11 wheels install them under
venv/Lib/site-packages/nvidia/<library>/bin, which is not on that path, so without this TF
prints "Could not load dynamic library" and quietly trains on the CPU.

Import this before tensorflow (model.py does). Does nothing if the GPU packages are not
installed or on non-Windows systems.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_cuda_dll_dirs() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        import nvidia  # namespace package created by the nvidia-*-cu11 wheels
    except ImportError:
        return []
    added = []
    for base in nvidia.__path__:
        for bin_dir in sorted(Path(base).glob("*/bin")):
            path = str(bin_dir)
            # TF uses plain LoadLibrary, which searches PATH; add_dll_directory covers
            # dependencies loaded by those DLLs
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
            added.append(path)
    return added


CUDA_DLL_DIRS = add_cuda_dll_dirs()
