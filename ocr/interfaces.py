from __future__ import annotations
from typing import Protocol
import numpy as np


class IPlateReplacer(Protocol):
    """Domain-service contract – implementation is swappable in tests."""

    def replace(self, img: np.ndarray) -> np.ndarray | None: ...
