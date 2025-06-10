from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplacePlateCommand:
    """DTO coming *into* the use-case (raw image bytes)."""
    image_bytes: bytes


@dataclass(frozen=True)
class ReplacePlateResult:
    """DTO going *out* of the use-case (NumPy image or None)."""
    image: np.ndarray | None
