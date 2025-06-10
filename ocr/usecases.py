import cv2
import numpy as np
from io import BytesIO
from .entities import ReplacePlateCommand, ReplacePlateResult
from .interfaces import IPlateReplacer


class ReplacePlateUseCase:
    """Application-layer orchestration (very thin)."""

    def __init__(self, replacer: IPlateReplacer):
        self._replacer = replacer

    # ――― single public method ―――
    def execute(self, cmd: ReplacePlateCommand) -> ReplacePlateResult:
        # 1) bytes ⇒ NumPy BGR
        nparr = np.frombuffer(cmd.image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Uploaded file is not a valid image")

        # 2) call domain service
        out = self._replacer.replace(img)

        return ReplacePlateResult(image=out)
