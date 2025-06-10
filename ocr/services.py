from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from .interfaces import IPlateReplacer


# ---------- helpers ----------------------------------------------------------
def _bbox_to_quad(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _order_quad(quad: np.ndarray) -> np.ndarray:
    s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
    o = np.zeros((4, 2), float)
    o[0], o[2] = quad[np.argmin(s)], quad[np.argmax(s)]  # TL, BR
    o[1], o[3] = quad[np.argmin(d)], quad[np.argmax(d)]  # TR, BL
    return o


def _warp_src(custom: np.ndarray, dst_shape, quad: np.ndarray):
    h, w = custom.shape[:2]
    H, _ = cv2.findHomography(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), _order_quad(quad).astype(np.float32)
    )
    warp = cv2.warpPerspective(custom, H, (dst_shape[1], dst_shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H,
                               (dst_shape[1], dst_shape[0]),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warp, mask


# ---------- concrete service -------------------------------------------------
class YoloPlateReplacer(IPlateReplacer):
    """
    Stateless domain service.  Heavy model is loaded once per
    Python process (module import), so the REST endpoint stays fast.
    """

    def __init__(self, det_weights: str | Path, custom_plate: str | Path, conf=0.25):
        self._detector = YOLO(str(det_weights))
        self._custom = cv2.imread(str(custom_plate), cv2.IMREAD_COLOR)
        if self._custom is None:
            raise FileNotFoundError(custom_plate)
        self._conf = conf

    # --- public API required by IPlateReplacer --------------------------------
    def replace(self, img: np.ndarray) -> np.ndarray | None:
        r = self._detector(img, conf=self._conf, verbose=False)[0]
        if len(r.boxes) == 0:
            return None  # plate not found

        # quad: seg mask preferred, else bbox
        if hasattr(r, "masks") and r.masks:
            areas = [cv2.contourArea(p.astype(np.float32)) for p in r.masks.xy]
            rect = cv2.minAreaRect(r.masks.xy[int(np.argmax(areas))].astype(int))
            quad = cv2.boxPoints(rect)
        else:
            quad = _bbox_to_quad(r.boxes.xyxy[r.boxes.conf.argmax()].cpu().numpy())

        out = img.copy()
        cv2.fillConvexPoly(out, quad.astype(int), (255, 255, 255))
        warp, mask = _warp_src(self._custom, out.shape, quad)
        out[mask == 255] = warp[mask == 255]
        return out
