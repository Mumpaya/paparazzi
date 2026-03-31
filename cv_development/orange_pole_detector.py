# pole_detector.py
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PoleDetection:
    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int
    area: int
    confidence: float


class OrangePoleDetector:
    """
    Detects orange poles via HSV masking + contour filtering.

    detect(bgr) returns a dict with keys:
        - 'boxes'   : list[PoleDetection]
        - 'mask'    : binary orange mask
        - 'blurred' : CLAHE-enhanced + blurred grayscale (for display)
        - 'edges'   : Canny edge image (applied on the masked region)
    """

    def __init__(
        self,

        # PERFECT IN SIMULATION:
        # hsv_lower: tuple = (5,  120,  80),
        # hsv_upper: tuple = (30, 255, 255),

        # TUNED FOR CYBERZOO DATASET:
        hsv_lower: tuple = (0,  40,  50),
        hsv_upper: tuple = (25, 255, 255),
        
        min_area: int = 500,
        min_aspect_ratio: float = 2.5,
        min_confidence: float = 0.3,
        morph_kernel_size: tuple = (5, 5),
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_confidence = min_confidence
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, morph_kernel_size
        )
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (clahe_blurred_gray, orange_mask)."""
        # Grayscale → CLAHE → Gaussian blur (for display row 2)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe_gray = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(clahe_gray, (5, 5), 0)

        # HSV orange mask → morphological cleanup
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        return blurred, mask

    def _canny_on_mask(self, blurred: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Canny edges restricted to the orange mask region."""
        masked_blur = cv2.bitwise_and(blurred, blurred, mask=mask)
        edges = cv2.Canny(masked_blur, self.canny_low, self.canny_high)
        return edges

    def _filter_contours(
        self, contours, mask: np.ndarray
    ) -> list[PoleDetection]:
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if (h / max(w, 1)) < self.min_aspect_ratio:
                continue

            roi_mask = mask[y:y + h, x:x + w]
            confidence = roi_mask.sum() / 255 / max(w * h, 1)
            if confidence < self.min_confidence:
                continue

            detections.append(PoleDetection(
                x=x, y=y, w=w, h=h,
                cx=x + w // 2,
                cy=y + h // 2,
                area=int(area),
                confidence=round(confidence, 2),
            ))

        detections.sort(key=lambda d: d.cx)
        return detections

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, bgr: np.ndarray) -> dict:
        """
        Run full detection pipeline.

        Returns
        -------
        dict with keys:
            'boxes'   : list[PoleDetection]  — sorted left→right
            'mask'    : np.ndarray (uint8)   — binary orange mask
            'blurred' : np.ndarray (uint8)   — CLAHE gray for plotting
            'edges'   : np.ndarray (uint8)   — Canny on masked region
        """
        blurred, mask = self._preprocess(bgr)
        edges = self._canny_on_mask(blurred, mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = self._filter_contours(contours, mask)

        return {
            "boxes":   boxes,
            "mask":    mask,
            "blurred": blurred,
            "edges":   edges,
        }

    def draw(self, bgr: np.ndarray, result: dict) -> np.ndarray:
        """
        Annotate image with bounding boxes, centroids, and vertical lines.

        Parameters
        ----------
        bgr    : original BGR image
        result : dict returned by detect()
        """
        out = bgr.copy()
        h_img = bgr.shape[0]

        # Subtle green tint over orange mask
        tint = out.copy()
        tint[result["mask"] > 0] = [0, 180, 0]
        out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)

        for i, d in enumerate(result["boxes"]):
            # Bounding box
            cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h),
                          (0, 255, 255), 2)
            # Centroid dot
            cv2.circle(out, (d.cx, d.cy), 5, (0, 0, 255), -1)
            # Vertical centre line
            cv2.line(out, (d.cx, 0), (d.cx, h_img), (0, 0, 255), 1)
            # Label
            cv2.putText(
                out,
                f"P{i} cx={d.cx} ({d.confidence:.2f})",
                (d.x, max(d.y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
            )

        return out
