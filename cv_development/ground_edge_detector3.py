


import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class GroundEdgeResult:
    green_mask: np.ndarray
    not_green_mask: np.ndarray
    not_uniform_mask: np.ndarray
    mask_good: np.ndarray
    mask_bad: np.ndarray
    goodness: float
    centroid: tuple
    over_edge: bool


class GroundEdgeDetector:
    def __init__(
        self,
        hsv_lower_green: tuple = (0, 15, 120),
        hsv_upper_green: tuple = (66, 225, 255),
        hsv_lower_not_green: tuple = (88, 0, 0),
        hsv_upper_not_green: tuple = (255, 255, 255),
        blur_ksize: int = 9,
        morph_ksize: int = 10,
    ):
        self.hsv_lower_green          = np.array(hsv_lower_green)
        self.hsv_upper_green          = np.array(hsv_upper_green)
        self.hsv_lower_not_green = np.array(hsv_lower_not_green)
        self.hsv_upper_not_green = np.array(hsv_upper_not_green)
        self.blur_ksize         = blur_ksize
        self.morph_ksize        = morph_ksize



    def mask_non_uniform(self, bgr: np.ndarray, ksize: int = 11, thresh: float = 15.0) -> np.ndarray:
        """
        Returns a binary mask where non-uniform (high-texture) regions are white.

        1. Convert to grayscale.
        2. Compute local mean and local mean-of-squares via box filter.
        3. Local std dev = sqrt(mean_of_squares - mean^2).
        4. Threshold: high std dev → non-uniform.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Local mean and local variance via box filter
        local_mean = cv2.blur(gray, (ksize, ksize))
        local_sq_mean = cv2.blur(gray * gray, (ksize, ksize))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        # Threshold: high std = non-uniform
        mask = (local_std > thresh).astype(np.uint8) * 255
        return mask


    def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
        """
        Main detection function. Steps:
        1. Blur + convert to HSV.
        2. Create green mask
        3. create non-uniform mask
        4. create not green mask
        5. clean up masks with morphology (not needed)
        6. combine masks (good = non-uniform OR green, bad = not-green AND NOT non-uniform)
        7. compute goodness = fraction of good pixels
        8. compute centroid of good pixels
        send to control: goodness, centroid-centre, over_edge
        """
        h_img, w_img = bgr.shape[:2]

        blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, self.hsv_lower_green, self.hsv_upper_green)
        mask_non_uniform = self.mask_non_uniform(bgr)
        mask_not_green = cv2.inRange(hsv, self.hsv_lower_not_green, self.hsv_upper_not_green)

        # closekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # openkernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, closekernel)
        # mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, openkernel)
        # mask_non_uniform = cv2.morphologyEx(mask_non_uniform, cv2.MORPH_CLOSE, closekernel)
        # mask_non_uniform = cv2.morphologyEx(mask_non_uniform, cv2.MORPH_OPEN, openkernel)
        # mask_non_uniform = cv2.morphologyEx(mask_non_uniform, cv2.MORPH_CLOSE, closekernel)
        # mask_not_green = cv2.morphologyEx(mask_not_green, cv2.MORPH_CLOSE, closekernel)
        # mask_not_green = cv2.morphologyEx(mask_not_green, cv2.MORPH_OPEN, openkernel)

        mask_bad = cv2.subtract(mask_not_green , mask_non_uniform)
        mask_good = cv2.bitwise_or(mask_green, mask_non_uniform)
        mask_good = cv2.subtract(mask_good, mask_bad)
        good_frac = np.mean(mask_good) / 255.0
        moments = cv2.moments(mask_good)

        if moments['m00'] > 0:  # m00 is the area (number of white pixels)
            avg_x = moments['m10'] / moments['m00']
            avg_y = moments['m01'] / moments['m00']
            centroid = (int(avg_x), int(avg_y))
        else:
            centroid = None

        mag = np.sqrt((centroid[0] - w_img//2)**2+(centroid[1] - h_img//2)**2) if centroid else float('inf')
        over_the_edge = mag > (w_img // 4) * 0.8

        return GroundEdgeResult(
            green_mask=mask_green,
            not_uniform_mask=mask_non_uniform,
            not_green_mask=mask_not_green,
            mask_good=mask_good,
            mask_bad=mask_bad,
            centroid=centroid,
            goodness=float(good_frac),
            over_edge=over_the_edge
        )



    def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
        overlay = bgr.copy()
        overlay*=0
        # overlay[result.green_mask > 0, 1] += 255
        # overlay[result.not_uniform_mask > 0, 0] +=255
        # overlay[result.mask_bad > 0, 2] +=255
        # overlay[result.mask_good > 0, 1] +=255
        out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
        h_img, w_img = bgr.shape[:2]
        if result.centroid is not None:
            frame_center = (w_img // 2, h_img // 2)
            cv2.arrowedLine(out, frame_center, result.centroid, (255, 255, 0), 2, tipLength=0.3)
        return out


@dataclass
class GroundFlowResult:
    base_im: np.ndarray
    flow_vector: tuple


class GroundFlow:
    def measure(self, bgr: np.ndarray) -> GroundFlowResult:
        base_im = bgr[::, ::, 1]
        base_im = cv2.GaussianBlur(base_im, (9, 9), 0)
        return GroundFlowResult(
            base_im=base_im,
            flow_vector=(0, 0),
        )

    def draw(self, bgr: np.ndarray, result: GroundFlowResult) -> np.ndarray:
        overlay = bgr.copy()
        overlay*=0
        out = result.base_im.copy()
        h_img, w_img = bgr.shape[:2]
        frame_center = (w_img // 2, h_img // 2)
        flow_endpoint = (frame_center[0] + int(result.flow_vector[0]*50), frame_center[1] + int(result.flow_vector[1]*50))
        cv2.arrowedLine(out, frame_center, flow_endpoint, (255, 255, 0), 2, tipLength=0.3)
        return out