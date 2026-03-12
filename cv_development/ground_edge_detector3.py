


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
        hough_threshold: int = 54,
        hough_min_length: int = 80,
        hough_max_gap: int = 36,
        blur_ksize: int = 9,
        morph_ksize: int = 10,
        min_area: int = 500,
        history_len: int = 5,
        confirm_frames: int = 2,
        angle_tol_deg: float = 12.0,
        dist_tol_px: float = 34.0,
        sample_dist: int = 20,
        n_samples: int = 20,
        green_frac_thresh: float = 0.35,
        bad_frac_thresh: float = 0.35,
        hue_var_thresh: float = 10.0,
        oscillation_thresh: float = 0.25,
    ):
        self.hsv_lower_green          = np.array(hsv_lower_green)
        self.hsv_upper_green          = np.array(hsv_upper_green)
        self.hsv_lower_not_green = np.array(hsv_lower_not_green)
        self.hsv_upper_not_green = np.array(hsv_upper_not_green)
        self.blur_ksize         = blur_ksize
        self.morph_ksize        = morph_ksize
        self.min_area           = min_area
        self.hough_threshold    = hough_threshold
        self.hough_min_length   = hough_min_length
        self.hough_max_gap      = hough_max_gap
        self.history_len        = history_len
        self.confirm_frames     = confirm_frames
        self.angle_tol          = np.deg2rad(angle_tol_deg)
        self.dist_tol           = dist_tol_px
        self.sample_dist        = sample_dist
        self.n_samples          = n_samples
        self.green_frac_thresh  = green_frac_thresh
        self.bad_frac_thresh    = bad_frac_thresh
        self.hue_var_thresh     = hue_var_thresh
        self.oscillation_thresh = oscillation_thresh

        self._history = deque(maxlen=history_len)

    @staticmethod
    def _line_params(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(abs(dy), abs(dx))
        mid   = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return angle, mid

    def _lines_match(self, a1, m1, a2, m2):
        angle_diff = min(abs(a1 - a2), np.pi - abs(a1 - a2))
        dist = np.hypot(m1[0] - m2[0], m1[1] - m2[1])
        return angle_diff < self.angle_tol and dist < self.dist_tol

    def _extend_line(self, p1, p2, img_width):
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1:
            return (x1, 0), (x1, 9999)
        slope   = (y2 - y1) / (x2 - x1)
        y_left  = int(y1 + slope * (0 - x1))
        y_right = int(y1 + slope * (img_width - 1 - x1))
        return (0, y_left), (img_width - 1, y_right)

    def _sample_sides(self, p1, p2, mask, hsv):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            return 0, 0, 999, 999

        px, py = -dy / length, dx / length
        h, w   = mask.shape
        left_hits, right_hits = [], []
        left_hues, right_hues = [], []

        for i in range(self.n_samples):
            t  = i / max(self.n_samples - 1, 1)
            lx = int(x1 + t * dx)
            ly = int(y1 + t * dy)

            lx_l, ly_l = int(lx + px * self.sample_dist), int(ly + py * self.sample_dist)
            lx_r, ly_r = int(lx - px * self.sample_dist), int(ly - py * self.sample_dist)

            if 0 <= ly_l < h and 0 <= lx_l < w:
                left_hits.append(1 if mask[ly_l, lx_l] > 0 else 0)
                left_hues.append(int(hsv[ly_l, lx_l, 0]))
            if 0 <= ly_r < h and 0 <= lx_r < w:
                right_hits.append(1 if mask[ly_r, lx_r] > 0 else 0)
                right_hues.append(int(hsv[ly_r, lx_r, 0]))

        left_frac  = np.mean(left_hits)  if left_hits  else 0
        right_frac = np.mean(right_hits) if right_hits else 0
        left_std   = np.std(left_hues)   if left_hues  else 999
        right_std  = np.std(right_hues)  if right_hues else 999

        return left_frac, right_frac, left_std, right_std

    def _green_side_oscillates(self, p1, p2, mask):
        """
        Sample the dominant (greener) side along the line.
        Returns mean abs diff between consecutive samples.
        High (~0.5) = oscillating green patches = mat.
        Low (~0.0)  = solid green = real ground.
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            return 0.0

        px, py = -dy / length, dx / length
        h, w = mask.shape
        left_vals, right_vals = [], []

        for i in range(self.n_samples):
            t  = i / max(self.n_samples - 1, 1)
            lx = int(x1 + t * dx)
            ly = int(y1 + t * dy)

            lx_l, ly_l = int(lx + px * self.sample_dist), int(ly + py * self.sample_dist)
            lx_r, ly_r = int(lx - px * self.sample_dist), int(ly - py * self.sample_dist)

            if 0 <= ly_l < h and 0 <= lx_l < w:
                left_vals.append(1.0 if mask[ly_l, lx_l] > 0 else 0.0)
            if 0 <= ly_r < h and 0 <= lx_r < w:
                right_vals.append(1.0 if mask[ly_r, lx_r] > 0 else 0.0)

        # Check the greener side
        dominant = left_vals if np.mean(left_vals) >= np.mean(right_vals) else right_vals
        if len(dominant) < 2:
            return 0.0

        diffs = [abs(dominant[i+1] - dominant[i]) for i in range(len(dominant) - 1)]
        return float(np.mean(diffs))

    def _is_mat_edge(self, p1, p2, mask, hsv, bad_mask):
        left_frac, right_frac, left_std, right_std = self._sample_sides(p1, p2, mask, hsv)


        left_frac_bad, right_frac_bad, left_std_bad, right_std_bad = self._sample_sides(p1, p2, bad_mask, hsv)
        left_green = left_frac > self.green_frac_thresh
        right_green = right_frac > self.green_frac_thresh
        left_bad  = left_frac_bad > self.bad_frac_thresh
        right_bad = right_frac_bad > self.bad_frac_thresh
        # Check 1: green on both sides → mat interior line
        if left_green and right_green:
            return "both-green"
        if left_green and not right_bad:
            return "not-bad-R"
        if right_green and not left_bad:
            return "not-bad-L"


        # Check 2: high hue variance on green side → printed mat pattern
        # if left_green and left_std > self.hue_var_thresh:
        #     return f"high-var-L({left_std:.0f})"
        # if right_green and right_std > self.hue_var_thresh:
        #     return f"high-var-R({right_std:.0f})"
        # if left_bad and left_std_bad > self.hue_var_thresh_bad:
        #     return f"high-var-bad-L({left_std_bad:.0f})"
        # if right_bad and right_std_bad > self.hue_var_thresh_bad:
        #     return f"high-var-bad-R({right_std_bad:.0f})"
        #
        # # Check 3: green side oscillates along line → mat stripes
        # osc = self._green_side_oscillates(p1, p2, mask)
        # if osc > self.oscillation_thresh:
        #     return f"oscillating({osc:.2f})"
        # # bad side oscillation check
        # osc_bad = self._green_side_oscillates(p1, p2, bad_mask)
        # if osc_bad > self.oscillation_thresh:
        #     return f"oscillating-bad({osc_bad:.2f})"



        return None

    @staticmethod
    def _draw_dashed_line(img, p1, p2, color, dash_len=12):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            return
        steps = max(int(length / dash_len), 1)
        for i in range(0, steps, 2):
            t0  = i / steps
            t1  = min((i + 1) / steps, 1.0)
            pt0 = (int(x1 + t0 * dx), int(y1 + t0 * dy))
            pt1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
            cv2.line(img, pt0, pt1, color, 2)


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

        # Clean up with morphology
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask


    def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
        """
        Main detection function. Steps:
        1. Blur + convert to HSV.
        2. Create green mask
        3. create non-uniform mask
        4. create not green mask
        5. clean up masks with morphology
        6. combine masks (good = non-uniform OR green, bad = not-green AND NOT non-uniform)
        """
        h_img, w_img = bgr.shape[:2]

        blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, self.hsv_lower_green, self.hsv_upper_green)
        mask_non_uniform = self.mask_non_uniform(bgr)
        mask_not_green = cv2.inRange(hsv, self.hsv_lower_not_green, self.hsv_upper_not_green)

        closekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        openkernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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
        overlay[result.mask_bad > 0, 2] +=255
        overlay[result.mask_good > 0, 1] +=255
        out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
        h_img, w_img = bgr.shape[:2]
        if result.centroid is not None:
            frame_center = (w_img // 2, h_img // 2)
            cv2.arrowedLine(out, frame_center, result.centroid, (255, 255, 0), 2, tipLength=0.3)
        return out

