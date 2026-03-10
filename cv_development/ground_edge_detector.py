# THIS IS GOOD, BUT NO MAT DETECTION

# import cv2
# import numpy as np
# from dataclasses import dataclass, field
# from collections import deque


# @dataclass
# class GroundEdgeResult:
#     mask: np.ndarray
#     edge: np.ndarray
#     contours: list = field(default_factory=list)
#     lines: list = field(default_factory=list)       # raw Hough segments this frame
#     confirmed_lines: list = field(default_factory=list)  # temporally stable lines


# class GroundEdgeDetector:
#     def __init__(
#         self,
        
#         # # Hough — strict, few false positives
#         # hough_threshold: int = 100,
#         # hough_min_length: int = 80,
#         # hough_max_gap: int = 20,
#         # # Temporal tracking
#         # history_len: int = 5,        # frames to keep in memory
#         # confirm_frames: int = 1,     # line must appear in this many of last N frames
#         # angle_tol_deg: float = 12.0, # two lines are "the same" if angle within this
#         # dist_tol_px: float = 34.0,   # and their midpoints are within this distance

#         # TUNED PARAMS
#         hsv_lower: tuple = (18, 17, 124),
#         hsv_upper: tuple = (76, 153, 255),

#         hough_threshold: int = 54,
#         hough_min_length: int = 80,
#         hough_max_gap: int = 36,
#         blur_ksize: int = 9,
#         morph_ksize: int = 5,
#         min_area: int = 500,
#         # Temporal tracking
#         history_len: int = 5,        # frames to keep in memory
#         confirm_frames: int = 2,     # line must appear in this many of last N frames
#         angle_tol_deg: float = 12.0, # two lines are "the same" if angle within this
#         dist_tol_px: float = 34.0,   # and their midpoints are within this distance
#     ):
#         self.hsv_lower        = np.array(hsv_lower)
#         self.hsv_upper        = np.array(hsv_upper)
#         self.blur_ksize       = blur_ksize
#         self.morph_ksize      = morph_ksize
#         self.min_area         = min_area
#         self.hough_threshold  = hough_threshold
#         self.hough_min_length = hough_min_length
#         self.hough_max_gap    = hough_max_gap
#         self.history_len      = history_len
#         self.confirm_frames   = confirm_frames
#         self.angle_tol        = np.deg2rad(angle_tol_deg)
#         self.dist_tol         = dist_tol_px

#         # Ring buffer: each entry is a list of (angle, midpoint, p1, p2) for that frame
#         self._history = deque(maxlen=history_len)

#     @staticmethod
#     def _line_params(p1, p2):
#         """Return (angle in [0,pi), midpoint) for a line segment."""
#         dx = p2[0] - p1[0]
#         dy = p2[1] - p1[1]
#         angle = np.arctan2(abs(dy), abs(dx))  # 0..pi/2, fold to be rotation-invariant
#         mid   = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
#         return angle, mid

#     def _lines_match(self, a1, m1, a2, m2):
#         """True if two lines have similar angle and midpoint."""
#         angle_diff = abs(a1 - a2)
#         angle_diff = min(angle_diff, np.pi - angle_diff)  # handle 0/180 wrap
#         dist = np.hypot(m1[0] - m2[0], m1[1] - m2[1])
#         return angle_diff < self.angle_tol and dist < self.dist_tol

#     def _extend_line(self, p1, p2, img_width):
#         """Extend line segment to full image width."""
#         x1, y1 = p1
#         x2, y2 = p2
#         if x2 == x1:  # vertical line
#             return (x1, 0), (x1, 9999)
#         slope = (y2 - y1) / (x2 - x1)
#         y_left  = int(y1 + slope * (0 - x1))
#         y_right = int(y1 + slope * (img_width - 1 - x1))
#         return (0, y_left), (img_width - 1, y_right)

#     def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
#         h_img, w_img = bgr.shape[:2]

#         blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
#         hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
#                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

#         # Boundary pixels only
#         eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
#         boundary = cv2.subtract(mask, eroded)

#         raw_lines = cv2.HoughLinesP(
#             boundary, rho=1, theta=np.pi / 180,
#             threshold=self.hough_threshold,
#             minLineLength=self.hough_min_length,
#             maxLineGap=self.hough_max_gap,
#         )

#         # Parse this frame's lines
#         frame_lines = []
#         if raw_lines is not None:
#             for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
#                 p1, p2 = (x1, y1), (x2, y2)
#                 angle, mid = self._line_params(p1, p2)
#                 frame_lines.append((angle, mid, p1, p2))

#         self._history.append(frame_lines)

#         # --- Temporal confirmation ---
#         # For each line in the current frame, count how many past frames
#         # contain a matching line
#         confirmed_lines = []
#         for angle, mid, p1, p2 in frame_lines:
#             match_count = 0
#             for past_frame in self._history:
#                 for pa, pm, _, _ in past_frame:
#                     if self._lines_match(angle, mid, pa, pm):
#                         match_count += 1
#                         break  # only count once per frame
#             if match_count >= self.confirm_frames:
#                 ep1, ep2 = self._extend_line(p1, p2, w_img)
#                 confirmed_lines.append((ep1, ep2))

#         # Edge image: raw Hough segments
#         edge = np.zeros_like(mask)
#         for _, _, p1, p2 in frame_lines:
#             cv2.line(edge, p1, p2, 255, 2)

#         return GroundEdgeResult(
#             mask=mask, edge=edge,
#             contours=contours,
#             lines=[(p1, p2) for _, _, p1, p2 in frame_lines],
#             confirmed_lines=confirmed_lines,
#         )

#     def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
#         overlay = bgr.copy()
#         overlay[result.mask > 0] = [0, 200, 0]
#         out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

#         # Raw Hough lines — dim yellow, so you can see what's being tracked
#         for p1, p2 in result.lines:
#             cv2.line(out, p1, p2, (0, 180, 180), 1)

#         # Confirmed + extended lines — bright yellow, thick
#         for p1, p2 in result.confirmed_lines:
#             cv2.line(out, p1, p2, (0, 255, 255), 2)

#         return out





# TRYING MAT DETECTION AS WELL


# import cv2
# import numpy as np
# from dataclasses import dataclass, field
# from collections import deque


# @dataclass
# class GroundEdgeResult:
#     mask: np.ndarray
#     edge: np.ndarray
#     contours: list = field(default_factory=list)
#     lines: list = field(default_factory=list)
#     confirmed_lines: list = field(default_factory=list)  # (ep1, ep2, orig_p1, orig_p2)
#     rejected_lines: list = field(default_factory=list)   # (ep1, ep2, reason)


# class GroundEdgeDetector:
#     def __init__(
#         self,
#         hsv_lower: tuple = (18, 17, 124),
#         hsv_upper: tuple = (76, 153, 255),
#         hough_threshold: int = 54,
#         hough_min_length: int = 80,
#         hough_max_gap: int = 36,
#         blur_ksize: int = 9,
#         morph_ksize: int = 5,
#         min_area: int = 500,
#         history_len: int = 5,
#         confirm_frames: int = 2,
#         angle_tol_deg: float = 12.0,
#         dist_tol_px: float = 34.0,
#         sample_dist: int = 15,
#         n_samples: int = 10,
#         green_frac_thresh: float = 0.3,
#         hue_var_thresh: float = 15.0,
#     ):
#         self.hsv_lower         = np.array(hsv_lower)
#         self.hsv_upper         = np.array(hsv_upper)
#         self.blur_ksize        = blur_ksize
#         self.morph_ksize       = morph_ksize
#         self.min_area          = min_area
#         self.hough_threshold   = hough_threshold
#         self.hough_min_length  = hough_min_length
#         self.hough_max_gap     = hough_max_gap
#         self.history_len       = history_len
#         self.confirm_frames    = confirm_frames
#         self.angle_tol         = np.deg2rad(angle_tol_deg)
#         self.dist_tol          = dist_tol_px
#         self.sample_dist       = sample_dist
#         self.n_samples         = n_samples
#         self.green_frac_thresh = green_frac_thresh
#         self.hue_var_thresh    = hue_var_thresh

#         self._history = deque(maxlen=history_len)

#     @staticmethod
#     def _line_params(p1, p2):
#         dx = p2[0] - p1[0]
#         dy = p2[1] - p1[1]
#         angle = np.arctan2(abs(dy), abs(dx))
#         mid   = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
#         return angle, mid

#     def _lines_match(self, a1, m1, a2, m2):
#         angle_diff = min(abs(a1 - a2), np.pi - abs(a1 - a2))
#         dist = np.hypot(m1[0] - m2[0], m1[1] - m2[1])
#         return angle_diff < self.angle_tol and dist < self.dist_tol

#     def _extend_line(self, p1, p2, img_width):
#         x1, y1 = p1
#         x2, y2 = p2
#         if x2 == x1:
#             return (x1, 0), (x1, 9999)
#         slope   = (y2 - y1) / (x2 - x1)
#         y_left  = int(y1 + slope * (0 - x1))
#         y_right = int(y1 + slope * (img_width - 1 - x1))
#         return (0, y_left), (img_width - 1, y_right)

#     def _sample_sides(self, p1, p2, mask, hsv):
#         x1, y1 = p1
#         x2, y2 = p2
#         dx, dy = x2 - x1, y2 - y1
#         length = np.hypot(dx, dy)
#         if length == 0:
#             return 0, 0, 999, 999

#         px, py = -dy / length, dx / length
#         h, w   = mask.shape
#         left_hits, right_hits = [], []
#         left_hues, right_hues = [], []

#         for i in range(self.n_samples):
#             t  = i / max(self.n_samples - 1, 1)
#             lx = int(x1 + t * dx)
#             ly = int(y1 + t * dy)

#             lx_l, ly_l = int(lx + px * self.sample_dist), int(ly + py * self.sample_dist)
#             lx_r, ly_r = int(lx - px * self.sample_dist), int(ly - py * self.sample_dist)

#             if 0 <= ly_l < h and 0 <= lx_l < w:
#                 left_hits.append(1 if mask[ly_l, lx_l] > 0 else 0)
#                 left_hues.append(int(hsv[ly_l, lx_l, 0]))
#             if 0 <= ly_r < h and 0 <= lx_r < w:
#                 right_hits.append(1 if mask[ly_r, lx_r] > 0 else 0)
#                 right_hues.append(int(hsv[ly_r, lx_r, 0]))

#         left_frac  = np.mean(left_hits)  if left_hits  else 0
#         right_frac = np.mean(right_hits) if right_hits else 0
#         left_std   = np.std(left_hues)   if left_hues  else 999
#         right_std  = np.std(right_hues)  if right_hues else 999

#         return left_frac, right_frac, left_std, right_std

#     def _is_mat_edge(self, p1, p2, mask, hsv):
#         left_frac, right_frac, left_std, right_std = self._sample_sides(p1, p2, mask, hsv)

#         if left_frac > self.green_frac_thresh and right_frac > self.green_frac_thresh:
#             return "both-green"
#         if left_frac > self.green_frac_thresh and left_std > self.hue_var_thresh:
#             return f"high-var-L({left_std:.0f})"
#         if right_frac > self.green_frac_thresh and right_std > self.hue_var_thresh:
#             return f"high-var-R({right_std:.0f})"

#         return None

#     @staticmethod
#     def _draw_dashed_line(img, p1, p2, color, dash_len=12):
#         x1, y1 = p1
#         x2, y2 = p2
#         dx, dy = x2 - x1, y2 - y1
#         length = np.hypot(dx, dy)
#         if length == 0:
#             return
#         steps = max(int(length / dash_len), 1)
#         for i in range(0, steps, 2):
#             t0  = i / steps
#             t1  = min((i + 1) / steps, 1.0)
#             pt0 = (int(x1 + t0 * dx), int(y1 + t0 * dy))
#             pt1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
#             cv2.line(img, pt0, pt1, color, 2)

#     def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
#         h_img, w_img = bgr.shape[:2]

#         blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
#         hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
#                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

#         eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
#         boundary = cv2.subtract(mask, eroded)

#         raw_lines = cv2.HoughLinesP(
#             boundary, rho=1, theta=np.pi / 180,
#             threshold=self.hough_threshold,
#             minLineLength=self.hough_min_length,
#             maxLineGap=self.hough_max_gap,
#         )

#         frame_lines = []
#         if raw_lines is not None:
#             for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
#                 p1, p2 = (x1, y1), (x2, y2)
#                 angle, mid = self._line_params(p1, p2)
#                 frame_lines.append((angle, mid, p1, p2))

#         self._history.append(frame_lines)

#         confirmed_lines = []
#         rejected_lines  = []

#         for angle, mid, p1, p2 in frame_lines:
#             match_count = sum(
#                 1 for past_frame in self._history
#                 if any(self._lines_match(angle, mid, pa, pm)
#                        for pa, pm, _, _ in past_frame)
#             )
#             if match_count < self.confirm_frames:
#                 continue

#             ep1, ep2 = self._extend_line(p1, p2, w_img)
#             reason   = self._is_mat_edge(p1, p2, mask, hsv)

#             if reason:
#                 rejected_lines.append((ep1, ep2, reason))
#             else:
#                 # Store extended line + original detected segment
#                 confirmed_lines.append((ep1, ep2, p1, p2))

#         edge = np.zeros_like(mask)
#         for _, _, p1, p2 in frame_lines:
#             cv2.line(edge, p1, p2, 255, 2)

#         return GroundEdgeResult(
#             mask=mask, edge=edge,
#             contours=contours,
#             lines=[(p1, p2) for _, _, p1, p2 in frame_lines],
#             confirmed_lines=confirmed_lines,
#             rejected_lines=rejected_lines,
#         )

#     def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
#         overlay = bgr.copy()
#         overlay[result.mask > 0] = [0, 200, 0]
#         out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

#         # Thin cyan — raw Hough, not yet confirmed
#         for p1, p2 in result.lines:
#             cv2.line(out, p1, p2, (0, 180, 180), 1)

#         # Confirmed real boundaries
#         for ep1, ep2, orig_p1, orig_p2 in result.confirmed_lines:
#             # Original detected segment — solid bright green, thick
#             cv2.line(out, orig_p1, orig_p2, (0, 255, 0), 3)
#             # Extended line — bright yellow, thin dashed
#             self._draw_dashed_line(out, ep1, ep2, (0, 255, 255), dash_len=10)
#             # Dots at original segment endpoints
#             cv2.circle(out, orig_p1, 4, (0, 255, 0), -1)
#             cv2.circle(out, orig_p2, 4, (0, 255, 0), -1)

#         # Red dashed — confirmed but rejected as mat
#         for ep1, ep2, reason in result.rejected_lines:
#             self._draw_dashed_line(out, ep1, ep2, (0, 0, 255), dash_len=12)
#             mx = (ep1[0] + ep2[0]) // 2
#             my = (ep1[1] + ep2[1]) // 2
#             cv2.putText(out, reason, (mx, my - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         return out







import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class GroundEdgeResult:
    mask: np.ndarray
    edge: np.ndarray
    contours: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    confirmed_lines: list = field(default_factory=list)  # (ep1, ep2, orig_p1, orig_p2)
    rejected_lines: list = field(default_factory=list)   # (ep1, ep2, reason)


class GroundEdgeDetector:
    def __init__(
        self,
        hsv_lower: tuple = (18, 17, 124),
        hsv_upper: tuple = (76, 153, 255),
        hough_threshold: int = 54,
        hough_min_length: int = 80,
        hough_max_gap: int = 36,
        blur_ksize: int = 9,
        morph_ksize: int = 5,
        min_area: int = 500,
        history_len: int = 5,
        confirm_frames: int = 2,
        angle_tol_deg: float = 12.0,
        dist_tol_px: float = 34.0,
        sample_dist: int = 20,
        n_samples: int = 20,
        green_frac_thresh: float = 0.3,
        hue_var_thresh: float = 15.0,
        oscillation_thresh: float = 0.25,
    ):
        self.hsv_lower          = np.array(hsv_lower)
        self.hsv_upper          = np.array(hsv_upper)
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

    def _is_mat_edge(self, p1, p2, mask, hsv):
        left_frac, right_frac, left_std, right_std = self._sample_sides(p1, p2, mask, hsv)

        # Check 1: green on both sides → mat interior line
        if left_frac > self.green_frac_thresh and right_frac > self.green_frac_thresh:
            return "both-green"

        # Check 2: high hue variance on green side → printed mat pattern
        if left_frac > self.green_frac_thresh and left_std > self.hue_var_thresh:
            return f"high-var-L({left_std:.0f})"
        if right_frac > self.green_frac_thresh and right_std > self.hue_var_thresh:
            return f"high-var-R({right_std:.0f})"

        # Check 3: green side oscillates along line → mat stripes
        osc = self._green_side_oscillates(p1, p2, mask)
        if osc > self.oscillation_thresh:
            return f"oscillating({osc:.2f})"

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

    def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
        h_img, w_img = bgr.shape[:2]

        blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

        eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = cv2.subtract(mask, eroded)

        raw_lines = cv2.HoughLinesP(
            boundary, rho=1, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        frame_lines = []
        if raw_lines is not None:
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
                p1, p2 = (x1, y1), (x2, y2)
                angle, mid = self._line_params(p1, p2)
                frame_lines.append((angle, mid, p1, p2))

        self._history.append(frame_lines)

        confirmed_lines = []
        rejected_lines  = []

        for angle, mid, p1, p2 in frame_lines:
            match_count = sum(
                1 for past_frame in self._history
                if any(self._lines_match(angle, mid, pa, pm)
                       for pa, pm, _, _ in past_frame)
            )
            if match_count < self.confirm_frames:
                continue

            ep1, ep2 = self._extend_line(p1, p2, w_img)
            reason   = self._is_mat_edge(p1, p2, mask, hsv)

            if reason:
                rejected_lines.append((ep1, ep2, reason))
            else:
                confirmed_lines.append((ep1, ep2, p1, p2))

        edge = np.zeros_like(mask)
        for _, _, p1, p2 in frame_lines:
            cv2.line(edge, p1, p2, 255, 2)

        return GroundEdgeResult(
            mask=mask, edge=edge,
            contours=contours,
            lines=[(p1, p2) for _, _, p1, p2 in frame_lines],
            confirmed_lines=confirmed_lines,
            rejected_lines=rejected_lines,
        )

    def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
        overlay = bgr.copy()
        overlay[result.mask > 0] = [0, 200, 0]
        out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

        # Thin cyan — raw Hough, not yet confirmed
        for p1, p2 in result.lines:
            cv2.line(out, p1, p2, (0, 180, 180), 1)

        # Confirmed real boundaries
        for ep1, ep2, orig_p1, orig_p2 in result.confirmed_lines:
            self._draw_dashed_line(out, ep1, ep2, (0, 255, 255), dash_len=10)
            cv2.line(out, orig_p1, orig_p2, (0, 255, 0), 3)
            cv2.circle(out, orig_p1, 4, (0, 255, 0), -1)
            cv2.circle(out, orig_p2, 4, (0, 255, 0), -1)

        # Red dashed — confirmed but rejected as mat
        for ep1, ep2, reason in result.rejected_lines:
            self._draw_dashed_line(out, ep1, ep2, (0, 0, 255), dash_len=12)
            mx = (ep1[0] + ep2[0]) // 2
            my = (ep1[1] + ep2[1]) // 2
            cv2.putText(out, reason, (mx, my - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return out

