# import cv2
# import numpy as np
# from dataclasses import dataclass, field


# @dataclass
# class GroundEdgeResult:
#     mask: np.ndarray
#     edge: np.ndarray
#     contours: list = field(default_factory=list)
#     lines: list = field(default_factory=list)   # fitted line segments for straight edges


# class GroundEdgeDetector:
#     def __init__(
#         self,
#         hsv_lower: tuple = (18, 17, 124),
#         hsv_upper: tuple = (76, 153, 255),
#         blur_ksize: int = 7,
#         morph_ksize: int = 5,
#         min_area: int = 500,
#         straightness_thresh: float = 0.15,  # max fraction of points far from fitted line
#         line_dist_thresh: float = 15.0,      # px — max distance from line to count as inlier
#     ):
#         self.hsv_lower           = np.array(hsv_lower)
#         self.hsv_upper           = np.array(hsv_upper)
#         self.blur_ksize          = blur_ksize
#         self.morph_ksize         = morph_ksize
#         self.min_area            = min_area
#         self.straightness_thresh = straightness_thresh
#         self.line_dist_thresh    = line_dist_thresh

#     def _fit_line(self, contour):
#         """Fit a line to contour points, return (vx,vy,x0,y0) and inlier fraction."""
#         pts = contour.reshape(-1, 2).astype(np.float32)
#         if len(pts) < 5:
#             return None, 0.0

#         vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

#         # Distance from each point to the line
#         dx = pts[:, 0] - x0
#         dy = pts[:, 1] - y0
#         dist = np.abs(dx * vy - dy * vx)   # cross product = perpendicular distance

#         inlier_frac = np.mean(dist < self.line_dist_thresh)
#         return (vx, vy, x0, y0), inlier_frac

#     def _line_endpoints(self, vx, vy, x0, y0, contour):
#         """Project contour points onto the line and return the two extreme endpoints."""
#         pts = contour.reshape(-1, 2).astype(np.float32)
#         t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
#         t_min, t_max = t.min(), t.max()
#         p1 = (int(x0 + t_min * vx), int(y0 + t_min * vy))
#         p2 = (int(x0 + t_max * vx), int(y0 + t_max * vy))
#         return p1, p2

#     def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
#         blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
#         hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
#         )
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
#                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

#         # --- Straightness filter ---
#         straight_contours = []
#         lines = []
#         for c in contours:
#             (vx, vy, x0, y0), inlier_frac = self._fit_line(c)
#             if inlier_frac >= (1.0 - self.straightness_thresh):
#                 straight_contours.append(c)
#                 p1, p2 = self._line_endpoints(vx, vy, x0, y0, c)
#                 lines.append((p1, p2))

#         edge = np.zeros_like(mask)
#         cv2.drawContours(edge, straight_contours, -1, 255, thickness=2)

#         return GroundEdgeResult(mask=mask, edge=edge,
#                                 contours=straight_contours, lines=lines)

#     def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
#         overlay = bgr.copy()
#         overlay[result.mask > 0] = [0, 200, 0]
#         out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

#         # Draw fitted straight lines in bright yellow (cleaner than raw contour edge)
#         for p1, p2 in result.lines:
#             cv2.line(out, p1, p2, (0, 255, 255), 2)

#         return out



# import cv2
# import numpy as np
# from dataclasses import dataclass, field


# @dataclass
# class GroundEdgeResult:
#     mask: np.ndarray
#     edge: np.ndarray
#     contours: list = field(default_factory=list)
#     lines: list = field(default_factory=list)  # list of (p1, p2) tuples


# class GroundEdgeDetector:
#     def __init__(
#         self,
#         hsv_lower: tuple = (18, 17, 124),
#         hsv_upper: tuple = (76, 153, 255),
#         blur_ksize: int = 7,
#         morph_ksize: int = 5,
#         min_area: int = 500,
#         hough_threshold: int = 80,      # votes needed to detect a line — raise to get fewer/stronger lines
#         hough_min_length: int = 60,     # minimum line length in px
#         hough_max_gap: int = 30,        # max gap between segments to still join them
#     ):
#         self.hsv_lower        = np.array(hsv_lower)
#         self.hsv_upper        = np.array(hsv_upper)
#         self.blur_ksize       = blur_ksize
#         self.morph_ksize      = morph_ksize
#         self.min_area         = min_area
#         self.hough_threshold  = hough_threshold
#         self.hough_min_length = hough_min_length
#         self.hough_max_gap    = hough_max_gap

#     def detect(self, bgr: np.ndarray) -> GroundEdgeResult:
#         blurred = cv2.GaussianBlur(bgr, (self.blur_ksize, self.blur_ksize), 0)
#         hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
#         )
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
#                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

#         # Contours for region info (count, area)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

#         # Extract ONLY the boundary pixels of the mask (not the interior)
#         # Erode mask and subtract → gives just the 1-px border
#         eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
#         boundary = cv2.subtract(mask, eroded)

#         # Hough on the boundary pixels — finds straight segments
#         raw_lines = cv2.HoughLinesP(
#             boundary,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=self.hough_threshold,
#             minLineLength=self.hough_min_length,
#             maxLineGap=self.hough_max_gap,
#         )

#         lines = []
#         edge  = np.zeros_like(mask)
#         if raw_lines is not None:
#             for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
#                 lines.append(((x1, y1), (x2, y2)))
#                 cv2.line(edge, (x1, y1), (x2, y2), 255, 2)

#         return GroundEdgeResult(mask=mask, edge=edge, contours=contours, lines=lines)

#     def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
#         overlay = bgr.copy()
#         overlay[result.mask > 0] = [0, 200, 0]
#         out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

#         for p1, p2 in result.lines:
#             cv2.line(out, p1, p2, (0, 255, 255), 2)

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
    lines: list = field(default_factory=list)       # raw Hough segments this frame
    confirmed_lines: list = field(default_factory=list)  # temporally stable lines


class GroundEdgeDetector:
    def __init__(
        self,
        
        # # Hough — strict, few false positives
        # hough_threshold: int = 100,
        # hough_min_length: int = 80,
        # hough_max_gap: int = 20,
        # # Temporal tracking
        # history_len: int = 5,        # frames to keep in memory
        # confirm_frames: int = 1,     # line must appear in this many of last N frames
        # angle_tol_deg: float = 12.0, # two lines are "the same" if angle within this
        # dist_tol_px: float = 34.0,   # and their midpoints are within this distance

        # TUNED PARAMS
        hsv_lower: tuple = (18, 17, 124),
        hsv_upper: tuple = (76, 153, 255),

        hough_threshold: int = 54,
        hough_min_length: int = 80,
        hough_max_gap: int = 36,
        blur_ksize: int = 9,
        morph_ksize: int = 5,
        min_area: int = 500,
        # Temporal tracking
        history_len: int = 5,        # frames to keep in memory
        confirm_frames: int = 2,     # line must appear in this many of last N frames
        angle_tol_deg: float = 12.0, # two lines are "the same" if angle within this
        dist_tol_px: float = 34.0,   # and their midpoints are within this distance
    ):
        self.hsv_lower        = np.array(hsv_lower)
        self.hsv_upper        = np.array(hsv_upper)
        self.blur_ksize       = blur_ksize
        self.morph_ksize      = morph_ksize
        self.min_area         = min_area
        self.hough_threshold  = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap    = hough_max_gap
        self.history_len      = history_len
        self.confirm_frames   = confirm_frames
        self.angle_tol        = np.deg2rad(angle_tol_deg)
        self.dist_tol         = dist_tol_px

        # Ring buffer: each entry is a list of (angle, midpoint, p1, p2) for that frame
        self._history = deque(maxlen=history_len)

    @staticmethod
    def _line_params(p1, p2):
        """Return (angle in [0,pi), midpoint) for a line segment."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(abs(dy), abs(dx))  # 0..pi/2, fold to be rotation-invariant
        mid   = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return angle, mid

    def _lines_match(self, a1, m1, a2, m2):
        """True if two lines have similar angle and midpoint."""
        angle_diff = abs(a1 - a2)
        angle_diff = min(angle_diff, np.pi - angle_diff)  # handle 0/180 wrap
        dist = np.hypot(m1[0] - m2[0], m1[1] - m2[1])
        return angle_diff < self.angle_tol and dist < self.dist_tol

    def _extend_line(self, p1, p2, img_width):
        """Extend line segment to full image width."""
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1:  # vertical line
            return (x1, 0), (x1, 9999)
        slope = (y2 - y1) / (x2 - x1)
        y_left  = int(y1 + slope * (0 - x1))
        y_right = int(y1 + slope * (img_width - 1 - x1))
        return (0, y_left), (img_width - 1, y_right)

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

        # Boundary pixels only
        eroded   = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = cv2.subtract(mask, eroded)

        raw_lines = cv2.HoughLinesP(
            boundary, rho=1, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        # Parse this frame's lines
        frame_lines = []
        if raw_lines is not None:
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
                p1, p2 = (x1, y1), (x2, y2)
                angle, mid = self._line_params(p1, p2)
                frame_lines.append((angle, mid, p1, p2))

        self._history.append(frame_lines)

        # --- Temporal confirmation ---
        # For each line in the current frame, count how many past frames
        # contain a matching line
        confirmed_lines = []
        for angle, mid, p1, p2 in frame_lines:
            match_count = 0
            for past_frame in self._history:
                for pa, pm, _, _ in past_frame:
                    if self._lines_match(angle, mid, pa, pm):
                        match_count += 1
                        break  # only count once per frame
            if match_count >= self.confirm_frames:
                ep1, ep2 = self._extend_line(p1, p2, w_img)
                confirmed_lines.append((ep1, ep2))

        # Edge image: raw Hough segments
        edge = np.zeros_like(mask)
        for _, _, p1, p2 in frame_lines:
            cv2.line(edge, p1, p2, 255, 2)

        return GroundEdgeResult(
            mask=mask, edge=edge,
            contours=contours,
            lines=[(p1, p2) for _, _, p1, p2 in frame_lines],
            confirmed_lines=confirmed_lines,
        )

    def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
        overlay = bgr.copy()
        overlay[result.mask > 0] = [0, 200, 0]
        out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

        # Raw Hough lines — dim yellow, so you can see what's being tracked
        for p1, p2 in result.lines:
            cv2.line(out, p1, p2, (0, 180, 180), 1)

        # Confirmed + extended lines — bright yellow, thick
        for p1, p2 in result.confirmed_lines:
            cv2.line(out, p1, p2, (0, 255, 255), 2)

        return out
