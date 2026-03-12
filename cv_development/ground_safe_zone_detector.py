import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SafeZoneResult:
    cell_scores: np.ndarray        # shape (grid_rows, grid_cols), 0=safe, 1=danger
    safe_cols: list                # column indices where ALL cells score below threshold
    obstacle_centroid_x: float     # [-1, 1], weighted X of danger; 0 = center
    obstacle_score: float          # EMA-smoothed max cell score
    obstacle_size: float           # fraction of analysis zone that is non-green
    safe_flag: bool                # True if obstacle_score < score_threshold
    suggested_turn: float          # [-1, 1], steer AWAY from danger centroid


@dataclass
class GroundEdgeResult:
    mask: np.ndarray
    edge: np.ndarray
    contours: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    confirmed_lines: list = field(default_factory=list)  # (ep1, ep2, orig_p1, orig_p2)
    rejected_lines: list = field(default_factory=list)   # (ep1, ep2, reason)
    safe_zone: SafeZoneResult = None


class GroundEdgeDetector:
    def __init__(
        self,
        # --- HSV tuning ---
        hsv_lower: tuple = (18, 17, 124),
        hsv_upper: tuple = (76, 153, 255),
        # --- Hough ---
        hough_threshold: int = 54,
        hough_min_length: int = 80,
        hough_max_gap: int = 36,
        # --- Preprocessing ---
        blur_ksize: int = 9,
        morph_ksize: int = 5,
        min_area: int = 500,
        # --- Temporal tracking ---
        history_len: int = 5,
        confirm_frames: int = 2,
        angle_tol_deg: float = 12.0,
        dist_tol_px: float = 34.0,
        # --- Mat rejection sampling ---
        sample_dist: int = 20,
        n_samples: int = 20,
        green_frac_thresh: float = 0.3,
        hue_var_thresh: float = 15.0,
        oscillation_thresh: float = 0.25,
        # --- Safe zone grid ---
        grid_cols: int = 3,
        grid_rows: int = 2,
        analysis_zone_frac: float = 2/3,  # fraction of image height to analyse
        w_ng: float = 0.50,               # weight: non-green area
        w_ed: float = 0.30,               # weight: edge density
        w_ll: float = 0.20,               # weight: confirmed line presence
        score_threshold: float = 0.25,
        ema_alpha: float = 0.3,
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
        self.grid_cols          = grid_cols
        self.grid_rows          = grid_rows
        self.analysis_zone_frac = analysis_zone_frac
        self.w_ng               = w_ng
        self.w_ed               = w_ed
        self.w_ll               = w_ll
        self.score_threshold    = score_threshold
        self.ema_alpha          = ema_alpha

        self._history     = deque(maxlen=history_len)
        self._ema_score    = None
        self._ema_centroid = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

            lx_l = int(lx + px * self.sample_dist)
            ly_l = int(ly + py * self.sample_dist)
            lx_r = int(lx - px * self.sample_dist)
            ly_r = int(ly - py * self.sample_dist)

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

            lx_l = int(lx + px * self.sample_dist)
            ly_l = int(ly + py * self.sample_dist)
            lx_r = int(lx - px * self.sample_dist)
            ly_r = int(ly - py * self.sample_dist)

            if 0 <= ly_l < h and 0 <= lx_l < w:
                left_vals.append(1.0 if mask[ly_l, lx_l] > 0 else 0.0)
            if 0 <= ly_r < h and 0 <= lx_r < w:
                right_vals.append(1.0 if mask[ly_r, lx_r] > 0 else 0.0)

        dominant = left_vals if np.mean(left_vals) >= np.mean(right_vals) else right_vals
        if len(dominant) < 2:
            return 0.0

        diffs = [abs(dominant[i+1] - dominant[i]) for i in range(len(dominant) - 1)]
        return float(np.mean(diffs))

    def _is_mat_edge(self, p1, p2, mask, hsv):
        left_frac, right_frac, left_std, right_std = self._sample_sides(p1, p2, mask, hsv)

        if left_frac > self.green_frac_thresh and right_frac > self.green_frac_thresh:
            return "both-green"
        if left_frac > self.green_frac_thresh and left_std > self.hue_var_thresh:
            return f"high-var-L({left_std:.0f})"
        if right_frac > self.green_frac_thresh and right_std > self.hue_var_thresh:
            return f"high-var-R({right_std:.0f})"

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

    # ------------------------------------------------------------------
    # Safe zone grid
    # ------------------------------------------------------------------

    def _compute_safe_zone(
        self,
        mask: np.ndarray,
        edge: np.ndarray,
        confirmed_lines: list,
        img_shape: tuple,
    ) -> SafeZoneResult:
        h_img, w_img = img_shape[:2]
        zone_h = int(h_img * self.analysis_zone_frac)
        cell_w = w_img  // self.grid_cols
        cell_h = zone_h // self.grid_rows
        diag   = np.hypot(w_img, zone_h)

        cell_scores = np.zeros((self.grid_rows, self.grid_cols), dtype=float)

        score_cx_sum     = 0.0
        score_weight_sum = 0.0

        for gr in range(self.grid_rows):
            # Rows closer to the bottom of the analysis zone (higher gr index)
            # are nearer to the drone → weight more heavily
            row_weight = 1.0 + 0.5 * gr

            for gc in range(self.grid_cols):
                x0 = gc * cell_w
                y0 = gr * cell_h
                x1 = w_img  if gc == self.grid_cols - 1 else (gc + 1) * cell_w
                y1 = zone_h if gr == self.grid_rows - 1 else (gr + 1) * cell_h
                cell_area = (x1 - x0) * (y1 - y0)
                if cell_area == 0:
                    continue

                # Non-green fraction
                green_in_cell = int(np.count_nonzero(mask[y0:y1, x0:x1]))
                non_green_frac = 1.0 - green_in_cell / cell_area

                # Edge density
                edge_in_cell  = int(np.count_nonzero(edge[y0:y1, x0:x1]))
                edge_density  = edge_in_cell / cell_area

                # Confirmed boundary line presence (normalised max length)
                max_line_norm = 0.0
                for ep1, ep2, orig_p1, orig_p2 in confirmed_lines:
                    mx = (orig_p1[0] + orig_p2[0]) / 2
                    my = (orig_p1[1] + orig_p2[1]) / 2
                    if x0 <= mx < x1 and y0 <= my < y1:
                        ll = np.hypot(orig_p2[0] - orig_p1[0], orig_p2[1] - orig_p1[1])
                        norm = min(ll / diag, 1.0)
                        if norm > max_line_norm:
                            max_line_norm = norm

                sc = float(np.clip(
                    self.w_ng * non_green_frac +
                    self.w_ed * edge_density   +
                    self.w_ll * max_line_norm,
                    0.0, 1.0
                ))
                cell_scores[gr, gc] = sc

                cell_cx = ((x0 + x1) / 2.0 / w_img) * 2.0 - 1.0
                score_cx_sum     += sc * row_weight * cell_cx
                score_weight_sum += sc * row_weight

        # Global danger score = worst cell
        raw_score = float(np.max(cell_scores))
        raw_cx    = float(np.clip(
            score_cx_sum / score_weight_sum if score_weight_sum > 1e-6 else 0.0,
            -1.0, 1.0
        ))

        # EMA smoothing
        if self._ema_score is None:
            self._ema_score    = raw_score
            self._ema_centroid = raw_cx
        else:
            a = self.ema_alpha
            self._ema_score    = a * raw_score + (1 - a) * self._ema_score
            self._ema_centroid = a * raw_cx    + (1 - a) * self._ema_centroid

        # Obstacle size: non-green fraction over the whole analysis zone
        zone_total  = w_img * zone_h
        zone_green  = int(np.count_nonzero(mask[:zone_h, :]))
        obstacle_size = (zone_total - zone_green) / zone_total

        # Safe columns: every row in this column must be below threshold
        safe_cols = [
            gc for gc in range(self.grid_cols)
            if np.all(cell_scores[:, gc] < self.score_threshold)
        ]

        return SafeZoneResult(
            cell_scores       = cell_scores,
            safe_cols         = safe_cols,
            obstacle_centroid_x = self._ema_centroid,
            obstacle_score    = self._ema_score,
            obstacle_size     = obstacle_size,
            safe_flag         = self._ema_score < self.score_threshold,
            suggested_turn    = float(np.clip(-self._ema_centroid, -1.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        safe_zone = self._compute_safe_zone(mask, edge, confirmed_lines, bgr.shape)

        return GroundEdgeResult(
            mask=mask,
            edge=edge,
            contours=contours,
            lines=[(p1, p2) for _, _, p1, p2 in frame_lines],
            confirmed_lines=confirmed_lines,
            rejected_lines=rejected_lines,
            safe_zone=safe_zone,
        )

    def draw(self, bgr: np.ndarray, result: GroundEdgeResult) -> np.ndarray:
        overlay = bgr.copy()
        overlay[result.mask > 0] = [0, 200, 0]
        out = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

        # Raw Hough — thin cyan
        for p1, p2 in result.lines:
            cv2.line(out, p1, p2, (0, 180, 180), 1)

        # Confirmed boundaries
        for ep1, ep2, orig_p1, orig_p2 in result.confirmed_lines:
            self._draw_dashed_line(out, ep1, ep2, (0, 255, 255), dash_len=10)
            cv2.line(out, orig_p1, orig_p2, (0, 255, 0), 3)
            cv2.circle(out, orig_p1, 4, (0, 255, 0), -1)
            cv2.circle(out, orig_p2, 4, (0, 255, 0), -1)

        # Mat-rejected lines — red dashed
        for ep1, ep2, reason in result.rejected_lines:
            self._draw_dashed_line(out, ep1, ep2, (0, 0, 255), dash_len=12)
            mx = (ep1[0] + ep2[0]) // 2
            my = (ep1[1] + ep2[1]) // 2
            cv2.putText(out, reason, (mx, my - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Safe zone grid overlay
        if result.safe_zone is not None:
            self._draw_safe_zone(out, result.safe_zone)

        return out

    def _draw_safe_zone(self, img: np.ndarray, sz: SafeZoneResult) -> None:
        h_img, w_img = img.shape[:2]
        zone_h = int(h_img * self.analysis_zone_frac)
        cell_w = w_img  // self.grid_cols
        cell_h = zone_h // self.grid_rows

        for gr in range(self.grid_rows):
            for gc in range(self.grid_cols):
                x0 = gc * cell_w
                y0 = gr * cell_h
                x1 = w_img  if gc == self.grid_cols - 1 else (gc + 1) * cell_w
                y1 = zone_h if gr == self.grid_rows - 1 else (gr + 1) * cell_h

                sc = sz.cell_scores[gr, gc]
                # green → yellow → red tint based on score
                r  = int(np.clip(sc * 2 * 255, 0, 255))
                g  = int(np.clip((1 - sc) * 2 * 255, 0, 255))
                tint = np.full((y1 - y0, x1 - x0, 3), (0, g, r), dtype=np.uint8)
                cv2.addWeighted(img[y0:y1, x0:x1], 0.75, tint, 0.25, 0,
                                img[y0:y1, x0:x1])

                border_col = (0, 255, 0) if gc in sz.safe_cols else (0, 80, 200)
                cv2.rectangle(img, (x0, y0), (x1, y1), border_col, 1)
                cv2.putText(img, f"{sc:.2f}", (x0 + 3, y0 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, f"{sc:.2f}", (x0 + 3, y0 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Danger centroid crosshair
        cx_px = int(((sz.obstacle_centroid_x + 1) / 2) * w_img)
        cx_px = int(np.clip(cx_px, 0, w_img - 1))
        cy_px = zone_h // 2
        cv2.line(img, (cx_px - 15, cy_px), (cx_px + 15, cy_px), (0, 255, 255), 2)
        cv2.line(img, (cx_px, cy_px - 15), (cx_px, cy_px + 15), (0, 255, 255), 2)
        cv2.circle(img, (cx_px, cy_px), 12, (0, 255, 255), 2)

        # Zone boundary
        for x in range(0, w_img, 8):
            cv2.line(img, (x, zone_h), (min(x + 4, w_img), zone_h), (255, 255, 255), 1)

        # HUD
        status     = "SAFE" if sz.safe_flag else "DANGER"
        status_col = (0, 200, 0) if sz.safe_flag else (0, 0, 255)
        h_txt = h_img - 8
        hud   = (f"SCORE {sz.obstacle_score:.2f}  "
                 f"CX {sz.obstacle_centroid_x:.2f}  "
                 f"TURN {sz.suggested_turn:+.2f}  "
                 f"SIZE {sz.obstacle_size:.2f}")
        cv2.putText(img, hud,    (4, h_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
        cv2.putText(img, hud,    (4, h_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(img, status, (w_img - 75, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(img, status, (w_img - 75, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2)
