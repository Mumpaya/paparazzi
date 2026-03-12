# import cv2
# import numpy as np
# from dataclasses import dataclass
# from typing import Tuple, List, Optional


# @dataclass
# class PoleDetection:
#     x: int
#     y: int
#     w: int
#     h: int
#     cx: int
#     cy: int
#     area: int
#     confidence: float


# @dataclass
# class ObstacleResult:
#     preprocessed: np.ndarray
#     roi_top: int
#     roi_bottom: int
#     roi_frame: np.ndarray
#     clahe_applied: bool = True
#     flow_points_prev: Optional[np.ndarray] = None
#     flow_points_next: Optional[np.ndarray] = None
#     flow_vectors: Optional[np.ndarray] = None
#     flow_vectors_comp: Optional[np.ndarray] = None
#     hough_boxes: Optional[List[Tuple]] = None
#     hough_scores: Optional[np.ndarray] = None
#     orange_boxes: Optional[List] = None
#     orange_mask: Optional[np.ndarray] = None
#     orange_scores: Optional[np.ndarray] = None
#     ground_mask: Optional[np.ndarray] = None
#     ground_green_scores: Optional[np.ndarray] = None
#     ground_obstacle_scores: Optional[np.ndarray] = None
#     flow_scores: Optional[np.ndarray] = None
#     fused_scores: Optional[np.ndarray] = None
#     obstacle_cols: Optional[List[int]] = None
#     gap_center_x: Optional[int] = None
#     n_cols: int = 22


# class ObstacleDetector:
#     """
#     Stage 1 — Rotate 90° CCW + CLAHE + ROI crop
#     Stage 2 — Sparse LK optical flow → column voting
#     Stage 3 — Mean flow subtraction (ego-motion compensation)
#     Stage 4 — Hough vertical line detector → pole boxes → column scores
#     Stage 5 — Orange HSV detector → column scores
#     Stage 6 — Green ground mask: missing-green columns → obstacle scores
#     Fusion  — fused = flow*w_f + hough*w_h + orange*w_o + ground*w_g
#     """

#     def __init__(
#         self,
#         roi_top_frac: float = 0.10,
#         roi_bottom_frac: float = 0.90,
#         clahe_clip_limit: float = 4.0,
#         clahe_tile_grid: Tuple[int, int] = (8, 8),
#         n_cols: int = 22,
#         flow_threshold: float = 0.7,
#         iir_alpha: float = 0.35,
#         max_corners: int = 150,
#         min_feature_dist: int = 4,
#         refresh_interval: int = 8,
#         flow_weight: float = 1.5,
#         hough_weight: float = 1.5,
#         hough_blur_kernel: int = 5,
#         hough_canny_low: int = 50,
#         hough_canny_high: int = 150,
#         hough_threshold: int = 30,
#         hough_min_length: int = 40,
#         hough_max_gap: int = 100,
#         hough_vertical_tol_deg: float = 20.0,
#         hough_cluster_gap: int = 35,
#         hough_max_box_width: int = 200,
#         hough_min_aspect: float = 1.5,
#         hough_max_aspect: float = 8.0,
#         hough_border_margin: int = 15,
#         hough_max_interior_edge_density: float = 0.15,
#         orange_weight: float = 2.5,
#         hsv_lower: Tuple = (0, 40, 50),
#         hsv_upper: Tuple = (25, 255, 255),
#         orange_min_area: int = 300,
#         orange_min_aspect: float = 1.5,
#         orange_min_confidence: float = 0.2,
#         orange_morph_kernel: Tuple[int, int] = (5, 5),
#         ground_weight: float = 2.0,
#         ground_green_lower: Tuple = (18, 17, 124),
#         ground_green_upper: Tuple = (76, 153, 255),
#         ground_morph_kernel: Tuple[int, int] = (7, 7),
#         ground_min_mean_green: float = 0.05,
#     ):
#         self.roi_top_frac    = roi_top_frac
#         self.roi_bottom_frac = roi_bottom_frac
#         self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
#                                      tileGridSize=clahe_tile_grid)
#         self.n_cols           = n_cols
#         self.flow_threshold   = flow_threshold
#         self.iir_alpha        = iir_alpha
#         self.max_corners      = max_corners
#         self.min_feature_dist = min_feature_dist
#         self.refresh_interval = refresh_interval
#         self.flow_weight      = flow_weight

#         self.lk_params = dict(
#             winSize=(21, 21),
#             maxLevel=3,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
#         )
#         self.feature_params = dict(
#             maxCorners=max_corners,
#             qualityLevel=0.005,
#             minDistance=min_feature_dist,
#             blockSize=5,
#         )

#         self.hough_weight                    = hough_weight
#         self.hough_blur_kernel               = hough_blur_kernel
#         self.hough_canny_low                 = hough_canny_low
#         self.hough_canny_high                = hough_canny_high
#         self.hough_threshold                 = hough_threshold
#         self.hough_min_length                = hough_min_length
#         self.hough_max_gap                   = hough_max_gap
#         self.hough_vertical_tol_deg          = hough_vertical_tol_deg
#         self.hough_cluster_gap               = hough_cluster_gap
#         self.hough_max_box_width             = hough_max_box_width
#         self.hough_min_aspect                = hough_min_aspect
#         self.hough_max_aspect                = hough_max_aspect
#         self.hough_border_margin             = hough_border_margin
#         self.hough_max_interior_edge_density = hough_max_interior_edge_density

#         self.orange_weight         = orange_weight
#         self.hsv_lower             = np.array(hsv_lower, dtype=np.uint8)
#         self.hsv_upper             = np.array(hsv_upper, dtype=np.uint8)
#         self.orange_min_area       = orange_min_area
#         self.orange_min_aspect     = orange_min_aspect
#         self.orange_min_confidence = orange_min_confidence
#         self.orange_morph_kernel   = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, orange_morph_kernel)

#         self.ground_weight         = ground_weight
#         self.ground_green_lower    = np.array(ground_green_lower, dtype=np.uint8)
#         self.ground_green_upper    = np.array(ground_green_upper, dtype=np.uint8)
#         self.ground_morph_kernel   = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, ground_morph_kernel)
#         self.ground_min_mean_green = ground_min_mean_green

#         self.use_flow   = True
#         self.use_hough  = True
#         self.use_orange = True
#         self.use_ground = True

#         self._prev_roi: Optional[np.ndarray] = None
#         self._prev_pts: Optional[np.ndarray] = None
#         self._flow_scores_smooth  = np.zeros(n_cols, dtype=np.float32)
#         self._fused_scores_smooth = np.zeros(n_cols, dtype=np.float32)
#         self._frame_count = 0

#     # -------------------------------------------------------------------------
#     def _stage1(self, img_bgr):
#         rotated    = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         h, w       = rotated.shape[:2]
#         gray       = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#         enhanced   = self.clahe.apply(gray)
#         roi_top    = int(h * self.roi_top_frac)
#         roi_bottom = int(h * self.roi_bottom_frac)
#         roi_frame  = enhanced[roi_top:roi_bottom, :]
#         return enhanced, roi_top, roi_bottom, roi_frame, h, w

#     # -------------------------------------------------------------------------
#     def _compensate_ego_motion(self, flow_vecs):
#         if flow_vecs is None or len(flow_vecs) == 0:
#             return flow_vecs
#         compensated = flow_vecs.copy().astype(np.float32)
#         compensated[:, 0] -= np.mean(compensated[:, 0])
#         compensated[:, 1] -= np.mean(compensated[:, 1])
#         return compensated

#     # -------------------------------------------------------------------------
#     def _stage2_3(self, roi_frame):
#         roi_w     = roi_frame.shape[1]
#         col_width = roi_w / self.n_cols

#         if self._prev_roi is None or self._prev_pts is None or len(self._prev_pts) < 4:
#             self._prev_roi = roi_frame.copy()
#             self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
#             return None, None, None, None, self._flow_scores_smooth.copy()

#         curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
#             self._prev_roi, roi_frame, self._prev_pts, None, **self.lk_params)

#         if curr_pts is None or status is None:
#             self._prev_roi = roi_frame.copy()
#             self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
#             return None, None, None, None, self._flow_scores_smooth.copy()

#         mask_good      = status.ravel() == 1
#         prev_good      = self._prev_pts[mask_good].reshape(-1, 2)
#         curr_good      = curr_pts[mask_good].reshape(-1, 2)
#         flow_vecs      = (curr_good - prev_good).astype(np.float32)
#         flow_vecs_comp = self._compensate_ego_motion(flow_vecs)

#         col_scores_raw = np.zeros(self.n_cols, dtype=np.float32)
#         col_counts     = np.zeros(self.n_cols, dtype=np.int32)
#         magnitudes     = np.linalg.norm(flow_vecs_comp, axis=1)

#         for pt, mag in zip(prev_good, magnitudes):
#             col_idx = int(np.clip(int(pt[0] / col_width), 0, self.n_cols - 1))
#             col_scores_raw[col_idx] += mag
#             col_counts[col_idx]     += 1
#         for c in range(self.n_cols):
#             if col_counts[c] > 0:
#                 col_scores_raw[c] /= col_counts[c]

#         max_f = col_scores_raw.max()
#         if max_f > 0:
#             col_scores_raw /= max_f

#         self._flow_scores_smooth = (
#             self.iir_alpha * col_scores_raw +
#             (1 - self.iir_alpha) * self._flow_scores_smooth
#         )

#         self._prev_roi = roi_frame.copy()
#         self._frame_count += 1
#         if (self._frame_count % self.refresh_interval == 0 or
#                 len(curr_good) < self.max_corners // 3):
#             self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
#             self._frame_count = 0
#         else:
#             self._prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

#         return prev_good, curr_good, flow_vecs, flow_vecs_comp, self._flow_scores_smooth.copy()

#     # -------------------------------------------------------------------------
#     def _stage4_hough(self, roi_frame):
#         roi_h, roi_w = roi_frame.shape[:2]
#         col_width    = roi_w / self.n_cols

#         blurred = cv2.GaussianBlur(roi_frame,
#                                    (self.hough_blur_kernel, self.hough_blur_kernel), 0)
#         edges   = cv2.Canny(blurred, self.hough_canny_low, self.hough_canny_high)

#         lines = cv2.HoughLinesP(
#             edges, rho=1, theta=np.pi / 180,
#             threshold=self.hough_threshold,
#             minLineLength=self.hough_min_length,
#             maxLineGap=self.hough_max_gap,
#         )

#         vertical = []
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 dx = abs(x2 - x1)
#                 dy = abs(y2 - y1)
#                 if np.degrees(np.arctan2(dx, max(dy, 1))) <= self.hough_vertical_tol_deg:
#                     vertical.append((x1, y1, x2, y2))

#         raw_boxes = []
#         if vertical:
#             sorted_lines = sorted(vertical, key=lambda l: (l[0] + l[2]) / 2)
#             clusters, current = [], [sorted_lines[0]]
#             for line in sorted_lines[1:]:
#                 prev_x = (current[-1][0] + current[-1][2]) / 2
#                 curr_x = (line[0] + line[2]) / 2
#                 if abs(curr_x - prev_x) <= self.hough_cluster_gap:
#                     current.append(line)
#                 else:
#                     clusters.append(current)
#                     current = [line]
#             clusters.append(current)

#             for cluster in clusters:
#                 xs = [p for l in cluster for p in (l[0], l[2])]
#                 ys = [p for l in cluster for p in (l[1], l[3])]
#                 x_min, x_max = min(xs), max(xs)
#                 y_min, y_max = min(ys), max(ys)
#                 if (x_max - x_min) <= self.hough_max_box_width:
#                     raw_boxes.append((x_min, y_min, x_max, y_max))
#                 else:
#                     x_mid = (x_min + x_max) // 2
#                     for half in [
#                         [l for l in cluster if (l[0]+l[2])/2 <= x_mid],
#                         [l for l in cluster if (l[0]+l[2])/2 >  x_mid],
#                     ]:
#                         if not half:
#                             continue
#                         hxs = [p for l in half for p in (l[0], l[2])]
#                         hys = [p for l in half for p in (l[1], l[3])]
#                         raw_boxes.append((min(hxs), min(hys), max(hxs), max(hys)))

#         filtered_boxes = []
#         for (x1, y1, x2, y2) in raw_boxes:
#             bw = max(x2 - x1, 1)
#             bh = max(y2 - y1, 1)
#             if x1 < self.hough_border_margin or x2 > roi_w - self.hough_border_margin:
#                 continue
#             ratio = bh / bw
#             if not (self.hough_min_aspect <= ratio <= self.hough_max_aspect):
#                 continue
#             shrink = 10
#             ix1 = min(x1 + shrink, x2)
#             ix2 = max(x2 - shrink, x1)
#             iy1 = min(y1 + shrink, y2)
#             iy2 = max(y2 - shrink, y1)
#             if (ix2 - ix1) > 0 and (iy2 - iy1) > 0:
#                 edge_density = (np.count_nonzero(edges[iy1:iy2, ix1:ix2]) /
#                                 ((ix2 - ix1) * (iy2 - iy1)))
#                 if edge_density > self.hough_max_interior_edge_density:
#                     continue
#             filtered_boxes.append((x1, y1, x2, y2))

#         hough_scores = np.zeros(self.n_cols, dtype=np.float32)
#         for (x1, y1, x2, y2) in filtered_boxes:
#             for c in range(self.n_cols):
#                 cx1     = int(c * col_width)
#                 cx2     = int((c + 1) * col_width)
#                 overlap = max(0, min(x2, cx2) - max(x1, cx1))
#                 if overlap > 0:
#                     hough_scores[c] = max(hough_scores[c],
#                                           overlap / max(cx2 - cx1, 1))

#         return filtered_boxes, hough_scores, vertical, edges

#     # -------------------------------------------------------------------------
#     def _stage5_orange(self, rotated_bgr, roi_top, roi_bottom):
#         roi_bgr      = rotated_bgr[roi_top:roi_bottom, :]
#         roi_h, roi_w = roi_bgr.shape[:2]
#         col_width    = roi_w / self.n_cols

#         hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.orange_morph_kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.orange_morph_kernel)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         boxes = []
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < self.orange_min_area:
#                 continue
#             x, y, bw, bh = cv2.boundingRect(cnt)
#             if (bh / max(bw, 1)) < self.orange_min_aspect:
#                 continue
#             conf = mask[y:y+bh, x:x+bw].sum() / 255 / max(bw * bh, 1)
#             if conf < self.orange_min_confidence:
#                 continue
#             boxes.append(PoleDetection(
#                 x=x, y=y, w=bw, h=bh,
#                 cx=x + bw // 2, cy=y + bh // 2,
#                 area=int(area), confidence=round(conf, 2)
#             ))
#         boxes.sort(key=lambda d: d.cx)

#         orange_scores = np.zeros(self.n_cols, dtype=np.float32)
#         for c in range(self.n_cols):
#             x1 = int(c * col_width)
#             x2 = int((c + 1) * col_width)
#             col_mask = mask[:, x1:x2]
#             orange_scores[c] = float(col_mask.sum()) / 255 / max(col_mask.size, 1)
#         max_o = orange_scores.max()
#         if max_o > 0:
#             orange_scores /= max_o

#         mask_full = np.zeros(rotated_bgr.shape[:2], dtype=np.uint8)
#         mask_full[roi_top:roi_bottom, :] = mask
#         return boxes, mask_full, orange_scores

#     # -------------------------------------------------------------------------
#     def _stage6_ground(self, rotated_bgr, roi_top, roi_bottom):
#         """
#         Per-column green fraction in the ROI.
#         Columns below mean green = likely occluded → obstacle score.
#         Silenced when mean green < ground_min_mean_green.
#         """
#         roi_bgr   = rotated_bgr[roi_top:roi_bottom, :]
#         roi_w     = roi_bgr.shape[1]
#         col_width = roi_w / self.n_cols

#         hsv        = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
#         green_mask = cv2.inRange(hsv, self.ground_green_lower, self.ground_green_upper)
#         green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  self.ground_morph_kernel)
#         green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.ground_morph_kernel)

#         green_scores = np.zeros(self.n_cols, dtype=np.float32)
#         for c in range(self.n_cols):
#             x1 = int(c * col_width)
#             x2 = int((c + 1) * col_width)
#             col = green_mask[:, x1:x2]
#             green_scores[c] = float(col.sum()) / 255 / max(col.size, 1)

#         mean_green = float(green_scores.mean())

#         if mean_green < self.ground_min_mean_green:
#             obstacle_scores = np.zeros(self.n_cols, dtype=np.float32)
#         else:
#             obstacle_scores = np.clip(mean_green - green_scores, 0.0, None)
#             max_o = obstacle_scores.max()
#             if max_o > 0:
#                 obstacle_scores /= max_o

#         mask_full = np.zeros(rotated_bgr.shape[:2], dtype=np.uint8)
#         mask_full[roi_top:roi_bottom, :] = green_mask
#         return mask_full, green_scores, obstacle_scores

#     # -------------------------------------------------------------------------
#     def _find_gap_center(self, obstacle_cols, roi_w):
#         col_width = roi_w / self.n_cols
#         free = [c for c in range(self.n_cols) if c not in obstacle_cols]
#         if not free:
#             return None
#         best_start, best_len = free[0], 1
#         cur_start,  cur_len  = free[0], 1
#         for i in range(1, len(free)):
#             if free[i] == free[i-1] + 1:
#                 cur_len += 1
#                 if cur_len > best_len:
#                     best_start, best_len = cur_start, cur_len
#             else:
#                 cur_start, cur_len = free[i], 1
#         return int((best_start + best_len / 2.0) * col_width)

#     # -------------------------------------------------------------------------
#     def detect(self, img_bgr, telem=None):
#         enhanced, roi_top, roi_bottom, roi_frame, frame_h, frame_w = self._stage1(img_bgr)
#         rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

#         prev_pts, curr_pts, flow_vecs, flow_vecs_comp, flow_scores = \
#             self._stage2_3(roi_frame)

#         hough_boxes, hough_scores, hough_verticals, hough_edges = \
#             self._stage4_hough(roi_frame)

#         boxes, orange_mask, orange_scores = \
#             self._stage5_orange(rotated, roi_top, roi_bottom)

#         ground_mask, ground_green_scores, ground_obstacle_scores = \
#             self._stage6_ground(rotated, roi_top, roi_bottom)

#         active_flow   = flow_scores            * self.flow_weight   if self.use_flow   else np.zeros(self.n_cols, dtype=np.float32)
#         active_hough  = hough_scores           * self.hough_weight  if self.use_hough  else np.zeros(self.n_cols, dtype=np.float32)
#         active_orange = orange_scores          * self.orange_weight if self.use_orange else np.zeros(self.n_cols, dtype=np.float32)
#         active_ground = ground_obstacle_scores * self.ground_weight if self.use_ground else np.zeros(self.n_cols, dtype=np.float32)

#         fused_raw = active_flow + active_hough + active_orange + active_ground

#         self._fused_scores_smooth = (
#             self.iir_alpha * fused_raw +
#             (1 - self.iir_alpha) * self._fused_scores_smooth
#         )

#         mean_f    = np.mean(self._fused_scores_smooth)
#         std_f     = np.std(self._fused_scores_smooth)
#         threshold = mean_f + self.flow_threshold * std_f

#         obstacle_cols = [
#             c for c in range(self.n_cols)
#             if self._fused_scores_smooth[c] > threshold
#         ]

#         roi_w        = roi_frame.shape[1]
#         gap_center_x = self._find_gap_center(obstacle_cols, roi_w)

#         return ObstacleResult(
#             preprocessed=enhanced,
#             roi_top=roi_top,
#             roi_bottom=roi_bottom,
#             roi_frame=roi_frame,
#             flow_points_prev=prev_pts,
#             flow_points_next=curr_pts,
#             flow_vectors=flow_vecs,
#             flow_vectors_comp=flow_vecs_comp,
#             hough_boxes=hough_boxes,
#             hough_scores=hough_scores,
#             orange_boxes=boxes,
#             orange_mask=orange_mask,
#             orange_scores=orange_scores,
#             ground_mask=ground_mask,
#             ground_green_scores=ground_green_scores,
#             ground_obstacle_scores=ground_obstacle_scores,
#             flow_scores=flow_scores,
#             fused_scores=self._fused_scores_smooth.copy(),
#             obstacle_cols=obstacle_cols,
#             gap_center_x=gap_center_x,
#             n_cols=self.n_cols,
#         )

#     # -------------------------------------------------------------------------
#     def draw(self, img_bgr, result):
#         """
#         4 rows × 3 cols = 12 slots, 10 used, last 2 black.

#         Row 1: [1. Raw]          [2. CLAHE+ROI]      [3. Flow vectors]
#         Row 2: [4. Flow scores]  [5. Hough poles]    [6. Orange detect]
#         Row 3: [7. Ground mask]  [8. Ground scores]  [9. Fused scores]
#         Row 4: [10. Final]       [11. (reserved)]    [12. (reserved)]
#         """
#         rotated      = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         h, w         = rotated.shape[:2]
#         enhanced_bgr = cv2.cvtColor(result.preprocessed, cv2.COLOR_GRAY2BGR)
#         alpha        = 0.45
#         roi_h        = result.roi_bottom - result.roi_top
#         col_width    = w / result.n_cols

#         # ── Helpers ──────────────────────────────────────────────────────────

#         def dim_outside_roi(panel):
#             panel[:result.roi_top, :]    = (panel[:result.roi_top, :]    * alpha).astype(np.uint8)
#             panel[result.roi_bottom:, :] = (panel[result.roi_bottom:, :] * alpha).astype(np.uint8)
#             cv2.line(panel, (0, result.roi_top),    (w, result.roi_top),    (0, 255, 255), 1)
#             cv2.line(panel, (0, result.roi_bottom), (w, result.roi_bottom), (0, 255, 255), 1)

#         def draw_score_bars(panel, scores, color_obs, color_free, label,
#                             obstacle_cols=None, draw_tint=False):
#             if scores is None:
#                 cv2.putText(panel, label, (5, 18),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
#                 return
#             mean_s = float(np.mean(scores))
#             std_s  = float(np.std(scores))
#             thresh = mean_s + self.flow_threshold * std_s

#             for c in range(result.n_cols):
#                 x1     = int(c * col_width)
#                 x2     = int((c + 1) * col_width)
#                 is_obs = obstacle_cols is not None and c in obstacle_cols
#                 score  = float(scores[c])
#                 bar_h  = int(min(score / max(thresh * 1.5, 1e-6), 1.0) * roi_h)
#                 cv2.rectangle(panel,
#                               (x1, result.roi_bottom - bar_h),
#                               (x2 - 1, result.roi_bottom),
#                               color_obs if is_obs else color_free, -1)
#                 if draw_tint and is_obs:
#                     tint = panel[result.roi_top:result.roi_bottom, x1:x2].copy()
#                     tint = (tint * 0.5 + np.array([0, 0, 80])).clip(0, 255).astype(np.uint8)
#                     panel[result.roi_top:result.roi_bottom, x1:x2] = tint
#                 cv2.line(panel, (x2, result.roi_top), (x2, result.roi_bottom), (55, 55, 55), 1)

#             thresh_y = result.roi_bottom - int(
#                 min(thresh / max(thresh * 1.5, 1e-6), 1.0) * roi_h)
#             cv2.line(panel, (0, thresh_y), (w, thresh_y), (0, 180, 255), 1)
#             cv2.putText(panel, "thr", (w - 28, thresh_y - 3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 255), 1)
#             cv2.putText(panel, label, (5, 18),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_free, 1)

#         # ── Panel 1: Raw rotated ──────────────────────────────────────────────
#         panel1 = rotated.copy()
#         cv2.putText(panel1, "1. Raw (rotated)", (5, 18),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # ── Panel 2: CLAHE + ROI ──────────────────────────────────────────────
#         panel2 = enhanced_bgr.copy()
#         dim_outside_roi(panel2)
#         cv2.putText(panel2, "2. CLAHE + ROI", (5, 18),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#         # ── Panel 3: Compensated flow vectors ─────────────────────────────────
#         panel3 = rotated.copy()
#         dim_outside_roi(panel3)
#         if result.flow_points_prev is not None and result.flow_vectors_comp is not None:
#             for pt, vec in zip(result.flow_points_prev, result.flow_vectors_comp):
#                 x, y   = int(pt[0]), int(pt[1]) + result.roi_top
#                 dx, dy = int(vec[0] * 3), int(vec[1] * 3)
#                 cv2.arrowedLine(panel3, (x, y), (x + dx, y + dy),
#                                 (255, 200, 0), 1, tipLength=0.3)
#                 cv2.circle(panel3, (x, y), 2, (255, 200, 0), -1)
#         cv2.putText(panel3,
#                     f"3. Flow vectors [{'ON' if self.use_flow else 'OFF'}]",
#                     (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (255, 200, 0) if self.use_flow else (80, 80, 80), 1)

#         # ── Panel 4: Flow column scores ───────────────────────────────────────
#         panel4 = rotated.copy()
#         dim_outside_roi(panel4)
#         draw_score_bars(panel4,
#                         result.flow_scores if self.use_flow else None,
#                         color_obs=(0, 60, 255), color_free=(0, 200, 0),
#                         label=f"4. Flow scores [{'ON' if self.use_flow else 'OFF'}]")

#         # ── Panel 5: Hough pole detection ─────────────────────────────────────
#         panel5 = rotated.copy()
#         dim_outside_roi(panel5)
#         if result.hough_boxes and self.use_hough:
#             for (x1, y1, x2, y2) in result.hough_boxes:
#                 fy1 = y1 + result.roi_top
#                 fy2 = y2 + result.roi_top
#                 cv2.rectangle(panel5, (x1, fy1), (x2, fy2), (100, 255, 100), 2)
#                 cx_box = (x1 + x2) // 2
#                 cv2.line(panel5, (cx_box, result.roi_top), (cx_box, result.roi_bottom),
#                          (100, 255, 100), 1)
#                 cv2.putText(panel5, f"{x2-x1}x{y2-y1}", (x1, fy1 - 4),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
#         draw_score_bars(panel5,
#                         result.hough_scores if self.use_hough else None,
#                         color_obs=(0, 60, 255), color_free=(100, 255, 100),
#                         label=f"5. Hough [{'ON' if self.use_hough else 'OFF'}]")

#         # ── Panel 6: Orange detection ─────────────────────────────────────────
#         panel6 = rotated.copy()
#         dim_outside_roi(panel6)
#         if result.orange_mask is not None and self.use_orange:
#             tint = panel6.copy()
#             tint[result.orange_mask > 0] = [0, 100, 255]
#             panel6 = cv2.addWeighted(panel6, 0.65, tint, 0.35, 0)
#         if result.orange_boxes:
#             for d in result.orange_boxes:
#                 bx, by = d.x, d.y + result.roi_top
#                 cv2.rectangle(panel6, (bx, by), (bx + d.w, by + d.h), (0, 200, 255), 2)
#                 cv2.circle(panel6, (d.cx, d.cy + result.roi_top), 4, (0, 0, 255), -1)
#                 cv2.line(panel6, (d.cx, result.roi_top), (d.cx, result.roi_bottom),
#                          (0, 80, 255), 1)
#                 cv2.putText(panel6, f"cx={d.cx} ({d.confidence:.2f})",
#                             (bx, max(by - 5, 14)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
#         draw_score_bars(panel6,
#                         result.orange_scores if self.use_orange else None,
#                         color_obs=(0, 60, 255), color_free=(0, 140, 255),
#                         label=f"6. Orange [{'ON' if self.use_orange else 'OFF'}]")

#         # ── Panel 7: Green ground mask overlay ────────────────────────────────
#         panel7 = rotated.copy()
#         dim_outside_roi(panel7)
#         if result.ground_mask is not None and self.use_ground:
#             tint = panel7.copy()
#             tint[result.ground_mask > 0] = [0, 200, 60]
#             panel7 = cv2.addWeighted(panel7, 0.55, tint, 0.45, 0)
#         cv2.putText(panel7,
#                     f"7. Ground mask [{'ON' if self.use_ground else 'OFF'}]",
#                     (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (0, 200, 60) if self.use_ground else (80, 80, 80), 1)

#         # ── Panel 8: Ground obstacle scores ───────────────────────────────────
#         panel8 = rotated.copy()
#         dim_outside_roi(panel8)
#         draw_score_bars(panel8,
#                         result.ground_obstacle_scores if self.use_ground else None,
#                         color_obs=(0, 60, 255), color_free=(0, 200, 60),
#                         label=f"8. Ground scores [{'ON' if self.use_ground else 'OFF'}]")

#         # ── Panel 9: Fused scores ─────────────────────────────────────────────
#         panel9 = rotated.copy()
#         dim_outside_roi(panel9)
#         draw_score_bars(panel9, result.fused_scores,
#                         color_obs=(0, 60, 255), color_free=(0, 220, 120),
#                         label="9. Fused scores",
#                         obstacle_cols=result.obstacle_cols, draw_tint=True)
#         if result.gap_center_x is not None:
#             gx = result.gap_center_x
#             cv2.arrowedLine(panel9,
#                             (gx, result.roi_top + roi_h // 2 + 20),
#                             (gx, result.roi_top + roi_h // 2 - 20),
#                             (0, 255, 120), 2, tipLength=0.4)
#             cv2.putText(panel9, "GO",
#                         (gx - 10, result.roi_top + roi_h // 2 + 35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
#         else:
#             cv2.putText(panel9, "NO GAP",
#                         (w // 2 - 45, result.roi_top + roi_h // 2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # ── Panel 10: Final overlay on raw image ──────────────────────────────
#         panel10 = rotated.copy()

#         for c in range(result.n_cols):
#             x1     = int(c * col_width)
#             x2     = int((c + 1) * col_width)
#             is_obs = result.obstacle_cols is not None and c in result.obstacle_cols
#             tint   = panel10[result.roi_top:result.roi_bottom, x1:x2].copy()
#             if is_obs:
#                 tint = (tint * 0.45 + np.array([0, 0, 120])).clip(0, 255).astype(np.uint8)
#             else:
#                 tint = (tint * 0.85 + np.array([0, 25, 0])).clip(0, 255).astype(np.uint8)
#             panel10[result.roi_top:result.roi_bottom, x1:x2] = tint
#             cv2.line(panel10, (x2, result.roi_top), (x2, result.roi_bottom), (40, 40, 40), 1)

#         panel10[:result.roi_top, :]    = (panel10[:result.roi_top, :]    * alpha).astype(np.uint8)
#         panel10[result.roi_bottom:, :] = (panel10[result.roi_bottom:, :] * alpha).astype(np.uint8)
#         cv2.line(panel10, (0, result.roi_top),    (w, result.roi_top),    (0, 255, 255), 1)
#         cv2.line(panel10, (0, result.roi_bottom), (w, result.roi_bottom), (0, 255, 255), 1)

#         if result.gap_center_x is not None:
#             gx      = result.gap_center_x
#             arrow_y = result.roi_top + roi_h // 2
#             cv2.arrowedLine(panel10,
#                             (gx, arrow_y + 35), (gx, arrow_y - 35),
#                             (0, 255, 80), 3, tipLength=0.3)
#             cv2.putText(panel10, "GO",
#                         (gx - 14, arrow_y + 55),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
#             cv2.line(panel10, (gx, result.roi_top), (gx, result.roi_bottom),
#                      (0, 255, 80), 1)
#         else:
#             cv2.putText(panel10, "NO GAP",
#                         (w // 2 - 45, result.roi_top + roi_h // 2 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(panel10, "STOP",
#                         (w // 2 - 35, result.roi_top + roi_h // 2 + 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         n_obs = len(result.obstacle_cols) if result.obstacle_cols else 0
#         cv2.putText(panel10, f"10. Final  obs:{n_obs}/{result.n_cols}",
#                     (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # ── Panels 11 & 12: Reserved (black) ──────────────────────────────────
#         panel11 = np.zeros((h, w, 3), dtype=np.uint8)
#         cv2.putText(panel11, "11. (reserved)", (5, 18),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

#         panel12 = np.zeros((h, w, 3), dtype=np.uint8)
#         cv2.putText(panel12, "12. (reserved)", (5, 18),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

#         # ── Assemble 4 rows × 3 cols ──────────────────────────────────────────
#         dv = np.full((h, 3, 3),         40, dtype=np.uint8)  # vertical divider
#         dh = np.full((3, w * 3 + 6, 3), 40, dtype=np.uint8)  # horizontal divider

#         row1 = np.hstack([panel1,  dv, panel2,  dv, panel3])
#         row2 = np.hstack([panel4,  dv, panel5,  dv, panel6])
#         row3 = np.hstack([panel7,  dv, panel8,  dv, panel9])
#         row4 = np.hstack([panel10, dv, panel11, dv, panel12])

#         return np.vstack([row1, dh, row2, dh, row3, dh, row4])



import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


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


@dataclass
class GapCandidate:
    """A free corridor between obstacles."""
    rank: int            # 1 = best
    center_x: int        # pixel x of gap centre in ROI-width space
    col_start: int       # first free column index
    col_end: int         # last free column index (inclusive)
    width_cols: int      # number of free columns
    safety: float        # composite safety score [0, 1]
    width_score: float
    clearness_score: float
    centrality_score: float


@dataclass
class ObstacleResult:
    preprocessed: np.ndarray
    roi_top: int
    roi_bottom: int
    roi_frame: np.ndarray
    clahe_applied: bool = True
    flow_points_prev: Optional[np.ndarray] = None
    flow_points_next: Optional[np.ndarray] = None
    flow_vectors: Optional[np.ndarray] = None
    flow_vectors_comp: Optional[np.ndarray] = None
    hough_boxes: Optional[List[Tuple]] = None
    hough_scores: Optional[np.ndarray] = None
    orange_boxes: Optional[List] = None
    orange_mask: Optional[np.ndarray] = None
    orange_scores: Optional[np.ndarray] = None
    ground_mask: Optional[np.ndarray] = None
    ground_green_scores: Optional[np.ndarray] = None
    ground_obstacle_scores: Optional[np.ndarray] = None
    flow_scores: Optional[np.ndarray] = None
    fused_scores: Optional[np.ndarray] = None
    obstacle_cols: Optional[List[int]] = None
    # ── new: top-3 gaps ───────────────────────────────────────────────────────
    gap_candidates: List[GapCandidate] = field(default_factory=list)
    gap_center_x: Optional[int] = None   # kept for backward compat (= candidates[0].center_x)
    n_cols: int = 22


class ObstacleDetector:
    """
    Stage 1 — Rotate 90° CCW + CLAHE + ROI crop
    Stage 2 — Sparse LK optical flow → column voting
    Stage 3 — Mean flow subtraction (ego-motion compensation)
    Stage 4 — Hough vertical line detector → pole boxes → column scores
    Stage 5 — Orange HSV detector → column scores
    Stage 6 — Green ground mask: missing-green columns → obstacle scores
    Fusion  — fused = flow*w_f + hough*w_h + orange*w_o + ground*w_g
    Gap     — top-3 free corridors ranked by composite safety score
    """

    def __init__(
        self,
        roi_top_frac: float = 0.10,
        roi_bottom_frac: float = 0.90,
        clahe_clip_limit: float = 4.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        n_cols: int = 22,
        flow_threshold: float = 0.7,
        iir_alpha: float = 0.35,
        max_corners: int = 150,
        min_feature_dist: int = 4,
        refresh_interval: int = 8,
        flow_weight: float = 1.5,
        hough_weight: float = 1.5,
        hough_blur_kernel: int = 5,
        hough_canny_low: int = 50,
        hough_canny_high: int = 150,
        hough_threshold: int = 30,
        hough_min_length: int = 40,
        hough_max_gap: int = 100,
        hough_vertical_tol_deg: float = 20.0,
        hough_cluster_gap: int = 35,
        hough_max_box_width: int = 200,
        hough_min_aspect: float = 1.5,
        hough_max_aspect: float = 8.0,
        hough_border_margin: int = 15,
        hough_max_interior_edge_density: float = 0.15,
        orange_weight: float = 2.5,
        hsv_lower: Tuple = (0, 40, 50),
        hsv_upper: Tuple = (25, 255, 255),
        orange_min_area: int = 300,
        orange_min_aspect: float = 1.5,
        orange_min_confidence: float = 0.2,
        orange_morph_kernel: Tuple[int, int] = (5, 5),
        ground_weight: float = 2.0,
        ground_green_lower: Tuple = (18, 17, 124),
        ground_green_upper: Tuple = (76, 153, 255),
        ground_morph_kernel: Tuple[int, int] = (7, 7),
        ground_min_mean_green: float = 0.05,
        # Gap safety weights (must sum to 1.0)
        gap_w_width: float = 0.50,
        gap_w_clearness: float = 0.40,
        gap_w_centrality: float = 0.10,
    ):
        self.roi_top_frac    = roi_top_frac
        self.roi_bottom_frac = roi_bottom_frac
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                     tileGridSize=clahe_tile_grid)
        self.n_cols           = n_cols
        self.flow_threshold   = flow_threshold
        self.iir_alpha        = iir_alpha
        self.max_corners      = max_corners
        self.min_feature_dist = min_feature_dist
        self.refresh_interval = refresh_interval
        self.flow_weight      = flow_weight

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.005,
            minDistance=min_feature_dist,
            blockSize=5,
        )

        self.hough_weight                    = hough_weight
        self.hough_blur_kernel               = hough_blur_kernel
        self.hough_canny_low                 = hough_canny_low
        self.hough_canny_high                = hough_canny_high
        self.hough_threshold                 = hough_threshold
        self.hough_min_length                = hough_min_length
        self.hough_max_gap                   = hough_max_gap
        self.hough_vertical_tol_deg          = hough_vertical_tol_deg
        self.hough_cluster_gap               = hough_cluster_gap
        self.hough_max_box_width             = hough_max_box_width
        self.hough_min_aspect                = hough_min_aspect
        self.hough_max_aspect                = hough_max_aspect
        self.hough_border_margin             = hough_border_margin
        self.hough_max_interior_edge_density = hough_max_interior_edge_density

        self.orange_weight         = orange_weight
        self.hsv_lower             = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper             = np.array(hsv_upper, dtype=np.uint8)
        self.orange_min_area       = orange_min_area
        self.orange_min_aspect     = orange_min_aspect
        self.orange_min_confidence = orange_min_confidence
        self.orange_morph_kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, orange_morph_kernel)

        self.ground_weight         = ground_weight
        self.ground_green_lower    = np.array(ground_green_lower, dtype=np.uint8)
        self.ground_green_upper    = np.array(ground_green_upper, dtype=np.uint8)
        self.ground_morph_kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, ground_morph_kernel)
        self.ground_min_mean_green = ground_min_mean_green

        self.gap_w_width      = gap_w_width
        self.gap_w_clearness  = gap_w_clearness
        self.gap_w_centrality = gap_w_centrality

        self.use_flow   = True
        self.use_hough  = True
        self.use_orange = True
        self.use_ground = True

        self._prev_roi: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._flow_scores_smooth  = np.zeros(n_cols, dtype=np.float32)
        self._fused_scores_smooth = np.zeros(n_cols, dtype=np.float32)
        self._frame_count = 0

    # -------------------------------------------------------------------------
    def _stage1(self, img_bgr):
        rotated    = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w       = rotated.shape[:2]
        gray       = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        enhanced   = self.clahe.apply(gray)
        roi_top    = int(h * self.roi_top_frac)
        roi_bottom = int(h * self.roi_bottom_frac)
        roi_frame  = enhanced[roi_top:roi_bottom, :]
        return enhanced, roi_top, roi_bottom, roi_frame, h, w

    # -------------------------------------------------------------------------
    def _compensate_ego_motion(self, flow_vecs):
        if flow_vecs is None or len(flow_vecs) == 0:
            return flow_vecs
        compensated = flow_vecs.copy().astype(np.float32)
        compensated[:, 0] -= np.mean(compensated[:, 0])
        compensated[:, 1] -= np.mean(compensated[:, 1])
        return compensated

    # -------------------------------------------------------------------------
    def _stage2_3(self, roi_frame):
        roi_w     = roi_frame.shape[1]
        col_width = roi_w / self.n_cols

        if self._prev_roi is None or self._prev_pts is None or len(self._prev_pts) < 4:
            self._prev_roi = roi_frame.copy()
            self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
            return None, None, None, None, self._flow_scores_smooth.copy()

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_roi, roi_frame, self._prev_pts, None, **self.lk_params)

        if curr_pts is None or status is None:
            self._prev_roi = roi_frame.copy()
            self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
            return None, None, None, None, self._flow_scores_smooth.copy()

        mask_good      = status.ravel() == 1
        prev_good      = self._prev_pts[mask_good].reshape(-1, 2)
        curr_good      = curr_pts[mask_good].reshape(-1, 2)
        flow_vecs      = (curr_good - prev_good).astype(np.float32)
        flow_vecs_comp = self._compensate_ego_motion(flow_vecs)

        col_scores_raw = np.zeros(self.n_cols, dtype=np.float32)
        col_counts     = np.zeros(self.n_cols, dtype=np.int32)
        magnitudes     = np.linalg.norm(flow_vecs_comp, axis=1)

        for pt, mag in zip(prev_good, magnitudes):
            col_idx = int(np.clip(int(pt[0] / col_width), 0, self.n_cols - 1))
            col_scores_raw[col_idx] += mag
            col_counts[col_idx]     += 1
        for c in range(self.n_cols):
            if col_counts[c] > 0:
                col_scores_raw[c] /= col_counts[c]

        max_f = col_scores_raw.max()
        if max_f > 0:
            col_scores_raw /= max_f

        self._flow_scores_smooth = (
            self.iir_alpha * col_scores_raw +
            (1 - self.iir_alpha) * self._flow_scores_smooth
        )

        self._prev_roi = roi_frame.copy()
        self._frame_count += 1
        if (self._frame_count % self.refresh_interval == 0 or
                len(curr_good) < self.max_corners // 3):
            self._prev_pts = cv2.goodFeaturesToTrack(roi_frame, **self.feature_params)
            self._frame_count = 0
        else:
            self._prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

        return prev_good, curr_good, flow_vecs, flow_vecs_comp, self._flow_scores_smooth.copy()

    # -------------------------------------------------------------------------
    def _stage4_hough(self, roi_frame):
        roi_h, roi_w = roi_frame.shape[:2]
        col_width    = roi_w / self.n_cols

        blurred = cv2.GaussianBlur(roi_frame,
                                   (self.hough_blur_kernel, self.hough_blur_kernel), 0)
        edges   = cv2.Canny(blurred, self.hough_canny_low, self.hough_canny_high)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        vertical = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if np.degrees(np.arctan2(dx, max(dy, 1))) <= self.hough_vertical_tol_deg:
                    vertical.append((x1, y1, x2, y2))

        raw_boxes = []
        if vertical:
            sorted_lines = sorted(vertical, key=lambda l: (l[0] + l[2]) / 2)
            clusters, current = [], [sorted_lines[0]]
            for line in sorted_lines[1:]:
                prev_x = (current[-1][0] + current[-1][2]) / 2
                curr_x = (line[0] + line[2]) / 2
                if abs(curr_x - prev_x) <= self.hough_cluster_gap:
                    current.append(line)
                else:
                    clusters.append(current)
                    current = [line]
            clusters.append(current)

            for cluster in clusters:
                xs = [p for l in cluster for p in (l[0], l[2])]
                ys = [p for l in cluster for p in (l[1], l[3])]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                if (x_max - x_min) <= self.hough_max_box_width:
                    raw_boxes.append((x_min, y_min, x_max, y_max))
                else:
                    x_mid = (x_min + x_max) // 2
                    for half in [
                        [l for l in cluster if (l[0]+l[2])/2 <= x_mid],
                        [l for l in cluster if (l[0]+l[2])/2 >  x_mid],
                    ]:
                        if not half:
                            continue
                        hxs = [p for l in half for p in (l[0], l[2])]
                        hys = [p for l in half for p in (l[1], l[3])]
                        raw_boxes.append((min(hxs), min(hys), max(hxs), max(hys)))

        filtered_boxes = []
        for (x1, y1, x2, y2) in raw_boxes:
            bw = max(x2 - x1, 1)
            bh = max(y2 - y1, 1)
            if x1 < self.hough_border_margin or x2 > roi_w - self.hough_border_margin:
                continue
            ratio = bh / bw
            if not (self.hough_min_aspect <= ratio <= self.hough_max_aspect):
                continue
            shrink = 10
            ix1 = min(x1 + shrink, x2)
            ix2 = max(x2 - shrink, x1)
            iy1 = min(y1 + shrink, y2)
            iy2 = max(y2 - shrink, y1)
            if (ix2 - ix1) > 0 and (iy2 - iy1) > 0:
                edge_density = (np.count_nonzero(edges[iy1:iy2, ix1:ix2]) /
                                ((ix2 - ix1) * (iy2 - iy1)))
                if edge_density > self.hough_max_interior_edge_density:
                    continue
            filtered_boxes.append((x1, y1, x2, y2))

        hough_scores = np.zeros(self.n_cols, dtype=np.float32)
        for (x1, y1, x2, y2) in filtered_boxes:
            for c in range(self.n_cols):
                cx1     = int(c * col_width)
                cx2     = int((c + 1) * col_width)
                overlap = max(0, min(x2, cx2) - max(x1, cx1))
                if overlap > 0:
                    hough_scores[c] = max(hough_scores[c],
                                          overlap / max(cx2 - cx1, 1))

        return filtered_boxes, hough_scores, vertical, edges

    # -------------------------------------------------------------------------
    def _stage5_orange(self, rotated_bgr, roi_top, roi_bottom):
        roi_bgr      = rotated_bgr[roi_top:roi_bottom, :]
        roi_h, roi_w = roi_bgr.shape[:2]
        col_width    = roi_w / self.n_cols

        hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.orange_morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.orange_morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.orange_min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if (bh / max(bw, 1)) < self.orange_min_aspect:
                continue
            conf = mask[y:y+bh, x:x+bw].sum() / 255 / max(bw * bh, 1)
            if conf < self.orange_min_confidence:
                continue
            boxes.append(PoleDetection(
                x=x, y=y, w=bw, h=bh,
                cx=x + bw // 2, cy=y + bh // 2,
                area=int(area), confidence=round(conf, 2)
            ))
        boxes.sort(key=lambda d: d.cx)

        orange_scores = np.zeros(self.n_cols, dtype=np.float32)
        for c in range(self.n_cols):
            x1 = int(c * col_width)
            x2 = int((c + 1) * col_width)
            col_mask = mask[:, x1:x2]
            orange_scores[c] = float(col_mask.sum()) / 255 / max(col_mask.size, 1)
        max_o = orange_scores.max()
        if max_o > 0:
            orange_scores /= max_o

        mask_full = np.zeros(rotated_bgr.shape[:2], dtype=np.uint8)
        mask_full[roi_top:roi_bottom, :] = mask
        return boxes, mask_full, orange_scores

    # -------------------------------------------------------------------------
    def _stage6_ground(self, rotated_bgr, roi_top, roi_bottom):
        roi_bgr   = rotated_bgr[roi_top:roi_bottom, :]
        roi_w     = roi_bgr.shape[1]
        col_width = roi_w / self.n_cols

        hsv        = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.ground_green_lower, self.ground_green_upper)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  self.ground_morph_kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.ground_morph_kernel)

        green_scores = np.zeros(self.n_cols, dtype=np.float32)
        for c in range(self.n_cols):
            x1 = int(c * col_width)
            x2 = int((c + 1) * col_width)
            col = green_mask[:, x1:x2]
            green_scores[c] = float(col.sum()) / 255 / max(col.size, 1)

        mean_green = float(green_scores.mean())

        if mean_green < self.ground_min_mean_green:
            obstacle_scores = np.zeros(self.n_cols, dtype=np.float32)
        else:
            obstacle_scores = np.clip(mean_green - green_scores, 0.0, None)
            max_o = obstacle_scores.max()
            if max_o > 0:
                obstacle_scores /= max_o

        mask_full = np.zeros(rotated_bgr.shape[:2], dtype=np.uint8)
        mask_full[roi_top:roi_bottom, :] = green_mask
        return mask_full, green_scores, obstacle_scores

    # -------------------------------------------------------------------------
    def _find_gap_candidates(self, obstacle_cols, fused_scores, roi_w, top_n: int = 3):
        """
        Find all contiguous free corridors, score each, return top_n sorted best-first.

        Safety score = w_width * width_score
                     + w_clearness * clearness_score
                     + w_centrality * centrality_score

        width_score      : gap_width / n_cols
        clearness_score  : mean(1 - fused_norm) over gap columns,
                           where fused_norm = fused / max(fused) clipped to [0,1]
        centrality_score : 1 - 2*|gap_centre_col - n_cols/2| / n_cols
        """
        col_width   = roi_w / self.n_cols
        obs_set     = set(obstacle_cols) if obstacle_cols else set()
        free        = [c for c in range(self.n_cols) if c not in obs_set]

        if not free:
            return []

        # Normalize fused scores to [0,1] for clearness calculation
        fused_norm = np.zeros(self.n_cols, dtype=np.float32)
        if fused_scores is not None:
            max_f = fused_scores.max()
            if max_f > 0:
                fused_norm = np.clip(fused_scores / max_f, 0.0, 1.0)

        # Enumerate all contiguous runs of free columns
        runs = []
        run_start = free[0]
        run_end   = free[0]
        for i in range(1, len(free)):
            if free[i] == free[i-1] + 1:
                run_end = free[i]
            else:
                runs.append((run_start, run_end))
                run_start = free[i]
                run_end   = free[i]
        runs.append((run_start, run_end))

        candidates = []
        for (cs, ce) in runs:
            width_cols = ce - cs + 1

            # Width score: fraction of total columns
            w_score = width_cols / self.n_cols

            # Clearness: how far below danger the gap columns are
            c_score = float(np.mean(1.0 - fused_norm[cs:ce+1]))

            # Centrality: peaks at 1.0 when gap centre == image centre
            gap_centre_col  = (cs + ce) / 2.0
            img_centre_col  = (self.n_cols - 1) / 2.0
            cent_score = 1.0 - 2.0 * abs(gap_centre_col - img_centre_col) / self.n_cols

            safety = (self.gap_w_width      * w_score   +
                      self.gap_w_clearness  * c_score   +
                      self.gap_w_centrality * cent_score)
            safety = float(np.clip(safety, 0.0, 1.0))

            center_x = int((cs + (ce - cs + 1) / 2.0) * col_width)

            candidates.append(GapCandidate(
                rank=0,
                center_x=center_x,
                col_start=cs,
                col_end=ce,
                width_cols=width_cols,
                safety=round(safety, 3),
                width_score=round(w_score, 3),
                clearness_score=round(c_score, 3),
                centrality_score=round(cent_score, 3),
            ))

        # Sort by safety descending, assign ranks
        candidates.sort(key=lambda g: g.safety, reverse=True)
        for i, g in enumerate(candidates[:top_n]):
            g.rank = i + 1

        return candidates[:top_n]

    # -------------------------------------------------------------------------
    def detect(self, img_bgr, telem=None):
        enhanced, roi_top, roi_bottom, roi_frame, frame_h, frame_w = self._stage1(img_bgr)
        rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        prev_pts, curr_pts, flow_vecs, flow_vecs_comp, flow_scores = \
            self._stage2_3(roi_frame)

        hough_boxes, hough_scores, hough_verticals, hough_edges = \
            self._stage4_hough(roi_frame)

        boxes, orange_mask, orange_scores = \
            self._stage5_orange(rotated, roi_top, roi_bottom)

        ground_mask, ground_green_scores, ground_obstacle_scores = \
            self._stage6_ground(rotated, roi_top, roi_bottom)

        active_flow   = flow_scores            * self.flow_weight   if self.use_flow   else np.zeros(self.n_cols, dtype=np.float32)
        active_hough  = hough_scores           * self.hough_weight  if self.use_hough  else np.zeros(self.n_cols, dtype=np.float32)
        active_orange = orange_scores          * self.orange_weight if self.use_orange else np.zeros(self.n_cols, dtype=np.float32)
        active_ground = ground_obstacle_scores * self.ground_weight if self.use_ground else np.zeros(self.n_cols, dtype=np.float32)

        fused_raw = active_flow + active_hough + active_orange + active_ground

        self._fused_scores_smooth = (
            self.iir_alpha * fused_raw +
            (1 - self.iir_alpha) * self._fused_scores_smooth
        )

        mean_f    = np.mean(self._fused_scores_smooth)
        std_f     = np.std(self._fused_scores_smooth)
        threshold = mean_f + self.flow_threshold * std_f

        obstacle_cols = [
            c for c in range(self.n_cols)
            if self._fused_scores_smooth[c] > threshold
        ]

        roi_w = roi_frame.shape[1]
        gap_candidates = self._find_gap_candidates(
            obstacle_cols, self._fused_scores_smooth, roi_w, top_n=3)

        gap_center_x = gap_candidates[0].center_x if gap_candidates else None

        return ObstacleResult(
            preprocessed=enhanced,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            roi_frame=roi_frame,
            flow_points_prev=prev_pts,
            flow_points_next=curr_pts,
            flow_vectors=flow_vecs,
            flow_vectors_comp=flow_vecs_comp,
            hough_boxes=hough_boxes,
            hough_scores=hough_scores,
            orange_boxes=boxes,
            orange_mask=orange_mask,
            orange_scores=orange_scores,
            ground_mask=ground_mask,
            ground_green_scores=ground_green_scores,
            ground_obstacle_scores=ground_obstacle_scores,
            flow_scores=flow_scores,
            fused_scores=self._fused_scores_smooth.copy(),
            obstacle_cols=obstacle_cols,
            gap_candidates=gap_candidates,
            gap_center_x=gap_center_x,
            n_cols=self.n_cols,
        )

    # -------------------------------------------------------------------------
    def draw(self, img_bgr, result):
        """
        4 rows × 3 cols = 12 slots, 10 used, last 2 black.

        Row 1: [1. Raw]          [2. CLAHE+ROI]      [3. Flow vectors]
        Row 2: [4. Flow scores]  [5. Hough poles]    [6. Orange detect]
        Row 3: [7. Ground mask]  [8. Ground scores]  [9. Fused scores]
        Row 4: [10. Final]       [11. (reserved)]    [12. (reserved)]
        """
        rotated      = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w         = rotated.shape[:2]
        enhanced_bgr = cv2.cvtColor(result.preprocessed, cv2.COLOR_GRAY2BGR)
        alpha        = 0.45
        roi_h        = result.roi_bottom - result.roi_top
        col_width    = w / result.n_cols

        # Gap arrow colours per rank
        GAP_COLORS = {1: (0, 255, 80), 2: (0, 200, 255), 3: (0, 120, 255)}

        def dim_outside_roi(panel):
            panel[:result.roi_top, :]    = (panel[:result.roi_top, :]    * alpha).astype(np.uint8)
            panel[result.roi_bottom:, :] = (panel[result.roi_bottom:, :] * alpha).astype(np.uint8)
            cv2.line(panel, (0, result.roi_top),    (w, result.roi_top),    (0, 255, 255), 1)
            cv2.line(panel, (0, result.roi_bottom), (w, result.roi_bottom), (0, 255, 255), 1)

        def draw_score_bars(panel, scores, color_obs, color_free, label,
                            obstacle_cols=None, draw_tint=False):
            if scores is None:
                cv2.putText(panel, label, (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                return
            mean_s = float(np.mean(scores))
            std_s  = float(np.std(scores))
            thresh = mean_s + self.flow_threshold * std_s

            for c in range(result.n_cols):
                x1     = int(c * col_width)
                x2     = int((c + 1) * col_width)
                is_obs = obstacle_cols is not None and c in obstacle_cols
                score  = float(scores[c])
                bar_h  = int(min(score / max(thresh * 1.5, 1e-6), 1.0) * roi_h)
                cv2.rectangle(panel,
                              (x1, result.roi_bottom - bar_h),
                              (x2 - 1, result.roi_bottom),
                              color_obs if is_obs else color_free, -1)
                if draw_tint and is_obs:
                    tint = panel[result.roi_top:result.roi_bottom, x1:x2].copy()
                    tint = (tint * 0.5 + np.array([0, 0, 80])).clip(0, 255).astype(np.uint8)
                    panel[result.roi_top:result.roi_bottom, x1:x2] = tint
                cv2.line(panel, (x2, result.roi_top), (x2, result.roi_bottom), (55, 55, 55), 1)

            thresh_y = result.roi_bottom - int(
                min(thresh / max(thresh * 1.5, 1e-6), 1.0) * roi_h)
            cv2.line(panel, (0, thresh_y), (w, thresh_y), (0, 180, 255), 1)
            cv2.putText(panel, "thr", (w - 28, thresh_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 255), 1)
            cv2.putText(panel, label, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_free, 1)

        def draw_gap_arrows(panel, small=False):
            """Draw all top-3 gap arrows with rank label and safety score."""
            for g in result.gap_candidates:
                color    = GAP_COLORS.get(g.rank, (180, 180, 180))
                gx       = g.center_x
                arrow_y  = result.roi_top + roi_h // 2
                size     = 20 if small else 30
                thickness = 1 if small else (3 if g.rank == 1 else 2)
                cv2.arrowedLine(panel,
                                (gx, arrow_y + size), (gx, arrow_y - size),
                                color, thickness, tipLength=0.35)
                label = f"#{g.rank} {g.safety:.2f}"
                cv2.putText(panel, label,
                            (gx - 18, arrow_y + size + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35 if small else 0.45, color,
                            1 if small else 1)

        # ── Panel 1 ───────────────────────────────────────────────────────────
        panel1 = rotated.copy()
        cv2.putText(panel1, "1. Raw (rotated)", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Panel 2 ───────────────────────────────────────────────────────────
        panel2 = enhanced_bgr.copy()
        dim_outside_roi(panel2)
        cv2.putText(panel2, "2. CLAHE + ROI", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ── Panel 3 ───────────────────────────────────────────────────────────
        panel3 = rotated.copy()
        dim_outside_roi(panel3)
        if result.flow_points_prev is not None and result.flow_vectors_comp is not None:
            for pt, vec in zip(result.flow_points_prev, result.flow_vectors_comp):
                x, y   = int(pt[0]), int(pt[1]) + result.roi_top
                dx, dy = int(vec[0] * 3), int(vec[1] * 3)
                cv2.arrowedLine(panel3, (x, y), (x + dx, y + dy),
                                (255, 200, 0), 1, tipLength=0.3)
                cv2.circle(panel3, (x, y), 2, (255, 200, 0), -1)
        cv2.putText(panel3,
                    f"3. Flow vectors [{'ON' if self.use_flow else 'OFF'}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 200, 0) if self.use_flow else (80, 80, 80), 1)

        # ── Panel 4 ───────────────────────────────────────────────────────────
        panel4 = rotated.copy()
        dim_outside_roi(panel4)
        draw_score_bars(panel4,
                        result.flow_scores if self.use_flow else None,
                        color_obs=(0, 60, 255), color_free=(0, 200, 0),
                        label=f"4. Flow scores [{'ON' if self.use_flow else 'OFF'}]")

        # ── Panel 5 ───────────────────────────────────────────────────────────
        panel5 = rotated.copy()
        dim_outside_roi(panel5)
        if result.hough_boxes and self.use_hough:
            for (x1, y1, x2, y2) in result.hough_boxes:
                fy1 = y1 + result.roi_top
                fy2 = y2 + result.roi_top
                cv2.rectangle(panel5, (x1, fy1), (x2, fy2), (100, 255, 100), 2)
                cx_box = (x1 + x2) // 2
                cv2.line(panel5, (cx_box, result.roi_top), (cx_box, result.roi_bottom),
                         (100, 255, 100), 1)
                cv2.putText(panel5, f"{x2-x1}x{y2-y1}", (x1, fy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        draw_score_bars(panel5,
                        result.hough_scores if self.use_hough else None,
                        color_obs=(0, 60, 255), color_free=(100, 255, 100),
                        label=f"5. Hough [{'ON' if self.use_hough else 'OFF'}]")

        # ── Panel 6 ───────────────────────────────────────────────────────────
        panel6 = rotated.copy()
        dim_outside_roi(panel6)
        if result.orange_mask is not None and self.use_orange:
            tint = panel6.copy()
            tint[result.orange_mask > 0] = [0, 100, 255]
            panel6 = cv2.addWeighted(panel6, 0.65, tint, 0.35, 0)
        if result.orange_boxes:
            for d in result.orange_boxes:
                bx, by = d.x, d.y + result.roi_top
                cv2.rectangle(panel6, (bx, by), (bx + d.w, by + d.h), (0, 200, 255), 2)
                cv2.circle(panel6, (d.cx, d.cy + result.roi_top), 4, (0, 0, 255), -1)
                cv2.line(panel6, (d.cx, result.roi_top), (d.cx, result.roi_bottom),
                         (0, 80, 255), 1)
                cv2.putText(panel6, f"cx={d.cx} ({d.confidence:.2f})",
                            (bx, max(by - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
        draw_score_bars(panel6,
                        result.orange_scores if self.use_orange else None,
                        color_obs=(0, 60, 255), color_free=(0, 140, 255),
                        label=f"6. Orange [{'ON' if self.use_orange else 'OFF'}]")

        # ── Panel 7 ───────────────────────────────────────────────────────────
        panel7 = rotated.copy()
        dim_outside_roi(panel7)
        if result.ground_mask is not None and self.use_ground:
            tint = panel7.copy()
            tint[result.ground_mask > 0] = [0, 200, 60]
            panel7 = cv2.addWeighted(panel7, 0.55, tint, 0.45, 0)
        cv2.putText(panel7,
                    f"7. Ground mask [{'ON' if self.use_ground else 'OFF'}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 60) if self.use_ground else (80, 80, 80), 1)

        # ── Panel 8 ───────────────────────────────────────────────────────────
        panel8 = rotated.copy()
        dim_outside_roi(panel8)
        draw_score_bars(panel8,
                        result.ground_obstacle_scores if self.use_ground else None,
                        color_obs=(0, 60, 255), color_free=(0, 200, 60),
                        label=f"8. Ground scores [{'ON' if self.use_ground else 'OFF'}]")

        # ── Panel 9: Fused scores + all 3 gap arrows ──────────────────────────
        panel9 = rotated.copy()
        dim_outside_roi(panel9)
        draw_score_bars(panel9, result.fused_scores,
                        color_obs=(0, 60, 255), color_free=(0, 220, 120),
                        label="9. Fused scores",
                        obstacle_cols=result.obstacle_cols, draw_tint=True)
        if result.gap_candidates:
            draw_gap_arrows(panel9, small=True)
        else:
            cv2.putText(panel9, "NO GAP",
                        (w // 2 - 45, result.roi_top + roi_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Panel 10: Final overlay ───────────────────────────────────────────
        panel10 = rotated.copy()

        for c in range(result.n_cols):
            x1     = int(c * col_width)
            x2     = int((c + 1) * col_width)
            is_obs = result.obstacle_cols is not None and c in result.obstacle_cols
            tint   = panel10[result.roi_top:result.roi_bottom, x1:x2].copy()
            if is_obs:
                tint = (tint * 0.45 + np.array([0, 0, 120])).clip(0, 255).astype(np.uint8)
            else:
                tint = (tint * 0.85 + np.array([0, 25, 0])).clip(0, 255).astype(np.uint8)
            panel10[result.roi_top:result.roi_bottom, x1:x2] = tint
            cv2.line(panel10, (x2, result.roi_top), (x2, result.roi_bottom), (40, 40, 40), 1)

        panel10[:result.roi_top, :]    = (panel10[:result.roi_top, :]    * alpha).astype(np.uint8)
        panel10[result.roi_bottom:, :] = (panel10[result.roi_bottom:, :] * alpha).astype(np.uint8)
        cv2.line(panel10, (0, result.roi_top),    (w, result.roi_top),    (0, 255, 255), 1)
        cv2.line(panel10, (0, result.roi_bottom), (w, result.roi_bottom), (0, 255, 255), 1)

        if result.gap_candidates:
            draw_gap_arrows(panel10, small=False)
            # Legend bottom-left
            ly = result.roi_bottom + 16
            for g in result.gap_candidates:
                color = GAP_COLORS.get(g.rank, (180, 180, 180))
                txt   = (f"#{g.rank} x={g.center_x}  cols={g.col_start}-{g.col_end}"
                         f"  W={g.width_score:.2f} C={g.clearness_score:.2f}"
                         f" Ctr={g.centrality_score:.2f}  safe={g.safety:.2f}")
                cv2.putText(panel10, txt, (5, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
                ly += 13
        else:
            cv2.putText(panel10, "NO GAP",
                        (w // 2 - 45, result.roi_top + roi_h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(panel10, "STOP",
                        (w // 2 - 35, result.roi_top + roi_h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        n_obs = len(result.obstacle_cols) if result.obstacle_cols else 0
        cv2.putText(panel10, f"10. Final  obs:{n_obs}/{result.n_cols}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Panels 11 & 12: Reserved ──────────────────────────────────────────
        panel11 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(panel11, "11. (reserved)", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        panel12 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(panel12, "12. (reserved)", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        # ── Assemble 4×3 grid ─────────────────────────────────────────────────
        dv = np.full((h, 3, 3),         40, dtype=np.uint8)
        dh = np.full((3, w * 3 + 6, 3), 40, dtype=np.uint8)

        row1 = np.hstack([panel1,  dv, panel2,  dv, panel3])
        row2 = np.hstack([panel4,  dv, panel5,  dv, panel6])
        row3 = np.hstack([panel7,  dv, panel8,  dv, panel9])
        row4 = np.hstack([panel10, dv, panel11, dv, panel12])

        return np.vstack([row1, dh, row2, dh, row3, dh, row4])
