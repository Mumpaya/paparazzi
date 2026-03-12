import cv2
import numpy as np
import matplotlib.pyplot as plt


class PoleDetector:
    """
    Detects vertical poles in undistorted, upright (rotated) images.

    Pipeline:
        1. Grayscale + optional CLAHE + Gaussian blur
        2. Canny edge detection
        3. Probabilistic Hough line transform
        4. Filter for near-vertical lines
        5. Cluster nearby lines into candidate boxes
        6. Filter boxes by aspect ratio, height, and border proximity
    """

    def __init__(
        self,
        # Preprocessing
        blur_kernel:      int   = 5,
        use_clahe:        bool  = True,
        clahe_clip:       float = 2.0,
        clahe_grid:       int   = 8,
        # Canny
        canny_low:        int   = 50,
        canny_high:       int   = 150,
        # Hough
        hough_threshold:  int   = 30,
        hough_min_length: int   = 40,
        hough_max_gap:    int   = 100,
        # Line filtering
        vertical_tol_deg: float = 20.0,
        # Box clustering
        cluster_gap:      int   = 35,
        max_box_width: int = 200,  # reject clusters wider than this (px)
        # Box filtering
        min_aspect_ratio: float = 1.5,    # min height/width for a valid pole box
        max_aspect_ratio: float = 8.0,  # reject anything taller than 8× its width
        min_height_frac:  float = 0.0,   # min box height as fraction of image height
        border_margin:    int   = 15,     # ignore boxes this close to left/right edge (px)
        max_interior_edge_density: float = 0.15,  # reject boxes with too many internal edges
    ):
        self.blur_kernel      = blur_kernel
        self.use_clahe        = use_clahe
        self.clahe_clip       = clahe_clip
        self.clahe_grid       = clahe_grid
        self.canny_low        = canny_low
        self.canny_high       = canny_high
        self.hough_threshold  = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap    = hough_max_gap
        self.vertical_tol_deg = vertical_tol_deg
        self.cluster_gap      = cluster_gap
        self.max_box_width = max_box_width
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height_frac  = min_height_frac
        self.border_margin    = border_margin
        self.max_interior_edge_density = max_interior_edge_density

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_grid, self.clahe_grid)
            )
            gray = clahe.apply(gray)
        return cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

    def _edges(self, blurred: np.ndarray) -> np.ndarray:
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def _hough_lines(self, edges: np.ndarray):
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

    def _filter_vertical(self, lines) -> list:
        vertical = []
        if lines is None:
            return vertical
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            angle_deg = np.degrees(np.arctan2(dx, dy))  # 0° = perfectly vertical
            if angle_deg <= self.vertical_tol_deg:
                vertical.append((x1, y1, x2, y2))
        return vertical

    # def _cluster_lines(self, vertical_lines: list) -> list:
    #     """Merge horizontally nearby vertical lines into bounding boxes."""
    #     if not vertical_lines:
    #         return []

    #     sorted_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)

    #     clusters = []
    #     current = [sorted_lines[0]]
    #     for line in sorted_lines[1:]:
    #         prev_x = (current[-1][0] + current[-1][2]) / 2
    #         curr_x = (line[0] + line[2]) / 2
    #         if abs(curr_x - prev_x) <= self.cluster_gap:
    #             current.append(line)
    #         else:
    #             clusters.append(current)
    #             current = [line]
    #     clusters.append(current)

    #     boxes = []
    #     for cluster in clusters:
    #         xs = [p for l in cluster for p in (l[0], l[2])]
    #         ys = [p for l in cluster for p in (l[1], l[3])]
    #         boxes.append((min(xs), min(ys), max(xs), max(ys)))
    #     return boxes

    def _cluster_lines(self, vertical_lines: list) -> list:
        """
        Merge horizontally nearby vertical lines into bounding boxes.
        Splits clusters wider than max_box_width into two halves.
        """
        if not vertical_lines:
            return []

        sorted_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)

        clusters = []
        current = [sorted_lines[0]]
        for line in sorted_lines[1:]:
            prev_x = (current[-1][0] + current[-1][2]) / 2
            curr_x = (line[0] + line[2]) / 2
            if abs(curr_x - prev_x) <= self.cluster_gap:
                current.append(line)
            else:
                clusters.append(current)
                current = [line]
        clusters.append(current)

        boxes = []
        for cluster in clusters:
            xs = [p for l in cluster for p in (l[0], l[2])]
            ys = [p for l in cluster for p in (l[1], l[3])]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if (x_max - x_min) <= self.max_box_width:
                boxes.append((x_min, y_min, x_max, y_max))
            else:
                # Split into two halves at the midpoint
                x_mid = (x_min + x_max) // 2
                left_lines  = [l for l in cluster if (l[0] + l[2]) / 2 <= x_mid]
                right_lines = [l for l in cluster if (l[0] + l[2]) / 2 >  x_mid]
                for half in [left_lines, right_lines]:
                    if not half:
                        continue
                    hxs = [p for l in half for p in (l[0], l[2])]
                    hys = [p for l in half for p in (l[1], l[3])]
                    boxes.append((min(hxs), min(hys), max(hxs), max(hys)))

        return boxes

    def _filter_boxes(self, boxes: list, edges: np.ndarray, img_h: int, img_w: int) -> list:
        """
        Remove boxes that are:
        - Too close to the left/right border (partially visible)
        - Too short relative to image height
        - Wrong aspect ratio (not pole-like)
        - Too cluttered inside (background objects have many internal edges)
        """
        valid = []
        for (x1, y1, x2, y2) in boxes:
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)

            # Skip boxes touching the border
            if x1 < self.border_margin or x2 > img_w - self.border_margin:
                continue

            # Skip boxes that are too short
            if h < self.min_height_frac * img_h:
                continue

            # Skip boxes outside aspect ratio range
            ratio = h / w
            if ratio < self.min_aspect_ratio or ratio > self.max_aspect_ratio:
                continue

            # ── Interior edge density check ───────────────────────────────────
            # Shrink the box inward to avoid counting the pole's own edges
            shrink = 10
            ix1 = min(x1 + shrink, x2)
            ix2 = max(x2 - shrink, x1)
            iy1 = min(y1 + shrink, y2)
            iy2 = max(y2 - shrink, y1)

            interior_w = ix2 - ix1
            interior_h = iy2 - iy1

            if interior_w <= 0 or interior_h <= 0:
                # Box too narrow to have a meaningful interior — keep it
                valid.append((x1, y1, x2, y2))
                continue

            interior_area  = interior_w * interior_h
            interior_edges = edges[iy1:iy2, ix1:ix2]
            edge_density   = np.count_nonzero(interior_edges) / interior_area

            if edge_density > self.max_interior_edge_density:
                continue  # Too cluttered inside — likely background

            valid.append((x1, y1, x2, y2))
        return valid

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, img_bgr: np.ndarray) -> dict:
        img_h, img_w = img_bgr.shape[:2]

        blurred        = self._preprocess(img_bgr)
        edges          = self._edges(blurred)
        lines          = self._hough_lines(edges)
        vertical       = self._filter_vertical(lines)
        raw_boxes      = self._cluster_lines(vertical)
        filtered_boxes = self._filter_boxes(raw_boxes, edges, img_h, img_w)  # ← edges passed here

        return {
            "blurred":        blurred,
            "edges":          edges,
            "all_lines":      lines,
            "vertical_lines": vertical,
            "raw_boxes":      raw_boxes,
            "boxes":          filtered_boxes,
        }

    def draw(self, img_bgr: np.ndarray, result: dict) -> np.ndarray:
        out = img_bgr.copy()

        # Vertical line segments — green
        for (x1, y1, x2, y2) in result["vertical_lines"]:
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Raw (unfiltered) candidate boxes — yellow
        accepted = set(result["boxes"])
        for box in result["raw_boxes"]:
            if box not in accepted:
                x1, y1, x2, y2 = box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)  # yellow, thin
                cv2.putText(out, f"{x2-x1}x{y2-y1}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Accepted pole boxes — red
        for (x1, y1, x2, y2) in result["boxes"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f"{x2-x1}x{y2-y1}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return out

    def visualize(self, img_bgr: np.ndarray, fname: str = ""):
        result  = self.detect(img_bgr)
        overlay = self.draw(img_bgr, result)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"Pole Detection  —  {fname}", fontsize=11)

        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input")

        axes[1].imshow(result["blurred"], cmap="gray")
        axes[1].set_title(f"{'CLAHE + ' if self.use_clahe else ''}Blurred")

        axes[2].imshow(result["edges"], cmap="gray")
        axes[2].set_title(f"Canny (low={self.canny_low}, high={self.canny_high})")

        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title(
            f"Detections: {len(result['boxes'])} pole(s)\n"
            f"(raw boxes: {len(result['raw_boxes'])} → filtered: {len(result['boxes'])})"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=False)

        return result