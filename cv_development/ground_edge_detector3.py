import cv2
import numpy as np
from dataclasses import dataclass


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

        cv2.putText(out, f"Goodness: {result.goodness}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255) if result.over_edge else (100, 100, 100), 2)

        return out


@dataclass
class GroundFlowResult:
    base_im: np.ndarray
    points: np.ndarray
    flows: np.ndarray
    avg_flow: np.ndarray


class GroundFlow:
    def __init__(
        self,
        max_corners: int = 120,
        quality_level: float = 0.01,
        min_distance: int = 7,
        block_size: int = 7,
        reseed_min_points: int = 20,
    ):
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.reseed_min_points = reseed_min_points
        self._prev_base_im = None
        self._prev_pts = None

    def _detect_features(self, gray_im: np.ndarray):
        return cv2.goodFeaturesToTrack(gray_im, mask=None, **self.feature_params)

    def measure(self, bgr: np.ndarray) -> GroundFlowResult:
        base_im = bgr[::, ::, 1]
        # base_im = cv2.GaussianBlur(base_im, (9, 9), 0)
        base_im = cv2.convertScaleAbs(base_im, alpha=1.3, beta=0)

        points = np.empty((0, 2), dtype=np.float32)
        flows = np.empty((0, 2), dtype=np.float32)
        avg_flow = np.array([0.0, 0.0], dtype=np.float32)

        # Bootstrap with the first frame: only detect seed points.
        if self._prev_base_im is None:
            self._prev_base_im = base_im
            self._prev_pts = self._detect_features(base_im)
            return GroundFlowResult(base_im=base_im, points=points, flows=flows, avg_flow=avg_flow)

        # Track points from previous frame into current frame.
        if self._prev_pts is not None and len(self._prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_base_im,
                base_im,
                self._prev_pts,
                None,
                **self.lk_params,
            )

            if next_pts is not None and status is not None:
                valid = status.reshape(-1) == 1
                good_old = self._prev_pts.reshape(-1, 2)[valid]
                good_new = next_pts.reshape(-1, 2)[valid]

                if len(good_new) > 0:
                    points = good_old.astype(np.float32)
                    flows = (good_new - good_old).astype(np.float32)
                    avg_flow = np.mean(flows, axis=0)


        # Always reseed fresh points on the current frame for the next call.
        self._prev_pts = self._detect_features(base_im)
        self._prev_base_im = base_im

        return GroundFlowResult(base_im=base_im, points=points, flows=flows, avg_flow=avg_flow)

    def draw(self, bgr: np.ndarray, result: GroundFlowResult) -> np.ndarray:
        out = cv2.cvtColor(result.base_im, cv2.COLOR_GRAY2BGR)

        for point, flow in zip(result.points, result.flows):
            start = (int(round(point[0])), int(round(point[1])))
            end = (int(round(point[0] + flow[0])), int(round(point[1] + flow[1])))
            cv2.arrowedLine(out, start, end, (255, 120, 0), 1, tipLength=0.25)

        # Draw average flow vector from center of image
        h, w = result.base_im.shape
        center = (w // 2, h // 2)
        avg_end = (int(round(center[0] + result.avg_flow[0] * 5)), int(round(center[1] + result.avg_flow[1] * 5)))
        cv2.arrowedLine(out, center, avg_end, (0, 0, 255), 2, tipLength=0.3)

        dt = 0.1
        alt = 1.25
        vel = (result.avg_flow / dt) *((2*np.tan(np.deg2rad(42)/2)*alt)/(240))
        vel_mag = np.linalg.norm(vel)
        cv2.putText(out, f"Avg Flow: ({result.avg_flow[0]:.1f}, {result.avg_flow[1]:.1f})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, f"vel: ({vel[0]:.1f}, {vel[1]:.1f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(out, f"mag: {vel_mag:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return out