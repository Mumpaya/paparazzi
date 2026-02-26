import os
import cv2
import numpy as np


class CameraCalibration:
    """
    Fisheye camera calibration using OpenCV's fisheye model.

    Usage:
        # --- Run calibration once and save ---
        cal = CameraCalibration(images_dir="path/to/checkerboard/images")
        cal.calibrate()
        cal.save("calibration_data.npz")

        # --- Load and use ---
        cal = CameraCalibration()
        cal.load("calibration_data.npz")
        undistorted = cal.undistort(frame)
    """

    # Inner corners: 9 along 520px axis, 6 along 240px axis
    PATTERN_SIZE  = (9, 6)
    SQUARE_SIZE_MM = 35.15

    def __init__(self, images_dir: str = None):
        self.images_dir = images_dir
        self.K = None   # Camera intrinsic matrix (3x3)
        self.D = None   # Fisheye distortion coefficients (4x1)
        self.image_size = None  # (width, height)

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self):
        """Detect checkerboard corners in all images and run fisheye calibration."""
        assert self.images_dir is not None, "images_dir must be set to run calibration"

        # 3D object points for one checkerboard view (Z=0 plane)
        objp = np.zeros((self.PATTERN_SIZE[0] * self.PATTERN_SIZE[1], 1, 3), np.float64)
        objp[:, 0, :2] = np.mgrid[
            0:self.PATTERN_SIZE[0],
            0:self.PATTERN_SIZE[1]
        ].T.reshape(-1, 2) * self.SQUARE_SIZE_MM

        obj_points = []   # 3D points in real world space
        img_points = []   # 2D points in image plane

        image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        assert len(image_files) > 0, f"No images found in {self.images_dir}"

        print(f"Found {len(image_files)} calibration images.")
        successful = 0

        for fname in image_files:
            img = cv2.imread(os.path.join(self.images_dir, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.image_size is None:
                self.image_size = (gray.shape[1], gray.shape[0])  # (width, height)

            ret, corners = cv2.findChessboardCorners(
                gray,
                self.PATTERN_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # Refine corner positions to subpixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners_refined)
                successful += 1
            else:
                print(f"  [WARN] Corners not found in {fname}")

        print(f"Calibration using {successful}/{len(image_files)} images.")
        assert successful >= 5, "Not enough valid images for calibration (need at least 5)"

        # Fisheye calibration flags
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(successful)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(successful)]

        rms, self.K, self.D, _, _ = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            self.image_size,
            K,
            D,
            rvecs,
            tvecs,
            flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        print(f"Calibration RMS reprojection error: {rms:.4f} px")
        print(f"K:\n{self.K}")
        print(f"D:\n{self.D}")

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save calibration parameters to a .npz file."""
        assert self.K is not None, "No calibration data to save. Run calibrate() first."
        np.savez(path, K=self.K, D=self.D, image_size=np.array(self.image_size))
        print(f"Calibration saved to {path}")

    def load(self, path: str):
        """Load calibration parameters from a .npz file."""
        data = np.load(path)
        self.K = data["K"]
        self.D = data["D"]
        self.image_size = tuple(data["image_size"])
        print(f"Calibration loaded from {path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Undistort
    # ──────────────────────────────────────────────────────────────────────────

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Undistort a fisheye frame using the loaded/computed calibration."""
        assert self.K is not None, "No calibration data. Run calibrate() or load() first."

        h, w = frame.shape[:2]
        # Compute optimal new camera matrix to retain full image area
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D,
            (w, h), np.eye(3),
            balance=0.0   # 0=crop all black, 1=keep all pixels (with black borders)
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K,
            (w, h), cv2.CV_16SC2
        )
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)