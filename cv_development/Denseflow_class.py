import numpy as np
import cv2 as cv

class DenseOpticalFlowManager:
    def __init__(self, mag_threshold=2.5, max_mag_cap=15.0, f_constant=50.0):
        self.mag_threshold = mag_threshold
        self.max_mag_cap = max_mag_cap
        self.f_constant = f_constant
        self.old_gray = None

    def process_frame(self, frame, yaw_rate, roi_box):
        """
        Calculates dense flow ONLY within the roi_box to save computation.
        roi_box: [x_min, y_min, x_max, y_max]
        Returns: mag_filtered (cropped), density_map (cropped), heatmap (full size), roi_intensity
        """
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = frame_gray.shape
        
        # 1. Coordinate Validation & Clipping
        x1, y1, x2, y2 = map(int, roi_box)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        # 2. Frame Cropping (The Performance Secret)
        # We only look at the slice of the image where the object is
        curr_roi = frame_gray[y1:y2, x1:x2]

        # Initialize or handle first frame
        if self.old_gray is None or self.old_gray.shape != frame_gray.shape:
            self.old_gray = frame_gray
            return None, None, None, 0.0

        prev_roi = self.old_gray[y1:y2, x1:x2]

        # 3. Calculate Dense Flow ONLY on the small crop
        # Farneback is much faster on a 200x200 crop than the full 1280x720
        flow = cv.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 4. Ego-Motion Compensation (Applied to cropped flow)
        flow[..., 0] -= (yaw_rate * self.f_constant)

        # 5. Intensity Filtering
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mag_filtered = np.where(mag < self.mag_threshold, 0, mag)
        mag_filtered = np.where(mag_filtered > self.max_mag_cap, self.max_mag_cap, mag_filtered)

        # 6. Generate Density Map (Cropped)
        mag_norm = cv.normalize(mag_filtered, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        density_roi = cv.GaussianBlur(mag_norm, (15, 15), 0) # Smaller kernel for smaller area
        
        # 7. Visualization (Full Frame Heatmap)
        # We create a black image and place the ROI heatmap into it
        full_heatmap = np.zeros_like(frame)
        if density_roi.size > 0:
            roi_heatmap = cv.applyColorMap(density_roi, cv.COLORMAP_JET)
            full_heatmap[y1:y2, x1:x2] = roi_heatmap
            roi_intensity = np.mean(density_roi) / 10.0
        else:
            roi_intensity = 0.0

        # Update state with the full gray frame for the next iteration's crop
        self.old_gray = frame_gray
        
        return mag_filtered, density_roi, full_heatmap, roi_intensity