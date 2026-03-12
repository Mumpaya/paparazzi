"""
ground_projection.py

Estimate obstacle distance from camera using bounding box + pixel count.
"""

from dataclasses import dataclass
import math


@dataclass
class CameraModel:
    focal_length_mm: float
    sensor_height_mm: float
    sensor_width_mm: float
    image_height_px: int
    image_width_px: int
    height_above_ground_m: float
    pitch_rad: float = 0.0  # camera pitch (down = positive)
    yaw_rad: float = 0.0
    roll_rad: float = 0.0

    @property
    def fy(self) -> float:
        return (self.focal_length_mm / self.sensor_height_mm) * self.image_height_px

    @property
    def fx(self) -> float:
        return (self.focal_length_mm / self.sensor_width_mm) * self.image_width_px


@dataclass
class ObstacleBox:
    x: float
    y: float
    w: float
    h: float
    pixel_count: int = None

    @property
    def bottom_center(self):
        return (self.x + self.w / 2.0, self.y + self.h)


class GroundProjection:
    def __init__(self, camera: CameraModel):
        self.cam = camera

    def estimate_distance_by_height(self, obj_real_height_m: float, box_height_px: float) -> float:
        """
        Estimate distance from camera to object using pinhole formula and known object height.
        Returns distance along optical axis (m).
        """
        if box_height_px <= 0:
            raise ValueError("box_height_px must be > 0")
        return (obj_real_height_m * self.cam.fy) / box_height_px

    def project_ground_distance_from_box(self, box: ObstacleBox) -> float:
        """
        Estimate ground distance to obstacle’s contact point using camera height and pitch.
        Uses the pixel row of the box bottom.
        """
        _, y_bottom = box.bottom_center
        # normalized image coordinate (0 top, 1 bottom)
        v = y_bottom / self.cam.image_height_px
        # angle from optical axis down to point (assuming principal point center)
        # v - 0.5: top/bottom from center
        vertical_angle_rad = math.atan2((v - 0.5) * self.cam.image_height_px, self.cam.fy)
        # world elevation ray angle
        total_angle = self.cam.pitch_rad + vertical_angle_rad
        if math.isclose(math.sin(total_angle), 0.0, abs_tol=1e-9):
            return float("inf")
        if total_angle <= 0:
            # viewing up or horizontal, cannot reliably intersect ground
            return float("inf")
        # simple flat-ground intersection along camera forward direction
        return self.cam.height_above_ground_m / math.tan(total_angle)

    def box_pixel_density(self, box: ObstacleBox) -> float:
        """
        Returns obstacle pixel density in box = pixels object / box area.
        Can be used for consistency/quality checks.
        """
        if box.w <= 0 or box.h <= 0:
            return 0.0
        area = box.w * box.h
        if not box.pixel_count:
            return 0.0
        return box.pixel_count / area


if __name__ == "__main__":
    # Example inputs (tune to your camera)
    cam = CameraModel(
        focal_length_mm=4.0,
        sensor_height_mm=2.76,
        sensor_width_mm=3.68,
        image_height_px=720,
        image_width_px=1280,
        height_above_ground_m=1.2,
        pitch_rad=math.radians(20.0),
    )
    proj = GroundProjection(cam)

    obstacle = ObstacleBox(x=560, y=350, w=160, h=220, pixel_count=30000)

    # 1) estimate distance by known real object height
    known_height = 1.7
    dist_optical = proj.estimate_distance_by_height(known_height, obstacle.h)
    print(f"Distance from object real height: {dist_optical:.2f} m")

    # 2) ground intercept distance from bottom of bounding box
    dist_ground = proj.project_ground_distance_from_box(obstacle)
    print(f"Ground projected distance (bottom pixel): {dist_ground:.2f} m")

    # 3) sensor density
    density = proj.box_pixel_density(obstacle)
    print(f"Pixel density in box: {density:.3f} px/px")