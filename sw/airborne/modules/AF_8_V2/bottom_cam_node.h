/*
 * bottom_cam_node.h — Bottom camera ground edge detector
 *
 * Detects the ground edge (green mat boundary) from the bottom camera.
 * Outputs:
 *   • centroid (x, y) — pixel position of detected ground edge
 *   • goodness — confidence score [0, 1]
 *   • magnitude — distance from frame center to centroid (pixels)
 *   • over_edge — boolean flag if centroid is beyond threshold
 *
 * The magnitude is used by control to generate steering commands:
 * if magnitude > threshold, turn towards the centroid direction.
 */

#ifndef BOTTOM_CAM_NODE_H
#define BOTTOM_CAM_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "detection_types.h"

/* ── Initialization ──────────────────────────────────────────────── */
void bottom_cam_node_init(void);

/* ── Main detection function ─────────────────────────────────────── */
void bottom_cam_node_process(const uint8_t *bgr_data, int width, int height,
                              BottomCamResult *result);

#ifdef __cplusplus
}
#endif

#endif /* BOTTOM_CAM_NODE_H */
