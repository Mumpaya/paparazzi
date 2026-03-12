/*
 * obstacle_detector_node.h — Multi-stage obstacle detector (C++ / OpenCV node)
 *
 * Stages: Rotate+CLAHE  →  Optical-flow  →  Hough-poles  →  Orange-HSV
 *         →  Green-ground  →  Score-fusion  →  Gap-finding
 * Writes results into an ObstacleResult struct from detection_types.h.
 */
#ifndef OBSTACLE_DETECTOR_NODE_H
#define OBSTACLE_DETECTOR_NODE_H

#include "detection_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Call once at startup */
void obstacle_detector_node_init(void);

/*
 * Process one BGR frame (original orientation — the node rotates internally).
 * @param bgr_data   pointer to BGR pixel data (row-major)
 * @param width       image width  (original, before rotation)
 * @param height      image height (original, before rotation)
 * @param out         result struct to fill
 */
void obstacle_detector_node_process(const uint8_t *bgr_data,
                                    int width, int height,
                                    ObstacleResult *out);

#ifdef __cplusplus
}
#endif

#endif /* OBSTACLE_DETECTOR_NODE_H */
