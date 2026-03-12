/*
 * ground_edge_node.h — Ground-edge boundary detector (C++ / OpenCV node)
 *
 * Detects green-ground boundaries via HSV masking + HoughLinesP,
 * applies temporal confirmation and mat-edge rejection.
 * Writes results into a GroundEdgeResult struct from detection_types.h.
 */
#ifndef GROUND_EDGE_NODE_H
#define GROUND_EDGE_NODE_H

#include "detection_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Call once at startup */
void ground_edge_node_init(void);

/*
 * Process one BGR frame.
 * @param bgr_data   pointer to BGR pixel data (row-major)
 * @param width       image width
 * @param height      image height
 * @param out         result struct to fill
 */
void ground_edge_node_process(const uint8_t *bgr_data,
                              int width, int height,
                              GroundEdgeResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GROUND_EDGE_NODE_H */
