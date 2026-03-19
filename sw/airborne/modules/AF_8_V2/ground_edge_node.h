/*
 * ground_edge_node.h - Ground edge detector node
 */

#ifndef GROUND_EDGE_NODE_H
#define GROUND_EDGE_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "detection_types.h"

void ground_edge_node_init(void);
void ground_edge_node_process(const uint8_t *bgr_data, int width, int height,
                               GroundEdgeResult *result);

#ifdef __cplusplus
}
#endif

#endif /* GROUND_EDGE_NODE_H */
