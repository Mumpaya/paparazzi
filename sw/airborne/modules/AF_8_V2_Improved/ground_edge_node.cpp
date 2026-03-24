/*
 * ground_edge_node.cpp - Ground edge detector node
 *
 * Stub implementation that returns empty results.
 * The main AF_8_V2 pipeline focuses on gate/obstacle detection.
 */

#include "ground_edge_node.h"
#include <cstring>

void ground_edge_node_init(void)
{
    /* No initialization needed */
}

void ground_edge_node_process(const uint8_t *bgr_data, int width, int height,
                               GroundEdgeResult *result)
{
    if (!result) return;
    
    /* Return empty result */
    memset(result, 0, sizeof(*result));
    result->has_lines = false;
    result->n_confirmed = 0;
    result->n_rejected = 0;
}
