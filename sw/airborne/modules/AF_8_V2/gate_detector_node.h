/*
 * gate_detector_node.h — Blue-gate detector (C++ / OpenCV node)
 *
 * Detects a pair of blue pillars, validates them as a gate,
 * estimates approach angle and distance.
 * Writes results into a GateResult struct from detection_types.h.
 */
#ifndef GATE_DETECTOR_NODE_H
#define GATE_DETECTOR_NODE_H

#include "detection_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Call once at startup */
void gate_detector_node_init(void);

/*
 * Process one BGR frame (already rotated 90° CCW if needed — the
 * caller is responsible for passing the correct orientation).
 *
 * @param bgr_data   pointer to BGR pixel data (row-major)
 * @param width       image width
 * @param height      image height
 * @param out         result struct to fill
 */
void gate_detector_node_process(const uint8_t *bgr_data,
                                int width, int height,
                                GateResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GATE_DETECTOR_NODE_H */
