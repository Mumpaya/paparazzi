
#ifndef GATE_DETECTOR_NODE_H
#define GATE_DETECTOR_NODE_H

#include "detection_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Call once at startup */
void gate_detector_node_init(void);

void gate_detector_node_process(const uint8_t *bgr_data,
                                int width, int height,
                                GateResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GATE_DETECTOR_NODE_H */
