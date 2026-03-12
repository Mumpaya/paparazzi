#ifndef GATE_DETECTOR_H
#define GATE_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

// These are the functions the Paparazzi C core will call
void gate_detector_init(void);
void gate_detector_periodic(void);

#ifdef __cplusplus
}
#endif

#endif