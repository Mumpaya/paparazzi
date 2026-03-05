#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Settings accessible by Paparazzi GCS
extern float max_speed;
extern float heading_rate;
extern float danger_threshold;
extern float gate_target_dist;

// Main functions called by Paparazzi
void controller_init(void);
void controller_periodic(void);
void orange_avoider_guided_retreat(void);

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_H */