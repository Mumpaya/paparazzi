/*
 * Copyright (C) 2026
 *
 * This file is part of Paparazzi.
 */

#ifndef CV_BOTTOM_FLOW_VELOCITY_H
#define CV_BOTTOM_FLOW_VELOCITY_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

struct cv_bottom_flow_velocity_state_t {
  float vx_mps;
  float vy_mps;
  float flow_px_x;
  float flow_px_y;
  uint16_t tracked_points;
  bool updated;
  bool valid;
};

extern struct cv_bottom_flow_velocity_state_t cv_bottom_flow_velocity_state;

extern float cv_bfv_altitude_m;
extern float cv_bfv_fov_deg;
extern uint16_t cv_bfv_img_width_ref;
extern uint8_t cv_bfv_min_points;
extern float cv_bfv_max_flow_px;
extern bool cv_bfv_draw;

extern void cv_bottom_flow_velocity_init(void);
extern void cv_bottom_flow_velocity_periodic(void);
#ifdef __cplusplus
}
#endif
#endif /* CV_BOTTOM_FLOW_VELOCITY_H */

