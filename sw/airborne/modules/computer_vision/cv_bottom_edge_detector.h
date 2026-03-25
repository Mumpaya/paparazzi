/*
 * Copyright (C) 2026
 *
 * This file is part of Paparazzi.
 */

#ifndef CV_BOTTOM_EDGE_DETECTOR_H
#define CV_BOTTOM_EDGE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

struct cv_bottom_edge_detector_state_t {
  float goodness;
  int16_t x_c;
  int16_t y_c;
  bool over_edge;
  bool updated;
  bool valid;
};

extern struct cv_bottom_edge_detector_state_t cv_bottom_edge_detector_state;

extern uint8_t cbed_h_min;
extern uint8_t cbed_h_max;
extern uint8_t cbed_s_min;
extern uint8_t cbed_s_max;
extern uint8_t cbed_v_min;
extern uint8_t cbed_v_max;
extern uint8_t cbed_notgreen_h_min;
extern uint8_t cbed_notgreen_h_max;
extern uint8_t cbed_notgreen_s_min;
extern uint8_t cbed_notgreen_s_max;
extern uint8_t cbed_notgreen_v_min;
extern uint8_t cbed_notgreen_v_max;
extern uint8_t cbed_nonuniform_thresh;
extern uint8_t cbed_edge_radius_ratio_pct;
extern bool cbed_draw;

extern void cv_bottom_edge_detector_init(void);
extern void cv_bottom_edge_detector_periodic(void);

#endif /* CV_BOTTOM_EDGE_DETECTOR_H */

