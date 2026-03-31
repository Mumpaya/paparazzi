#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include <stdint.h>

// Tunable settings (exposed to GCS)
extern float green_max_speed;       // forward speed [m/s]
extern float green_heading_rate;    // turn rate [rad/s]
extern float edge_threshold;        // legacy edge threshold (px)

// Obstacle detector tunable settings (exposed to GCS)
extern float obs_score_threshold;   // obstacle score threshold [0..1]
extern float obs_ema_alpha;         // EMA smoothing factor [0..1]
extern float obs_w_ng;              // non-green weight
extern float obs_w_ed;              // edge-density weight
extern float obs_w_ll;              // line-length weight
extern float obs_min_size;          // minimum obstacle area fraction in front half
extern int   obs_danger_confirm_frames; // consecutive danger frames before reaction
extern int   obs_hsv_h_lo;
extern int   obs_hsv_h_hi;
extern int   obs_hsv_s_lo;
extern int   obs_hsv_s_hi;
extern int   obs_hsv_v_lo;
extern int   obs_hsv_v_hi;
extern void init_func(void);
extern void periodic_func(void);
extern void orange_avoider_guided_retreat(void);
extern void ground_calibration_start(void);

#endif
