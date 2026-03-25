/*
 * cv_main.h — AF_8_V2 central CV pipeline — Paparazzi module interface
 *
 * This module:
 *   1. Registers a callback on front_camera (YUV422).
 *   2. Converts to BGR, rotates 90° CCW.
 *   3. Runs ground-edge, obstacle and gate detector nodes.
 *   4. Runs the heading/velocity controller state machine.
 *   5. Publishes results in the global `frame_results` struct.
 *   6. If compiled with AF_8_V2_DRAW=1, draws the detection overlays
 *      and the controller compass back onto the camera image buffer
 *      so the video stream shows the annotations live.
 *
 * ── Compile flags ─────────────────────────────────────────────────
 *   AF_8_V2_DRAW   (default 1)
 *       Set to 0 to skip all drawing (saves ~1 ms per frame on Bebop).
 *
 * ── Usage from controller / flight plan ──────────────────────────
 *   #include "modules/AF_8_V2/cv_main.h"
 *
 *   // read latest results
 *   FrameResults  fr  = *((FrameResults*)&frame_results);
 *   ControlOutput cmd = *((ControlOutput*)&ctrl_output);
 */

#ifndef CV_MAIN_H
#define CV_MAIN_H

#include "detection_types.h"
#include "controller.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── global results (written by camera callback, read by controller) ── */

/** Latest detection results — updated every camera frame. */
extern volatile FrameResults frame_results;

/** Latest controller output — updated every camera frame (after detectors). */
extern volatile ControlOutput ctrl_output;

/* ── Paparazzi module entry points ──────────────────────────────── */
void cv_main_init(void);
void cv_main_periodic(void);

/* ── cycle-time profiling (µs) — exposed as dl_settings ─────────── */
extern volatile uint32_t t_ground_us;
extern volatile uint32_t t_obstacle_us;
extern volatile uint32_t t_gate_us;
extern volatile uint32_t t_bottom_us;
extern volatile uint32_t t_controller_us;
extern volatile uint32_t t_draw_us;
extern volatile uint32_t t_total_cb_us;


#ifdef __cplusplus
}
#endif

/* ── datalink-tunable settings ── */
extern int   af8_draw_overlay;
extern float af8_k_yaw;
extern float af8_v_std;
extern float af8_v_gate;
extern float af8_bottom_cam_steer_threshold;  /* magnitude threshold for bottom cam steering */

#endif /* CV_MAIN_H */
