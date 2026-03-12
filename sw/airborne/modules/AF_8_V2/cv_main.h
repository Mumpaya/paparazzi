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

#ifdef __cplusplus
}
#endif

#endif /* CV_MAIN_H */
