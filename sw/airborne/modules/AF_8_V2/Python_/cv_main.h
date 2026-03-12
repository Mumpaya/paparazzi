/*
 * cv_main.h — Central computer-vision module for AF_8_V2
 *
 * Registers a single camera callback that runs every frame,
 * invokes the three detector nodes (ground-edge, obstacle, gate),
 * and populates the global FrameResults struct.
 *
 * The controller (controller.cpp) reads frame_results each periodic.
 */
#ifndef CV_MAIN_H
#define CV_MAIN_H

#include "detection_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The single global result that all other modules can read.
 * Written by the camera callback, read by controller_periodic. */
extern volatile FrameResults frame_results;

/* Called once from module init (registers CV callback on front_camera) */
void cv_main_init(void);

/* Called periodically — currently a no-op (all work is in the camera
   callback) but required by Paparazzi module skeleton. */
void cv_main_periodic(void);

#ifdef __cplusplus
}
#endif

#endif /* CV_MAIN_H */
