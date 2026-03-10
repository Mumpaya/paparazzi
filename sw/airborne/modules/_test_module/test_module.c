/*
 * Copyright (C) 2026 Douwe Rijs <Douwe_r@standofl.nl>
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

/** @file "modules/_test_module/test_module.c"
 * @author Douwe Rijs <Douwe_r@standofl.nl>
 * Edge-based floor avoider: uses OpenCV Canny edge detection on the bottom
 * camera.  The floor (green carpet) is smooth with few edges, while walls,
 * nets, poles, and other obstacles produce many edges.  When the edge
 * density rises above a threshold the drone turns away from the side with
 * more edges.  Runs in GUIDED mode.
 */

#include "modules/_test_module/test_module.h"
#include "modules/_test_module/edge_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "state.h"
#include "generated/airframe.h"
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define TEST_MODULE_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[edge_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if TEST_MODULE_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// ── Configurable defines ──────────────────────────────────────────────────────
#ifndef EDGE_DETECTOR_CAMERA
#define EDGE_DETECTOR_CAMERA bottom_camera
#endif

#ifndef EDGE_DETECTOR_FPS
#define EDGE_DETECTOR_FPS 0
#endif

#ifndef EDGE_CANNY_LOW
#define EDGE_CANNY_LOW 30
#endif

#ifndef EDGE_CANNY_HIGH
#define EDGE_CANNY_HIGH 90
#endif

#ifndef EDGE_DETECTOR_ABI_ID
#define EDGE_DETECTOR_ABI_ID 2
#endif

// ── Tunable settings (GCS) ───────────────────────────────────────────────────
float green_max_speed    = 0.5f;   // forward speed when safe [m/s]
float green_heading_rate = 0.5f;   // turn rate when avoiding [rad/s]
float edge_threshold     = 0.08f;  // edge fraction above which we consider danger

// ── State machine ─────────────────────────────────────────────────────────────
enum edge_state_t {
  EDGE_SAFE,
  EDGE_OBSTACLE,
  EDGE_TURNING,
};

static enum edge_state_t edge_state = EDGE_TURNING;  // start by searching

// ── Shared vision data (written by camera thread, read by periodic) ──────────
static volatile float edge_left   = 0.f;
static volatile float edge_center = 0.f;
static volatile float edge_right  = 0.f;
static volatile float edge_total  = 0.f;

static int16_t  safe_frame_cnt  = 0;
static float    turn_direction  = 1.f;
static const int16_t frames_confirm_safe = 5;

// ── Vision callback — runs in camera thread ───────────────────────────────────
static struct image_t *edge_cv_func(struct image_t *img,
                                    uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) {
    return NULL;
  }

  struct edge_result res = detect_edges(
    (char *)img->buf, img->w, img->h,
    EDGE_CANNY_LOW, EDGE_CANNY_HIGH
  );

  edge_left   = res.left_frac;
  edge_center = res.center_frac;
  edge_right  = res.right_frac;
  edge_total  = res.total_frac;

  return NULL;
}

// ── Module init ───────────────────────────────────────────────────────────────
void init_func(void)
{
  srand(time(NULL));
  turn_direction = (rand() % 2 == 0) ? 1.f : -1.f;

  cv_add_to_device(&EDGE_DETECTOR_CAMERA, edge_cv_func, EDGE_DETECTOR_FPS, 0);

  VERBOSE_PRINT("Edge-based avoider initialised (bottom camera).\n");
}

// ── Module periodic — avoidance state machine ─────────────────────────────────
void periodic_func(void)
{
  // Publish detection via ABI (quality = edge_total * 1000 for integer)
  int32_t quality = (int32_t)(edge_total * 1000.f);
  int16_t found   = (edge_total > edge_threshold) ? 1 : 0;

  AbiSendMsgVISUAL_DETECTION(
    EDGE_DETECTOR_ABI_ID,
    0, 0,
    0, 0,
    quality,
    found
  );

  // Only steer in GUIDED mode
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    edge_state     = EDGE_TURNING;
    safe_frame_cnt = 0;
    return;
  }

  // Determine if too many edges are visible (= not smooth floor)
  int8_t danger = (edge_total > edge_threshold);

  if (!danger) {
    safe_frame_cnt++;
  } else {
    safe_frame_cnt = 0;
  }

  VERBOSE_PRINT("state=%d  L=%.3f C=%.3f R=%.3f tot=%.3f thr=%.3f safe=%d\n",
                edge_state, edge_left, edge_center, edge_right,
                edge_total, edge_threshold, safe_frame_cnt);

  switch (edge_state) {

    case EDGE_SAFE:
      if (danger) {
        guidance_h_set_body_vel(0, 0);
        // Turn away from the side with more edges
        if (edge_left > edge_right) {
          turn_direction = 1.f;   // more edges on left → turn right
        } else if (edge_right > edge_left) {
          turn_direction = -1.f;  // more edges on right → turn left
        } else {
          turn_direction = (rand() % 2 == 0) ? 1.f : -1.f;
        }
        edge_state = EDGE_OBSTACLE;
      } else {
        // Clear floor below, fly forward
        guidance_h_set_body_vel(green_max_speed, 0);
      }
      break;

    case EDGE_OBSTACLE:
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(turn_direction * green_heading_rate);
      edge_state = EDGE_TURNING;
      break;

    case EDGE_TURNING:
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(turn_direction * green_heading_rate);
      if (safe_frame_cnt >= frames_confirm_safe) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        edge_state     = EDGE_SAFE;
        safe_frame_cnt = 0;
      }
      break;

    default:
      edge_state = EDGE_TURNING;
      break;
  }
}

// Called by the flight plan RETREAT block
void orange_avoider_guided_retreat(void)
{
  guidance_h_set_body_vel(-green_max_speed, 0);
  edge_state     = EDGE_TURNING;
  safe_frame_cnt = 0;
}

