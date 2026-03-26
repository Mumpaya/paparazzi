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
#include <math.h>

#define TEST_MODULE_VERBOSE TRUE
#define PRINT(string,...) printf(stderr, "[edge_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
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
float green_heading_rate = 1.0f;   // turn rate when avoiding [rad/s]
float edge_threshold     = 50.f;   // legacy threshold (px), kept for compat

// ── Obstacle detector tunable settings (GCS) ─────────────────────────────────
float obs_score_threshold = 0.25f;
float obs_ema_alpha       = 0.3f;
float obs_w_ng            = 0.50f;
float obs_w_ed            = 0.30f;
float obs_w_ll            = 0.20f;
float obs_min_size        = 0.04f;
int   obs_danger_confirm_frames = 3;
int   obs_hsv_h_lo        = 10;   // beige/tan floor hue starts around 10 (OpenCV 0-180)
int   obs_hsv_h_hi        = 35;   // upper bound for warm yellow-tan
int   obs_hsv_s_lo        = 5;    // beige has very low saturation
int   obs_hsv_s_hi        = 80;   // cap saturation — richer colours are obstacles
int   obs_hsv_v_lo        = 120;  // floor is brightly lit
int   obs_hsv_v_hi        = 255;
// ── State machine ─────────────────────────────────────────────────────────────
enum edge_state_t {
  EDGE_SAFE,
  EDGE_OBSTACLE,
  EDGE_BACKING_UP,
  EDGE_TURNING,
  EDGE_RETREAT,
};

static enum edge_state_t edge_state = EDGE_SAFE;  // start safe, fly forward

// ── Shared vision data (written by camera thread, read by periodic) ──────────
static volatile float front_line_total = 0.f;
static volatile float front_line_left  = 0.f;
static volatile float front_line_right = 0.f;
static volatile float longest_line     = 0.f;
static volatile int   front_line_count = 0;

// New fused obstacle outputs
static volatile float obs_score      = 0.f;
static volatile float obs_centroid_x = 0.f;
static volatile float obs_size       = 0.f;
static volatile float obs_green_ratio = 0.f;
static volatile int   obs_calib_request = 0;

static int16_t  safe_frame_cnt  = 0;
static int16_t  danger_frame_cnt = 0;
static float    turn_direction  = 1.f;
static const int16_t frames_confirm_safe = 5;
static int16_t  backup_cnt      = 0;
static const int16_t backup_ticks = 5;  // 0.5s at 10 Hz

// ── Two-full-rotation retreat logic ───────────────────────────────────────────
static float    approach_heading     = 0.f;   // heading while flying forward (EDGE_SAFE)
static float    prev_turn_heading    = 0.f;   // previous psi sample during turn
static float    accumulated_rotation = 0.f;   // total |delta-psi| during EDGE_TURNING
static const float TWO_FULL_ROTATIONS = (float)(4.0 * M_PI);  // 2 × 360°
static int16_t  retreat_cnt          = 0;
static const int16_t retreat_ticks   = 10;    // 1 s at 10 Hz

// ── Build ObstacleConfig from GCS-tunable variables ──────────────────────────
static struct ObstacleConfig build_obstacle_cfg(void)
{
  struct ObstacleConfig cfg = obstacle_config_defaults();
  cfg.hsv_h_lo       = obs_hsv_h_lo;
  cfg.hsv_h_hi       = obs_hsv_h_hi;
  cfg.hsv_s_lo       = obs_hsv_s_lo;
  cfg.hsv_s_hi       = obs_hsv_s_hi;
  cfg.hsv_v_lo       = obs_hsv_v_lo;
  cfg.hsv_v_hi       = obs_hsv_v_hi;
  cfg.w_ng           = obs_w_ng;
  cfg.w_ed           = obs_w_ed;
  cfg.w_ll           = obs_w_ll;
  cfg.ema_alpha      = obs_ema_alpha;
  cfg.score_threshold = obs_score_threshold;
  return cfg;
}

// ── Vision callback — runs in camera thread ───────────────────────────────────
static struct image_t *edge_cv_func(struct image_t *img,
                                    uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) {
    return NULL;
  }

  if (obs_calib_request) {
    obstacle_start_ground_calibration();
    obs_calib_request = 0;
    VERBOSE_PRINT("Ground calibration requested. Sampling floor ROI...\n");
  }

  struct ObstacleConfig cfg = build_obstacle_cfg();

  struct obstacle_result res = detect_obstacles(
    (char *)img->buf, img->w, img->h,
    EDGE_CANNY_LOW, EDGE_CANNY_HIGH, cfg
  );

  /* Legacy-compatible shared variables */
  front_line_total = res.front_line_total;
  front_line_left  = res.front_left_line_length;
  front_line_right = res.front_right_line_length;
  longest_line     = res.longest_line;
  front_line_count = res.line_count;

  /* New fused outputs */
  obs_score       = res.obstacle_score;
  obs_centroid_x  = res.obstacle_centroid_x;
  obs_size        = res.obstacle_size;
  obs_green_ratio = res.green_ratio;

  {
    int h_lo, h_hi, s_lo, s_hi, v_lo, v_hi;
    if (obstacle_consume_calibrated_hsv(&h_lo, &h_hi, &s_lo, &s_hi, &v_lo, &v_hi)) {
      obs_hsv_h_lo = h_lo;
      obs_hsv_h_hi = h_hi;
      obs_hsv_s_lo = s_lo;
      obs_hsv_s_hi = s_hi;
      obs_hsv_v_lo = v_lo;
      obs_hsv_v_hi = v_hi;
      VERBOSE_PRINT("Applied floor HSV calibration: H[%d..%d] S[%d..%d] V[%d..%d]\n",
                    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi);
    }
  }

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

// ── Helper: normalise angle to (-π, π] ───────────────────────────────────────
static float wrap_pi(float a)
{
  while (a >  (float)M_PI) a -= (float)(2.0 * M_PI);
  while (a < -(float)M_PI) a += (float)(2.0 * M_PI);
  return a;
}

// ── Module periodic — avoidance state machine ─────────────────────────────────
void periodic_func(void)
{
  // Publish detection via ABI (backwards-compatible: quality = score*1000, extra = found)
  int32_t quality = (int32_t)(obs_score * 1000.f);
  int16_t found   = (obs_score > obs_score_threshold) ? 1 : 0;

  // Centroid mapped to pixel coords for ABI pixel_x field
  int16_t cx_px = (int16_t)(((obs_centroid_x + 1.f) * 0.5f) * 320.f);

  AbiSendMsgVISUAL_DETECTION(
    EDGE_DETECTOR_ABI_ID,
    cx_px, 0,
    0, 0,
    quality,
    found
  );

  // Only steer in GUIDED mode
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    edge_state     = EDGE_SAFE;
    safe_frame_cnt = 0;
    danger_frame_cnt = 0;
    return;
  }

  // Danger only if score and non-green area both persist.
  int8_t danger_raw = (obs_score > obs_score_threshold) &&
                      (obs_size > obs_min_size);
  int16_t confirm_frames = (obs_danger_confirm_frames < 1) ? 1 : obs_danger_confirm_frames;
  if (danger_raw) {
    if (danger_frame_cnt < 1000) {
      danger_frame_cnt++;
    }
  } else {
    danger_frame_cnt = 0;
  }
  int8_t danger = (danger_frame_cnt >= confirm_frames);

  if (!danger_raw) {
    safe_frame_cnt++;
  } else {
    safe_frame_cnt = 0;
  }

  VERBOSE_PRINT("state=%d score=%.2f size=%.2f cx=%.2f green=%.2f danger=%d/%d safe=%d\n",
                edge_state, (double)obs_score, (double)obs_size, (double)obs_centroid_x,
                (double)obs_green_ratio, danger_frame_cnt, confirm_frames,
                safe_frame_cnt);

  switch (edge_state) {

    case EDGE_SAFE:
      if (danger) {
        // Stop forward motion immediately
        guidance_h_set_body_vel(0, 0);
        // Turn away from obstacle centroid: centroid > 0 → obstacle on right → turn left
        if (obs_centroid_x > 0.05f) {
          turn_direction = -1.f;
        } else if (obs_centroid_x < -0.05f) {
          turn_direction = 1.f;
        } else {
          turn_direction = (rand() % 2 == 0) ? 1.f : -1.f;
        }
        edge_state = EDGE_OBSTACLE;
      } else {
        // Safe — fly forward; remember heading for possible retreat
        approach_heading = stateGetNedToBodyEulers_f()->psi;
        guidance_h_set_body_vel(green_max_speed, 0);
      }
      break;

    case EDGE_OBSTACLE:
      // Decide turn direction, then start backing up
      guidance_h_set_body_vel(-green_max_speed, 0);
      guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);  // lock heading while reversing
      backup_cnt = 0;
      edge_state = EDGE_BACKING_UP;
      break;

    case EDGE_BACKING_UP:
      // Pure reverse, no turning — get away from the line first
      // Don't check danger here; complete the full backup phase
      guidance_h_set_body_vel(-green_max_speed, 0);
      backup_cnt++;
      if (backup_cnt >= backup_ticks) {
        // Done backing up, reset safe counter so turning waits for fresh readings
        safe_frame_cnt = 0;
        prev_turn_heading    = stateGetNedToBodyEulers_f()->psi;
        accumulated_rotation = 0.f;
        edge_state = EDGE_TURNING;
      }
      break;

    case EDGE_TURNING: {
      // Stop moving, just turn in place until front is clear
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(turn_direction * green_heading_rate);

      // Track cumulative rotation
      float cur_psi = stateGetNedToBodyEulers_f()->psi;
      float delta   = wrap_pi(cur_psi - prev_turn_heading);
      accumulated_rotation += fabsf(delta);
      prev_turn_heading = cur_psi;

      if (accumulated_rotation >= TWO_FULL_ROTATIONS) {
        // Two full rotations without finding a clear path — retreat
        float retreat_heading = wrap_pi(approach_heading + (float)M_PI);
        guidance_h_set_heading_rate(0);  // stop yaw
        guidance_h_set_heading(retreat_heading);
        retreat_cnt = 0;
        edge_state  = EDGE_RETREAT;
        VERBOSE_PRINT("2 full rotations — retreating on heading %.2f\n",
                      (double)retreat_heading);
      } else if (safe_frame_cnt >= frames_confirm_safe) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        edge_state     = EDGE_SAFE;
        safe_frame_cnt = 0;
        danger_frame_cnt = 0;
      }
      break;
    }

    case EDGE_RETREAT:
      // Fly forward along the retreat heading (opposite of approach)
      guidance_h_set_body_vel(green_max_speed, 0);
      retreat_cnt++;
      if (retreat_cnt >= retreat_ticks) {
        edge_state     = EDGE_SAFE;
        safe_frame_cnt = 0;
        danger_frame_cnt = 0;
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

void ground_calibration_start(void)
{
  obs_calib_request = 1;
}

