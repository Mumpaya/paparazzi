/*
 * @file "modules/our_orange_avoider/avoider.c"
 */

#include "modules/our_orange_avoider/avoider.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "modules/core/abi.h"
#include "state.h"
#include "generated/airframe.h"
#include <stdio.h>
#include <time.h>

#define AVOIDER_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// ── Settings (tunable from GCS) ───────────────────────────────────────────────
float max_speed        = 0.3f;   // forward speed [m/s]
float heading_rate     = 0.4f;   // turning rate [rad/s]
float center_threshold = 0.20f;  // fraction of image width defining "center zone"
float danger_threshold = 8000.f; // pixel count above which a pole is "dangerous"

// ── ABI ───────────────────────────────────────────────────────────────────────
#ifndef POLE_DETECTOR_ABI_ID
#define POLE_DETECTOR_ABI_ID 1
#endif

// ── State machine ─────────────────────────────────────────────────────────────
enum avoider_state_t {
  FORWARD,
  AVOID_RIGHT,
  AVOID_LEFT,
  STOP_AND_TURN,
  SEARCH,
};

static enum avoider_state_t avoider_state = SEARCH;

// ── ABI data ──────────────────────────────────────────────────────────────────
static int16_t  last_cx     = -1;
static int16_t  last_cy     = -1;
static int32_t  last_count  = 0;
static int16_t  last_found  = 0;
static int16_t  safe_frames = 0;
static float    turn_dir    = 1.f;

const int16_t frames_to_confirm_safe = 5;

static abi_event pole_ev;
static void pole_cb(uint8_t __attribute__((unused)) sender_id,
                    int16_t pixel_x, int16_t pixel_y,
                    int16_t __attribute__((unused)) pixel_width,
                    int16_t __attribute__((unused)) pixel_height,
                    int32_t quality, int16_t extra)
{
  last_cx    = pixel_x;
  last_cy    = pixel_y;
  last_count = quality;
  last_found = extra;
}

void avoider_init(void)
{
  srand(time(NULL));
  AbiBindMsgVISUAL_DETECTION(POLE_DETECTOR_ABI_ID, &pole_ev, pole_cb);
  VERBOSE_PRINT("Avoider initialised.\n");
}

void avoider_periodic(void)
{
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    avoider_state = SEARCH;
    safe_frames   = 0;
    return;
  }

  float img_w = (float)front_camera.output_size.w;

  // A pole is only "dangerous" if it's detected AND large enough (close enough)
  int8_t dangerous = last_found && (last_count > (int32_t)danger_threshold);

  // Normalised centroid (only meaningful when dangerous)
  float cx_norm = (dangerous && img_w > 0) ? (last_cx / img_w) : 0.5f;

  // Safe frame counter: increment when no dangerous pole seen
  if (!dangerous) {
    safe_frames++;
  } else {
    safe_frames = 0;
  }

  VERBOSE_PRINT("state=%d found=%d count=%d dangerous=%d cx=%.2f safe=%d\n",
                avoider_state, last_found, last_count, dangerous, cx_norm, safe_frames);

  switch (avoider_state) {

    case SEARCH:
      // Rotate until we've seen enough non-dangerous frames
      guidance_h_set_heading_rate(heading_rate);
      if (safe_frames >= frames_to_confirm_safe) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        avoider_state = FORWARD;
        safe_frames   = 0;
      }
      break;

    case FORWARD:
      if (dangerous) {
        safe_frames = 0;
        if (cx_norm < (0.5f - center_threshold)) {
          turn_dir      = 1.f;   // pole on LEFT → turn right
          avoider_state = AVOID_RIGHT;
        } else if (cx_norm > (0.5f + center_threshold)) {
          turn_dir      = -1.f;  // pole on RIGHT → turn left
          avoider_state = AVOID_LEFT;
        } else {
          turn_dir      = (rand() % 2 == 0) ? 1.f : -1.f;
          avoider_state = STOP_AND_TURN;
        }
      } else {
        guidance_h_set_body_vel(max_speed, 0);
      }
      break;

    case AVOID_RIGHT:
    case AVOID_LEFT:
    case STOP_AND_TURN:
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(turn_dir * heading_rate);
      if (safe_frames >= frames_to_confirm_safe) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        avoider_state = FORWARD;
        safe_frames   = 0;
      }
      break;

    default:
      avoider_state = SEARCH;
      break;
  }
}

void orange_avoider_guided_retreat(void)
{
  guidance_h_set_body_vel(-max_speed, 0);
  avoider_state = SEARCH;
  safe_frames   = 0;
}
