/*
 * Copyright (C) 2026
 *
 * This file is part of Paparazzi.
 */

#include "modules/computer_vision/cv_bottom_edge_detector.h"
#include "modules/computer_vision/cv.h"
#include "std.h"

#include <string.h>
#include <pthread.h>

#ifndef CV_BOTTOM_EDGE_DETECTOR_FPS
#define CV_BOTTOM_EDGE_DETECTOR_FPS 0
#endif

#ifndef CV_BED_H_MIN
#define CV_BED_H_MIN 0
#endif
#ifndef CV_BED_H_MAX
#define CV_BED_H_MAX 66
#endif
#ifndef CV_BED_S_MIN
#define CV_BED_S_MIN 15
#endif
#ifndef CV_BED_S_MAX
#define CV_BED_S_MAX 225
#endif
#ifndef CV_BED_V_MIN
#define CV_BED_V_MIN 120
#endif
#ifndef CV_BED_V_MAX
#define CV_BED_V_MAX 255
#endif
#ifndef CV_BED_NOTGREEN_H_MIN
#define CV_BED_NOTGREEN_H_MIN 88
#endif
#ifndef CV_BED_NOTGREEN_H_MAX
#define CV_BED_NOTGREEN_H_MAX 255
#endif
#ifndef CV_BED_NOTGREEN_S_MIN
#define CV_BED_NOTGREEN_S_MIN 0
#endif
#ifndef CV_BED_NOTGREEN_S_MAX
#define CV_BED_NOTGREEN_S_MAX 255
#endif
#ifndef CV_BED_NOTGREEN_V_MIN
#define CV_BED_NOTGREEN_V_MIN 0
#endif
#ifndef CV_BED_NOTGREEN_V_MAX
#define CV_BED_NOTGREEN_V_MAX 255
#endif
#ifndef CV_BED_NONUNIFORM_THRESH
#define CV_BED_NONUNIFORM_THRESH 15
#endif
#ifndef CV_BED_EDGE_RADIUS_RATIO_PCT
#define CV_BED_EDGE_RADIUS_RATIO_PCT 80
#endif
#ifndef CV_BED_DRAW
#define CV_BED_DRAW false
#endif

struct cv_bottom_edge_detector_state_t cv_bottom_edge_detector_state;

uint8_t cbed_h_min = CV_BED_H_MIN;
uint8_t cbed_h_max = CV_BED_H_MAX;
uint8_t cbed_s_min = CV_BED_S_MIN;
uint8_t cbed_s_max = CV_BED_S_MAX;
uint8_t cbed_v_min = CV_BED_V_MIN;
uint8_t cbed_v_max = CV_BED_V_MAX;
uint8_t cbed_notgreen_h_min = CV_BED_NOTGREEN_H_MIN;
uint8_t cbed_notgreen_h_max = CV_BED_NOTGREEN_H_MAX;
uint8_t cbed_notgreen_s_min = CV_BED_NOTGREEN_S_MIN;
uint8_t cbed_notgreen_s_max = CV_BED_NOTGREEN_S_MAX;
uint8_t cbed_notgreen_v_min = CV_BED_NOTGREEN_V_MIN;
uint8_t cbed_notgreen_v_max = CV_BED_NOTGREEN_V_MAX;
uint8_t cbed_nonuniform_thresh = CV_BED_NONUNIFORM_THRESH;
uint8_t cbed_edge_radius_ratio_pct = CV_BED_EDGE_RADIUS_RATIO_PCT;
bool cbed_draw = CV_BED_DRAW;

static pthread_mutex_t cbed_mutex;

static inline int clamp(int v, int lo, int hi)
{
  if (v < lo) {
    return lo;
  } else if (v > hi) {
    return hi;
  } else {
    return v;
  }
}
static inline void yuv_to_rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t *r, uint8_t *g, uint8_t *b)
{
  *r = (uint8_t)clamp((int)(y + 1.402f * ((int)v - 128)), 0, 255);
  *g = (uint8_t)clamp((int)(y - 0.344136f * ((int)u - 128) - 0.714136f * ((int)v - 128)), 0, 255);
  *b = (uint8_t)clamp((int)(y + 1.772f * ((int)u - 128)), 0, 255);
}
static inline void rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b, uint8_t *h, uint8_t *s, uint8_t *v)
{
  uint8_t hsv[3] = {0, 0, 0};
  float rf = (float)r / 255.0f;
  float gf = (float)g / 255.0f;
  float bf = (float)b / 255.0f;
  float max = fmaxf(rf, fmaxf(gf, bf));
  float min = fminf(rf, fminf(gf, bf));
  float delta = max - min;
  float hue;
  if (max == min) {
    hue = 0.0f;
  } else {
    if (max == rf) {
      hue = fmod(60.0f * ((gf - bf) / delta) + 360.0f, 360.0f);
    } else if (max == gf) {
      hue = fmod(60.0f * ((bf - rf) / delta) + 120.0f, 360.0f);
    } else {
      hue = fmod(60.0f * ((rf - gf) / delta) + 240.0f, 360.0f);
    }
  }
  hsv[0] = (uint8_t)(hue / 360.0f * 255.0f);
  if (max == 0.0f) {
    hsv[1] = 0;
  } else {
    hsv[1] = (uint8_t)(delta / max * 255.0f);
  }
  hsv[2] = (uint8_t)(max * 255.0f);
  *h = hsv[0];
  *s = hsv[1];
  *v = hsv[2];
}

static inline bool in_range(uint8_t v, uint8_t lo, uint8_t hi)
{
  return (v >= lo && v <= hi);
}

static struct image_t *cv_bottom_edge_detector_cb(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img == NULL || img->buf == NULL || img->w < 4 || img->h < 4) {
    return img;
  }

  uint8_t *buf = (uint8_t *)img->buf;
  uint32_t good_cnt = 0;
  uint32_t sum_x = 0;
  uint32_t sum_y = 0;
  uint16_t stride = img->w * 2;

  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {
      uint8_t *yp;
      uint8_t *up;
      uint8_t *vp;

      if ((x & 1) == 0) {
        up = &buf[y * stride + 2 * x];
        yp = &buf[y * stride + 2 * x + 1];
        vp = &buf[y * stride + 2 * x + 2];
      } else {
        up = &buf[y * stride + 2 * x - 2];
        vp = &buf[y * stride + 2 * x];
        yp = &buf[y * stride + 2 * x + 1];
      }

      uint8_t r, g, b;
      yuv_to_rgb(*yp, *up, *vp, &r, &g, &b);
      uint8_t h, s, v;
      rgb_to_hsv(r, g, b, &h, &s, &v);

      bool green = in_range(h, cbed_h_min, cbed_h_max) &&
                   in_range(s, cbed_s_min, cbed_s_max) &&
                   in_range(v, cbed_v_min, cbed_v_max);

      bool not_green = in_range(h, cbed_notgreen_h_min, cbed_notgreen_h_max) &&
                       in_range(s, cbed_notgreen_s_min, cbed_notgreen_s_max) &&
                       in_range(v, cbed_notgreen_v_min, cbed_notgreen_v_max);

      uint8_t local_diff = 0;
      if (x > 0) {
        uint8_t left_y = buf[y * stride + 2 * (x - 1) + 1];
        local_diff = (uint8_t)abs((int)(*yp) - (int)left_y);
      }
      bool non_uniform = (local_diff >= cbed_nonuniform_thresh);

      bool bad = not_green && !non_uniform;
      //bool good = (green || non_uniform) && !bad;
      bool good = not_green;

      if (good) {
        good_cnt++;
        sum_x += x;
        sum_y += y;
        if (cbed_draw) {
          *yp = 255;
        }
      }
    }
  }

  float goodness = (float)good_cnt / (float)(img->w * img->h);
  int16_t x_c = 0;
  int16_t y_c = 0;
  bool valid = good_cnt > 0;
  bool over_edge = false;

  if (valid) {
    float cx = (float)sum_x / (float)good_cnt;
    float cy = (float)sum_y / (float)good_cnt;
    x_c = (int16_t)roundf(cx - (float)img->w * 0.5f);
    y_c = (int16_t)roundf((float)img->h * 0.5f - cy);

    float mag2 = (float)x_c * (float)x_c + (float)y_c * (float)y_c;
    float edge_r = ((float)img->w * 0.25f) * ((float)cbed_edge_radius_ratio_pct / 100.0f);
    over_edge = (mag2 > edge_r * edge_r);
  }

  pthread_mutex_lock(&cbed_mutex);
  cv_bottom_edge_detector_state.goodness = goodness;
  cv_bottom_edge_detector_state.x_c = x_c;
  cv_bottom_edge_detector_state.y_c = y_c;
  cv_bottom_edge_detector_state.over_edge = over_edge;
  cv_bottom_edge_detector_state.valid = valid;
  cv_bottom_edge_detector_state.updated = true;
  pthread_mutex_unlock(&cbed_mutex);

  return img;
}

void cv_bottom_edge_detector_init(void)
{
  memset(&cv_bottom_edge_detector_state, 0, sizeof(cv_bottom_edge_detector_state));
  pthread_mutex_init(&cbed_mutex, NULL);

#ifdef CV_BOTTOM_EDGE_DETECTOR_CAMERA
  cv_add_to_device(&CV_BOTTOM_EDGE_DETECTOR_CAMERA, cv_bottom_edge_detector_cb, CV_BOTTOM_EDGE_DETECTOR_FPS, 0);
#endif
}

void cv_bottom_edge_detector_periodic(void)
{
  pthread_mutex_lock(&cbed_mutex);
  cv_bottom_edge_detector_state.updated = false;
  pthread_mutex_unlock(&cbed_mutex);
}
