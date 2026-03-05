#include "modules/AF_8_V1/controller.h"
#include "modules/AF_8_V1/pole_detector_2.h"
extern "C" {
#include "modules/computer_vision/cv.h"
}
#include "modules/core/abi.h"
#include "generated/airframe.h"
#include <iostream>
#include <cstdint>

// ============================================================
// COLOR THRESHOLDS (YUV422)
// YUV is the raw camera format equivalent of HSV in Python.
//
// PERFECT IN SIMULATION (matches Python HSV (5,120,80)-(30,255,255)):
// #define POLE_Y_MIN  41
// #define POLE_Y_MAX 183
// #define POLE_U_MIN  53
// #define POLE_U_MAX 121
// #define POLE_V_MIN 134
// #define POLE_V_MAX 249
//
// TUNED FOR CYBERZOO (matches Python HSV (0,40,50)-(25,255,255)):
#define POLE_Y_MIN  30
#define POLE_Y_MAX 220
#define POLE_U_MIN  60
#define POLE_U_MAX 130
#define POLE_V_MIN 125
#define POLE_V_MAX 255
// ============================================================

// ============================================================
// BLOB FILTER PARAMS
// Match Python: min_area=500, min_aspect_ratio=2.5
#define POLE_MIN_PIXEL_COUNT  500   // min orange pixels in bounding box
#define POLE_MIN_ASPECT_RATIO 2.5f  // height/width must be >= this (tall thin pole)
// ============================================================

#ifndef POLE_DETECTOR_ABI_ID
#define POLE_DETECTOR_ABI_ID 1
#endif

// Internal state
static int16_t pole_cx    = -1;
static int16_t pole_cy    = -1;
static int32_t pole_count = 0;
static int16_t pole_found = 0;

// Vision callback - processes YUV422 directly (U Y1 V Y2 layout)
static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) return nullptr;

  uint8_t  *buf    = (uint8_t *)img->buf;
  uint32_t  width  = img->w;
  uint32_t  height = img->h;

  // Bounding box tracking + centroid
  int64_t  sum_x  = 0;
  int64_t  sum_y  = 0;
  int32_t  count  = 0;
  uint32_t x_min  = width;
  uint32_t x_max  = 0;
  uint32_t y_min  = height;
  uint32_t y_max  = 0;

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x += 2) {
      uint32_t idx = (y * width + x) * 2;

      uint8_t u    = buf[idx];
      uint8_t lum1 = buf[idx + 1];
      uint8_t v    = buf[idx + 2];
      uint8_t lum2 = buf[idx + 3];

      // First pixel (Y1, U, V)
      if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += x; sum_y += y; count++;
        if (x   < x_min) x_min = x;
        if (x   > x_max) x_max = x;
        if (y   < y_min) y_min = y;
        if (y   > y_max) y_max = y;
      }
      // Second pixel (Y2, U, V)
      if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += (x + 1); sum_y += y; count++;
        if ((x+1) < x_min) x_min = x+1;
        if ((x+1) > x_max) x_max = x+1;
        if (y     < y_min) y_min = y;
        if (y     > y_max) y_max = y;
      }
    }
  }

  // Apply filters: min pixel count + min aspect ratio (height/width)
  uint32_t blob_w = (x_max > x_min) ? (x_max - x_min + 1) : 1;
  uint32_t blob_h = (y_max > y_min) ? (y_max - y_min + 1) : 1;
  float    aspect = (float)blob_h / (float)blob_w;

  if (count >= POLE_MIN_PIXEL_COUNT && aspect >= POLE_MIN_ASPECT_RATIO) {
    pole_cx    = static_cast<int16_t>(sum_x / count);
    pole_cy    = static_cast<int16_t>(sum_y / count);
    pole_found = 1;
  } else {
    pole_cx    = -1;
    pole_cy    = -1;
    pole_found = 0;
  }
  pole_count = count;

  return nullptr;
}

extern "C" {

void pole_detector_2_init(void) {
  cv_add_to_device(&front_camera, detect_orange, 0, 0);
}

void pole_detector_2_periodic(void) {
  AbiSendMsgVISUAL_DETECTION(POLE_DETECTOR_ABI_ID, pole_cx, pole_cy, 0, 0, pole_count, pole_found);
}

} // extern "C"