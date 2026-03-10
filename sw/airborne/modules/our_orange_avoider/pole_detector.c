/*
 * @file "modules/our_orange_avoider/pole_detector.c"
 * Detects orange poles via YUV color thresholding on the front camera.
 * Publishes the centroid x-position and pixel count via ABI VISUAL_DETECTION.
 *
 * ABI message fields used:
 *   pixel_x  = centroid x of largest orange blob (-1 if none found)
 *   pixel_y  = centroid y of largest orange blob (-1 if none found)
 *   quality  = total orange pixel count
 *   extra    = 1 if pole detected, 0 if not
 */

#include "modules/our_orange_avoider/pole_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "generated/airframe.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <stdio.h>
#include <stdint.h>

#define POLE_DETECTOR_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[pole_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if POLE_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// ── Color thresholds in YUV (Y=luma, U=Cb, V=Cr) ────────────────────────────
// #ifndef POLE_Y_MIN
// #define POLE_Y_MIN 41
// #endif
// #ifndef POLE_Y_MAX
// #define POLE_Y_MAX 183
// #endif
// #ifndef POLE_U_MIN
// #define POLE_U_MIN 53
// #endif
// #ifndef POLE_U_MAX
// #define POLE_U_MAX 121
// #endif
// #ifndef POLE_V_MIN
// #define POLE_V_MIN 134
// #endif
// #ifndef POLE_V_MAX
// #define POLE_V_MAX 249
// #endif

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


#ifndef POLE_MIN_PIXEL_COUNT
#define POLE_MIN_PIXEL_COUNT 100
#endif

#ifndef POLE_DETECTOR_CAMERA
#define POLE_DETECTOR_CAMERA front_camera
#endif

#ifndef POLE_DETECTOR_FPS
#define POLE_DETECTOR_FPS 0
#endif

#ifndef POLE_DETECTOR_ABI_ID
#define POLE_DETECTOR_ABI_ID 1
#endif

// ── Shared result (written by vision thread, read by periodic) ────────────────
static int16_t  pole_cx    = -1;
static int16_t  pole_cy    = -1;
static int32_t  pole_count = 0;
static int16_t  pole_found = 0;

// ── Vision callback — runs in camera thread ───────────────────────────────────
// static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
// {
//   if (img->type != IMAGE_YUV422) {
//     return NULL;
//   }

//   uint8_t  *buf    = img->buf;
//   uint32_t  width  = img->w;
//   uint32_t  height = img->h;

//   int64_t sum_x = 0;
//   int64_t sum_y = 0;
//   int32_t count = 0;

//   // YUV422: U Y V Y per 2 pixels
//   for (uint32_t y = 0; y < height; y++) {
//     for (uint32_t x = 0; x < width; x += 2) {
//       uint32_t idx = (y * width + x) * 2;
//       uint8_t u    = buf[idx];
//       uint8_t lum1 = buf[idx + 1];
//       uint8_t v    = buf[idx + 2];
//       uint8_t lum2 = buf[idx + 3];

//       if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
//           u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
//           v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
//         sum_x += x;
//         sum_y += y;
//         count++;
//       }
//       if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
//           u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
//           v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
//         sum_x += (x + 1);
//         sum_y += y;
//         count++;
//       }
//     }
//   }

//   if (count >= POLE_MIN_PIXEL_COUNT) {
//     pole_cx    = (int16_t)(sum_x / count);
//     pole_cy    = (int16_t)(sum_y / count);
//     pole_count = count;
//     pole_found = 1;
//   } else {
//     pole_cx    = -1;
//     pole_cy    = -1;
//     pole_count = count;
//     pole_found = 0;
//   }

//   return NULL;
// }


// Green in UYVY format: {U, Y, V, Y}
static uint8_t green_color[4] = {0, 255, 0, 255};   // bright green box
static uint8_t red_color[4]   = {255, 76, 255, 76};  // red crosshair

static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) return NULL;

  uint8_t  *buf    = img->buf;
  uint32_t  width  = img->w;
  uint32_t  height = img->h;

  int64_t  sum_x = 0, sum_y = 0;
  int32_t  count = 0;
  uint32_t x_min = width,  x_max = 0;
  uint32_t y_min = height, y_max = 0;

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x += 2) {
      uint32_t idx = (y * width + x) * 2;
      uint8_t u    = buf[idx];
      uint8_t lum1 = buf[idx + 1];
      uint8_t v    = buf[idx + 2];
      uint8_t lum2 = buf[idx + 3];

      if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += x; sum_y += y; count++;
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
      }
      if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += (x + 1); sum_y += y; count++;
        if ((x + 1) < x_min) x_min = x + 1;
        if ((x + 1) > x_max) x_max = x + 1;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
      }
    }
  }

  if (count >= POLE_MIN_PIXEL_COUNT) {
    pole_cx    = (int16_t)(sum_x / count);
    pole_cy    = (int16_t)(sum_y / count);
    pole_count = count;
    pole_found = 1;

    // Draw bounding box
    image_draw_rectangle(img, (int)x_min, (int)x_max, (int)y_min, (int)y_max, green_color);

    // Draw crosshair at centroid
    struct point_t centroid;
    centroid.x = (uint32_t)pole_cx;
    centroid.y = (uint32_t)pole_cy;
    image_draw_crosshair(img, &centroid, red_color, 10);

  } else {
    pole_cx    = -1;
    pole_cy    = -1;
    pole_count = count;
    pole_found = 0;
  }

  return img;  // return img so Paparazzi streams the annotated frame
}


// ── Module init ───────────────────────────────────────────────────────────────
void pole_detector_init(void)
{
  cv_add_to_device(&POLE_DETECTOR_CAMERA, detect_orange, POLE_DETECTOR_FPS, 0);
  VERBOSE_PRINT("Pole detector initialised.\n");
}

// ── Module periodic — publishes ABI message ───────────────────────────────────
void pole_detector_periodic(void)
{
  AbiSendMsgVISUAL_DETECTION(
    POLE_DETECTOR_ABI_ID,
    pole_cx,
    pole_cy,
    0,
    0,
    pole_count,
    pole_found
  );

  VERBOSE_PRINT("found=%d  cx=%d  cy=%d  count=%d\n",
                pole_found, pole_cx, pole_cy, pole_count);
}