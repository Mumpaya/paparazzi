#include "modules/our_orange_avoider/pole_detector_2.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "generated/airframe.h"
#include <iostream>
#include <cstdint>

// -- Default Thresholds (YUV) --
#ifndef POLE_Y_MIN
#define POLE_Y_MIN 41
#define POLE_Y_MAX 183
#define POLE_U_MIN 53
#define POLE_U_MAX 121
#define POLE_V_MIN 134
#define POLE_V_MAX 249
#endif

#ifndef POLE_MIN_PIXEL_COUNT
#define POLE_MIN_PIXEL_COUNT 100
#endif

#ifndef POLE_DETECTOR_ABI_ID
#define POLE_DETECTOR_ABI_ID 1
#endif

// Internal state
static int16_t pole_cx = -1;
static int16_t pole_cy = -1;
static int32_t pole_count = 0;
static int16_t pole_found = 0;

// Vision callback - Processes YUV422 directly (U Y1 V Y2)
static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) return nullptr;

  uint8_t *buf = (uint8_t *)img->buf;
  uint32_t width = img->w;
  uint32_t height = img->h;

  int64_t sum_x = 0;
  int64_t sum_y = 0;
  int32_t count = 0;

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x += 2) {
      uint32_t idx = (y * width + x) * 2;
      
      uint8_t u    = buf[idx];
      uint8_t lum1 = buf[idx + 1];
      uint8_t v    = buf[idx + 2];
      uint8_t lum2 = buf[idx + 3];

      // Check first pixel (Y1, U, V)
      if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
          u >= POLE_U_MIN && u <= POLE_U_MAX &&
          v >= POLE_V_MIN && v <= POLE_V_MAX) {
        sum_x += x;
        sum_y += y;
        count++;
      }
      // Check second pixel (Y2, U, V)
      if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
          u >= POLE_U_MIN && u <= POLE_U_MAX &&
          v >= POLE_V_MIN && v <= POLE_V_MAX) {
        sum_x += (x + 1);
        sum_y += y;
        count++;
      }
    }
  }

  if (count >= POLE_MIN_PIXEL_COUNT) {
    pole_cx = static_cast<int16_t>(sum_x / count);
    pole_cy = static_cast<int16_t>(sum_y / count);
    pole_found = 1;
  } else {
    pole_cx = -1;
    pole_cy = -1;
    pole_found = 0;
  }
  pole_count = count;

  return nullptr;
}

extern "C" {

void pole_detector_2_init(void) {
  // Add to front_camera (or whatever is defined in XML)
  cv_add_to_device(&front_camera, detect_orange, 0, 0);
}

void pole_detector_2_periodic(void) {
  // Push results to the ABI bus so the avoider module can hear it
  AbiSendMsgVISUAL_DETECTION(POLE_DETECTOR_ABI_ID, pole_cx, pole_cy, 0, 0, pole_count, pole_found);
}

} // extern "C"