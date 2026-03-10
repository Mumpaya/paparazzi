/*
 * @file "modules/_test_module/edge_detector.h"
 * OpenCV Canny edge detection on the bottom camera.
 * Returns the fraction of edge pixels in left, center, and right thirds.
 */

#ifndef EDGE_DETECTOR_H
#define EDGE_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct edge_result {
  float left_frac;    // edge pixel fraction in left third
  float center_frac;  // edge pixel fraction in center third
  float right_frac;   // edge pixel fraction in right third
  float total_frac;   // edge pixel fraction in entire image
};

/*
 * Run Canny edge detection on a YUV422 image buffer.
 * Writes the edge image back into the buffer for RTP visualization.
 */
struct edge_result detect_edges(char *img, int width, int height,
                                int canny_low, int canny_high);

#ifdef __cplusplus
}
#endif

#endif
