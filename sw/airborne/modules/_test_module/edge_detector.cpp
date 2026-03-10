/*
 * @file "modules/_test_module/edge_detector.cpp"
 * OpenCV Canny edge detection for the bottom camera.
 * Splits the image into left / center / right thirds and returns
 * the fraction of edge pixels in each region.
 */

#include "edge_detector.h"

using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include "opencv_image_functions.h"

struct edge_result detect_edges(char *img, int width, int height,
                                int canny_low, int canny_high)
{
  struct edge_result res = {0.f, 0.f, 0.f, 0.f};

  // Wrap the YUV422 buffer in an OpenCV Mat (2 bytes per pixel, interleaved)
  Mat yuv(height, width, CV_8UC2, img);

  // Convert to grayscale
  Mat gray;
  cvtColor(yuv, gray, cv::COLOR_YUV2GRAY_Y422);

  // Gaussian blur to suppress noise
  GaussianBlur(gray, gray, Size(5, 5), 1.5);

  // Canny edge detection
  Mat edges;
  Canny(gray, edges, canny_low, canny_high);

  // Split into three vertical thirds: left, center, right
  int third = width / 3;
  int left_end   = third;
  int right_start = width - third;

  int left_count   = 0;
  int center_count = 0;
  int right_count  = 0;
  int total_count  = 0;

  for (int y = 0; y < height; y++) {
    const uchar *row = edges.ptr<uchar>(y);
    for (int x = 0; x < width; x++) {
      if (row[x] > 0) {
        total_count++;
        if (x < left_end) {
          left_count++;
        } else if (x >= right_start) {
          right_count++;
        } else {
          center_count++;
        }
      }
    }
  }

  float region_pixels = (float)(third * height);
  float total_pixels  = (float)(width * height);

  if (region_pixels > 0.f) {
    res.left_frac   = (float)left_count   / region_pixels;
    res.center_frac = (float)center_count / region_pixels;
    res.right_frac  = (float)right_count  / region_pixels;
  }
  if (total_pixels > 0.f) {
    res.total_frac = (float)total_count / total_pixels;
  }

  // Write the edge image back into the YUV buffer for RTP visualization
  grayscale_opencv_to_yuv422(edges, img, width, height);

  return res;
}
