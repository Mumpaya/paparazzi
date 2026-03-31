/*
 * Copyright (C) 2026
 *
 * This file is part of Paparazzi.
 */

extern "C" {
#include "modules/computer_vision/cv_bottom_flow_velocity.h"
#include "modules/computer_vision/cv.h"
#include "std.h"
#include "mcu_periph/sys_time.h"
}

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>

#include <math.h>
#include <string.h>
#include <pthread.h>
#include <vector>

#ifndef CV_BOTTOM_FLOW_VELOCITY_FPS
#define CV_BOTTOM_FLOW_VELOCITY_FPS 0
#endif

#ifndef CV_BFV_ALTITUDE_M
#define CV_BFV_ALTITUDE_M 1.25f
#endif
#ifndef CV_BFV_FOV_DEG
#define CV_BFV_FOV_DEG 42.0f
#endif
#ifndef CV_BFV_IMG_WIDTH_REF
#define CV_BFV_IMG_WIDTH_REF 240
#endif
#ifndef CV_BFV_MIN_POINTS
#define CV_BFV_MIN_POINTS 8
#endif
#ifndef CV_BFV_MAX_FLOW_PX
#define CV_BFV_MAX_FLOW_PX 20.0f
#endif
#ifndef CV_BFV_DRAW
#define CV_BFV_DRAW false
#endif

struct cv_bottom_flow_velocity_state_t cv_bottom_flow_velocity_state;
float cv_bfv_altitude_m = CV_BFV_ALTITUDE_M;
float cv_bfv_fov_deg = CV_BFV_FOV_DEG;
uint16_t cv_bfv_img_width_ref = CV_BFV_IMG_WIDTH_REF;
uint8_t cv_bfv_min_points = CV_BFV_MIN_POINTS;
float cv_bfv_max_flow_px = CV_BFV_MAX_FLOW_PX;
bool cv_bfv_draw = CV_BFV_DRAW;
int cv_bfv_max_points = 50;

static pthread_mutex_t bfv_mutex;
static bool bfv_has_prev = false;
static uint32_t bfv_prev_t = 0;
static cv::Mat bfv_prev_base_im;
static std::vector<cv::Point2f> bfv_prev_pts;
static cv::Ptr<cv::FastFeatureDetector> bfv_fast_feature_detector =
  cv::FastFeatureDetector::create(15, true, cv::FastFeatureDetector::TYPE_7_12);

static inline float bfv_px_to_m(float px, float alt_m, float fov_deg, uint16_t width_px)
{
  if (width_px == 0) {
    return 0.0f;
  }
  float width_m = 2.0f * alt_m * tanf((fov_deg * (float)M_PI / 180.0f) * 0.5f);
  return px * (width_m / (float)width_px);
}

static inline void bfv_detect_features(const cv::Mat &base_im, std::vector<cv::Point2f> &pts)
{
  //const int max_corners = 20;
  //const double quality_level = 0.01;
  //const double min_distance = 7.0;
  //const int block_size = 7;
  //cv::goodFeaturesToTrack(base_im, pts, max_corners, quality_level, min_distance, cv::Mat(), block_size);
  pts.clear();

  if (!bfv_fast_feature_detector.empty()) {
    std::vector<cv::KeyPoint> keypoints;
    bfv_fast_feature_detector->detect(base_im, keypoints);
    const int take_every = keypoints.size() / cv_bfv_max_points + 1;
    pts.reserve(cv_bfv_max_points);
    for (size_t i = 0; i < keypoints.size(); i += take_every) {
      pts.push_back(keypoints[i].pt);
    }
  }
  printf("number of features detected: %zu\n", pts.size());
}

static struct image_t *cv_bottom_flow_velocity_cb(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img == NULL || img->buf == NULL || img->w < 4 || img->h < 4) {
    return img;
  }


  uint8_t *buf = (uint8_t *)img->buf;
  uint32_t t_now = get_sys_time_usec();
  const uint16_t w = img->w;
  const uint16_t h = img->h;
  const uint32_t stride = (uint32_t)w * 2U;

  static cv::Mat base_im;
  static bool mats_init = false;
  if (!mats_init) {
      base_im = cv::Mat((int)h, (int)w, CV_8UC1);
      mats_init = true;
  }
  // Fast Y extraction with row pointers.
  for (uint16_t y =0; y < h; y++) {
    uint8_t *dst = base_im.ptr<uint8_t>((int)y);
    const uint8_t *src = &buf[(uint32_t)y * stride +1U];
    for (uint16_t x =0; x < w; x++) {
      dst[x] = src[(uint32_t)x *2U];
    }
  }
  printf("bfv copy time: %.3f ms\n",
         (double)(get_sys_time_usec() - t_now) / 1000.0
    );

  //for (uint16_t y = 0; y < h; y++) {
  //  for (uint16_t x = 0; x < w; x++) {
  //    uint32_t idx = (uint32_t)y * stride + (uint32_t)2U * x + 1U;
  //    base_im.at<uint8_t>((int)y, (int)x) = buf[idx];
  //  }
  //}
  //cv::convertScaleAbs(base_im, base_im, 1.3, 0.0);

  std::vector<cv::Point2f> points;
  std::vector<cv::Point2f> flows;
  cv::Point2f avg_flow(0.0f, 0.0f);

  if (!bfv_has_prev) {
    bfv_detect_features(base_im, bfv_prev_pts);
    bfv_prev_base_im = base_im.clone();
    bfv_prev_t = t_now;
    bfv_has_prev = true;
    return img;
  }

  if (!bfv_prev_pts.empty()) {
    std::vector<cv::Point2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;

    uint32_t t_of_start = get_sys_time_usec();
    cv::calcOpticalFlowPyrLK(
      bfv_prev_base_im,
      base_im,
      bfv_prev_pts,
      next_pts,
      status,
      err,
      cv::Size(9, 9),
      2,
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03)
    );
      printf("bfv optical flow time: %.3f ms\n",
          (double)(get_sys_time_usec() - t_of_start) / 1000.0);

    if (!next_pts.empty() && !status.empty()) {
      cv::Point2f sum_flow(0.0f, 0.0f);
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
          cv::Point2f f = next_pts[i] - bfv_prev_pts[i];
          points.push_back(bfv_prev_pts[i]);
          flows.push_back(f);
          sum_flow += f;
        }
      }
      if (!flows.empty()) {
        avg_flow = sum_flow * (1.0f / (float)flows.size());
      }
    }
  }

  uint32_t t_detect_start = get_sys_time_usec();
  bfv_detect_features(base_im, bfv_prev_pts);
  printf("bfv detect features time: %.3f ms\n",
         (double)(get_sys_time_usec() - t_detect_start) / 1000.0);
  bfv_prev_base_im = base_im.clone();

  uint16_t tracked_points = (uint16_t)flows.size();
  bool valid = false;
  float vx = 0.0f;
  float vy = 0.0f;

  if (tracked_points >= cv_bfv_min_points && t_now > bfv_prev_t) {
    float mag = sqrtf(avg_flow.x * avg_flow.x + avg_flow.y * avg_flow.y);
    if (mag <= cv_bfv_max_flow_px) {
      float dt = (float)(t_now - bfv_prev_t) * 1e-6f;
      uint16_t width_ref = cv_bfv_img_width_ref ? cv_bfv_img_width_ref : w;
      float dx_m = bfv_px_to_m(avg_flow.x, cv_bfv_altitude_m, cv_bfv_fov_deg, width_ref);
      float dy_m = bfv_px_to_m(avg_flow.y, cv_bfv_altitude_m, cv_bfv_fov_deg, width_ref);
      vx = dx_m / dt;
      vy = dy_m / dt;
      valid = true;
    }
  }


  uint32_t t_draw_start = get_sys_time_usec();

  if (cv_bfv_draw) {
    // Draw per-feature flow vectors.
    for (size_t i = 0; i < points.size() && i < flows.size(); i++) {
      cv::Point start((int)lrintf(points[i].x), (int)lrintf(points[i].y));
      cv::Point end(
        (int)lrintf(points[i].x + flows[i].x),
        (int)lrintf(points[i].y + flows[i].y)
      );
      cv::arrowedLine(base_im, start, end, cv::Scalar(255), 1, cv::LINE_AA, 0, 0.25);
    }

    // Draw average flow vector from image center.
    cv::Point center((int)w / 2, (int)h / 2);
    cv::Point avg_end(
      (int)lrintf((float)center.x + avg_flow.x * 10.0f),
      (int)lrintf((float)center.y + avg_flow.y * 10.0f)
    );
    cv::arrowedLine(base_im, center, avg_end, cv::Scalar(255), 2, cv::LINE_AA, 0, 0.3);

	// Draw velocity text overlay.
 char text[96];
 snprintf(text, sizeof(text), "vx:%+.2f m/s vy:%+.2f m/s", (double)vx, (double)vy);
 cv::putText(
 base_im,
 text,
 cv::Point(6,18),
 cv::FONT_HERSHEY_SIMPLEX,
0.45,
 cv::Scalar(255),
1,
 cv::LINE_AA );


    // Copy grayscale overlay back into Y channel so vectors appear in output stream.
    for (uint16_t y = 0; y < h; y++) {
      for (uint16_t x = 0; x < w; x++) {
        uint32_t idx = (uint32_t)y * stride + (uint32_t)2U * (uint32_t)x + 1U;
        buf[idx] = base_im.at<uint8_t>((int)y, (int)x);
      }
    }
  }


  pthread_mutex_lock(&bfv_mutex);
  cv_bottom_flow_velocity_state.vx_mps = vx;
  cv_bottom_flow_velocity_state.vy_mps = vy;
  cv_bottom_flow_velocity_state.flow_px_x = avg_flow.x;
  cv_bottom_flow_velocity_state.flow_px_y = avg_flow.y;
  cv_bottom_flow_velocity_state.tracked_points = tracked_points;
  cv_bottom_flow_velocity_state.valid = valid;
  cv_bottom_flow_velocity_state.updated = true;
  pthread_mutex_unlock(&bfv_mutex);

  printf("bfv draw time: %.3f ms\n",
         (double)(get_sys_time_usec() - t_draw_start) / 1000.0);

  bfv_prev_t = t_now;
printf("BFV processing time: %.3f ms\n",
       (double)(get_sys_time_usec() - t_now) / 1000.0);
  return img;
}

extern "C" void cv_bottom_flow_velocity_init(void)
{
  memset(&cv_bottom_flow_velocity_state, 0, sizeof(cv_bottom_flow_velocity_state));
  pthread_mutex_init(&bfv_mutex, NULL);
  bfv_prev_base_im.release();
  bfv_prev_pts.clear();
  bfv_has_prev = false;
  bfv_prev_t = 0;

#ifdef CV_BOTTOM_FLOW_VELOCITY_CAMERA
  cv_add_to_device(&CV_BOTTOM_FLOW_VELOCITY_CAMERA, cv_bottom_flow_velocity_cb, CV_BOTTOM_FLOW_VELOCITY_FPS, 0);
#endif
}

extern "C" void cv_bottom_flow_velocity_periodic(void)
{
  pthread_mutex_lock(&bfv_mutex);
  cv_bottom_flow_velocity_state.updated = false;
  pthread_mutex_unlock(&bfv_mutex);
}

