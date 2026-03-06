#include "gate_detector.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
extern "C" {
#include "modules/computer_vision/cv.h"
}
#include "modules/core/abi.h"

#ifndef GATE_DETECTOR_ABI_ID
#define GATE_DETECTOR_ABI_ID 2
#endif

#ifndef GATE_DETECTOR_CAMERA
#define GATE_DETECTOR_CAMERA front_camera
#endif

// ============================================================
// Config — swap the two blocks to switch sim / real
//
// REAL (cyberzoo yellow gate):
// namespace Config {
//     static const float GATE_W       = 1.0f;
//     static const float GATE_H       = 1.0f;
//     static const float H_MIN        = 2.f;
//     static const float H_MAX        = 17.f;
//     static const float S_MIN        = 134.f;
//     static const float V_MIN        = 126.f;
//     static const float WARP         = 0.013f;
//     static const float CONF_RATIO   = 0.17f;
//     static const float ALPHA        = 0.2f;
//     static const float MIN_AREA     = 700.f;
//     static const float MAX_SOLIDITY = 1.0f;  // disabled for real
// }
//
// SIMULATION:
namespace Config {
    static const float GATE_W       = 1.0f;
    static const float GATE_H       = 1.0f;
    static const float H_MIN        = 18.f;   // Cuts out red/orange, starts at Golden Yellow
    static const float H_MAX        = 32.f;   // Tightens around pure Yellow
    static const float S_MIN        = 165.f;  // More saturated — less washed out
    static const float V_MIN        = 130.f;  // Ignores dark shadows
    static const float WARP         = 0.013f;
    static const float CONF_RATIO   = 0.25f;  // Fraction of contour outline that must be yellow
    static const float ALPHA        = 0.3f;
    static const float MIN_AREA     = 450.f;  // Filters small distant noise
    static const float MAX_SOLIDITY = 0.60f;  // Hollow gate shapes only
}
// ============================================================

static float   s_dist = 0.f, s_ox = 0.f, s_oy = 0.f, s_yaw = 0.f;
static int16_t gate_found = 0;

// Sort 4 corner points into: top-left, top-right, bottom-right, bottom-left
// Matches Python's solve_gate_spatial corner ordering
static void sort_corners(std::vector<cv::Point2f>& pts)
{
    // Sort by Y (top two first)
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    // Top two: sort by X left→right
    if (pts[0].x > pts[1].x) std::swap(pts[0], pts[1]);
    // Bottom two: sort by X right→left
    if (pts[2].x < pts[3].x) std::swap(pts[2], pts[3]);
}

static struct image_t *gate_detect_cb(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    if (!img || img->type != IMAGE_YUV422) return NULL;

    int img_w = img->w;
    int img_h = img->h;

    // Convert YUV422 → BGR
    cv::Mat yuv(img_h, img_w, CV_8UC2, img->buf);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YUYV);

    // --- HSV mask (used for confidence check) ---
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv,
        cv::Scalar(Config::H_MIN, Config::S_MIN, Config::V_MIN),
        cv::Scalar(Config::H_MAX, 255, 255),
        mask);

    // --- Canny edges on grayscale (catches thin/distant gates better than mask alone) ---
    cv::Mat gray, blurred, edges, dilated;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 40, 120);
    cv::dilate(edges, dilated, cv::Mat::ones(3, 3, CV_8U));

    // Find contours on edge image (same as Python: RETR_LIST)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    gate_found = 0;

    for (auto& cnt : contours) {
        if (cv::contourArea(cnt) < Config::MIN_AREA) continue;

        // Solidity filter (hollow gate frame has low solidity)
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        double hull_area = cv::contourArea(hull);
        if (hull_area > 0.0 && (cv::contourArea(cnt) / hull_area) > Config::MAX_SOLIDITY) continue;

        // Approximate to polygon
        double peri = cv::arcLength(cnt, true);
        std::vector<cv::Point2f> approx;
        cv::approxPolyDP(cnt, approx, Config::WARP * peri, true);
        if (approx.size() != 4) continue;

        // Confidence check: fraction of contour outline that is yellow
        cv::Mat line_mask = cv::Mat::zeros(gray.size(), CV_8U);
        cv::drawContours(line_mask, std::vector<std::vector<cv::Point>>{cnt}, -1, 255, 6);
        double nonzero_line = cv::countNonZero(line_mask);
        cv::Mat overlap;
        cv::bitwise_and(mask, line_mask, overlap);
        double ratio = cv::countNonZero(overlap) / (nonzero_line + 1e-5);
        if (ratio < Config::CONF_RATIO) continue;

        // Sort corners: top-left, top-right, bottom-right, bottom-left
        sort_corners(approx);

        // PnP — same focal length assumption as Python (focal = img_w)
        std::vector<cv::Point3f> obj_pts = {
            {-Config::GATE_W/2,  Config::GATE_H/2, 0},
            { Config::GATE_W/2,  Config::GATE_H/2, 0},
            { Config::GATE_W/2, -Config::GATE_H/2, 0},
            {-Config::GATE_W/2, -Config::GATE_H/2, 0}
        };
        cv::Mat cam_matrix = (cv::Mat_<double>(3,3)
            << img_w, 0, img_w/2.0,
               0, img_w, img_h/2.0,
               0, 0, 1);
        cv::Mat rvec, tvec;
        if (!cv::solvePnP(obj_pts, approx, cam_matrix, cv::Mat(), rvec, tvec)) continue;

        float raw_dist = (float)cv::norm(tvec);
        // Pixel offset of centroid from image centre (matches Python off_x/off_y)
        float cx = 0.f, cy = 0.f;
        for (auto& p : approx) { cx += p.x; cy += p.y; }
        float raw_ox = (cx / 4.f) - (img_w / 2.f);
        float raw_oy = (cy / 4.f) - (img_h / 2.f);
        // Yaw from tvec (degrees, same as Python)
        float raw_yaw = (float)(atan2(tvec.at<double>(0), tvec.at<double>(2)) * 180.0 / M_PI);

        // Alpha smoothing
        s_dist = Config::ALPHA * raw_dist + (1.f - Config::ALPHA) * s_dist;
        s_ox   = Config::ALPHA * raw_ox   + (1.f - Config::ALPHA) * s_ox;
        s_oy   = Config::ALPHA * raw_oy   + (1.f - Config::ALPHA) * s_oy;
        s_yaw  = Config::ALPHA * raw_yaw  + (1.f - Config::ALPHA) * s_yaw;

        gate_found = 1;
        break;
    }

    return NULL;
}

extern "C" {

void gate_detector_init(void)
{
    cv_add_to_device(&GATE_DETECTOR_CAMERA, gate_detect_cb, 0, 0);
}

void gate_detector_periodic(void)
{
    // off_x / off_y in pixels (scaled down so they fit int16)
    // dist in mm, yaw in tenths of degree
    int16_t px_x = (int16_t)s_ox;
    int16_t px_y = (int16_t)s_oy;
    int32_t qual = (int32_t)(s_dist * 1000.f);   // metres → mm
    int16_t yaw  = (int16_t)(s_yaw * 10.f);      // degrees → tenths
    AbiSendMsgVISUAL_DETECTION(GATE_DETECTOR_ABI_ID,
                                px_x, px_y, yaw, 0,
                                qual, gate_found);
}

} // extern "C"