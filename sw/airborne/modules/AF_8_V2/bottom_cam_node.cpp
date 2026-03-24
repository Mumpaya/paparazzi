/*
 * bottom_cam_node.cpp — Bottom camera ground edge detector
 *
 * Detects the ground edge from the bottom camera feed using:
 *   1. HSV green mask detection
 *   2. Non-uniformity mask (texture detection)
 *   3. Combined mask to find good pixels
 *   4. Centroid calculation
 *   5. Magnitude and over_edge thresholds
 *
 * Tuning parameters (can be exposed to datalink):
 *   • HSV_LOWER_GREEN, HSV_UPPER_GREEN — green colour range
 *   • BLUR_KSIZE — Gaussian blur kernel size
 *   • NON_UNIFORM_THRESH — texture detection threshold
 *   • EDGE_THRESHOLD_FRACTION — over_edge trigger (fraction of frame width)
 */

#include "bottom_cam_node.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cmath>

/* ══════════════════════════════════════════════════════════════════
 * Tuning Constants
 * ════════════════════════════════════════════════════════════════ */

/* HSV colour range for green mat */
#define HSV_LOWER_H_GREEN 0
#define HSV_LOWER_S_GREEN 15
#define HSV_LOWER_V_GREEN 120

#define HSV_UPPER_H_GREEN 66
#define HSV_UPPER_S_GREEN 225
#define HSV_UPPER_V_GREEN 255

/* HSV colour range for "not green" */
#define HSV_LOWER_H_NOT_GREEN 88
#define HSV_LOWER_S_NOT_GREEN 0
#define HSV_LOWER_V_NOT_GREEN 0

#define HSV_UPPER_H_NOT_GREEN 255
#define HSV_UPPER_S_NOT_GREEN 255
#define HSV_UPPER_V_NOT_GREEN 255

/* Image processing parameters */
#define BLUR_KSIZE 9
#define NON_UNIFORM_KSIZE 11
#define NON_UNIFORM_THRESH 15.0f

/* Threshold: centroid beyond 80% of half-width → over_edge = true */
#define EDGE_THRESHOLD_FRACTION 0.8f

/* ══════════════════════════════════════════════════════════════════
 * Local State
 * ════════════════════════════════════════════════════════════════ */

static cv::Scalar hsv_lower_green;
static cv::Scalar hsv_upper_green;
static cv::Scalar hsv_lower_not_green;
static cv::Scalar hsv_upper_not_green;

/* ══════════════════════════════════════════════════════════════════
 * Helper: Compute non-uniformity mask (texture detection)
 * ════════════════════════════════════════════════════════════════ */
// static cv::Mat compute_non_uniform_mask(const cv::Mat &bgr, int ksize = NON_UNIFORM_KSIZE,
//                                         float thresh = NON_UNIFORM_THRESH)
// {
//     /* Convert to grayscale */
//     cv::Mat gray;
//     cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
//     gray.convertTo(gray, CV_32F);

//     /* Local mean and local variance via box filter */
//     cv::Mat local_mean, local_sq_mean;
//     cv::blur(gray, local_mean, cv::Size(ksize, ksize));
//     cv::Mat gray_sq = gray.mul(gray);
//     cv::blur(gray_sq, local_sq_mean, cv::Size(ksize, ksize));

//     /* Local standard deviation */
//     cv::Mat local_var = local_sq_mean - local_mean.mul(local_mean);
//     cv::Mat local_std;
//     cv::sqrt(cv::max(local_var, 0.0f), local_std);

//     /* Threshold: high std → non-uniform (texture) */
//     cv::Mat mask = (local_std > thresh);
//     cv::Mat result;
//     mask.convertTo(result, CV_8U, 255.0);
//     return result;
// }

static cv::Mat compute_non_uniform_mask(const cv::Mat &bgr, float thresh = NON_UNIFORM_THRESH)
{
    cv::Mat gray, lap;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_16S, 3);
    cv::Mat lap_abs;
    cv::convertScaleAbs(lap, lap_abs);
    cv::Mat result;
    cv::threshold(lap_abs, result, thresh, 255, cv::THRESH_BINARY);
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════ */
void bottom_cam_node_init(void)
{
    /* Pre-compute HSV bounds */
    hsv_lower_green = cv::Scalar(HSV_LOWER_H_GREEN, HSV_LOWER_S_GREEN, HSV_LOWER_V_GREEN);
    hsv_upper_green = cv::Scalar(HSV_UPPER_H_GREEN, HSV_UPPER_S_GREEN, HSV_UPPER_V_GREEN);
    hsv_lower_not_green = cv::Scalar(HSV_LOWER_H_NOT_GREEN, HSV_LOWER_S_NOT_GREEN, HSV_LOWER_V_NOT_GREEN);
    hsv_upper_not_green = cv::Scalar(HSV_UPPER_H_NOT_GREEN, HSV_UPPER_S_NOT_GREEN, HSV_UPPER_V_NOT_GREEN);
}

/* ══════════════════════════════════════════════════════════════════
 * Main Detection Function
 * ════════════════════════════════════════════════════════════════ */
void bottom_cam_node_process(const uint8_t *bgr_data, int width, int height,
                              BottomCamResult *result)
{
    if (!bgr_data || !result || width <= 0 || height <= 0) {
        result->detected = false;
        return;
    }

    /* Wrap raw buffer into OpenCV Mat (BGR, continuous) */
    cv::Mat bgr(height, width, CV_8UC3, (uint8_t *)bgr_data);

    // Downsample to half res — 4x fewer pixels for all operations
    cv::Mat bgr_small;
    cv::resize(bgr, bgr_small, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    int proc_w = bgr_small.cols, proc_h = bgr_small.rows;



    /* ── Blur and convert to HSV ────────────────────────────── */
    cv::Mat blurred;
    cv::GaussianBlur(bgr_small, blurred, cv::Size(BLUR_KSIZE, BLUR_KSIZE), 0);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    /* ── Create masks ────────────────────────────────────────── */
    cv::Mat mask_green;
    cv::Mat mask_not_green;
    cv::inRange(hsv, hsv_lower_green, hsv_upper_green, mask_green);
    cv::Mat mask_non_uniform = compute_non_uniform_mask(bgr_small);
    cv::inRange(hsv, hsv_lower_not_green, hsv_upper_not_green, mask_not_green);

    /* ── Combine masks ────────────────────────────────────────── */
    /* mask_bad = not_green AND NOT non_uniform (bad pixels) */
    // cv::Mat mask_bad = cv::Mat::zeros(height, width, CV_8U);
    // for (int i = 0; i < height * width; i++) {
    //     uint8_t ng = mask_not_green.data[i];
    //     uint8_t nu = mask_non_uniform.data[i];
    //     if (ng > 0 && nu == 0) {
    //         mask_bad.data[i] = 255;
    //     }
    // }

    // FAST: replace with bitwise ops
    cv::Mat mask_uniform;
    cv::bitwise_not(mask_non_uniform, mask_uniform);
    cv::Mat mask_bad;
    cv::bitwise_and(mask_not_green, mask_uniform, mask_bad);

    /* mask_good = (green OR non_uniform) AND NOT bad */
    cv::Mat mask_good;
    cv::bitwise_or(mask_green, mask_non_uniform, mask_good);
    cv::subtract(mask_good, mask_bad, mask_good);

    /* ── Compute goodness (fraction of good pixels) ────────────── */
    double good_pixels = cv::countNonZero(mask_good);
    // double total_pixels = width * height;
    double total_pixels = proc_w * proc_h;  // ← was: width * height
    float goodness = (float)(good_pixels / total_pixels);


    /* ── Compute centroid ────────────────────────────────────── */
    cv::Moments m = cv::moments(mask_good);
    int32_t cx_px = 0, cy_px = 0;
    bool has_centroid = false;

    // if (m.m00 > 0) {
    //     cx_px = (int32_t)(m.m10 / m.m00);
    //     cy_px = (int32_t)(m.m01 / m.m00);
    //     has_centroid = true;
    // }

    if (m.m00 > 0) {
        cx_px = (int32_t)(m.m10 / m.m00) * 2;  // ← scale back to full res
        cy_px = (int32_t)(m.m01 / m.m00) * 2;  // ← scale back to full res
        has_centroid = true;
    }


    /* ── Compute magnitude and over_edge ─────────────────────── */
    float magnitude = 0.0f;
    bool over_edge = false;

    if (has_centroid) {
        int frame_cx = width / 2;
        int frame_cy = height / 2;
        float dx = (float)(cx_px - frame_cx);
        float dy = (float)(cy_px - frame_cy);
        magnitude = sqrtf(dx * dx + dy * dy);

        /* Threshold: 80% of half-width */
        float edge_thresh = (width / 2.0f) * EDGE_THRESHOLD_FRACTION;
        over_edge = (magnitude > edge_thresh);
    }

    /* ── Write result ────────────────────────────────────────── */
    result->cx_px = cx_px;
    result->cy_px = cy_px;
    result->goodness = goodness;
    result->magnitude = magnitude;
    result->over_edge = over_edge;
    result->detected = has_centroid;
}
