/*
 * ground_edge_node.cpp — Green-ground boundary detector
 *
 * Port of ground_edge_detector.py (active version at bottom of file).
 * Uses OpenCV C++ API internally, exposes extern "C" interface.
 */

#include "ground_edge_node.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>

/* ── tunable parameters (mirrors Python defaults) ───────────────── */
static const cv::Scalar HSV_LOWER(18, 17, 124);
static const cv::Scalar HSV_UPPER(76, 153, 255);
static const int   BLUR_KSIZE       = 9;
static const int   MORPH_KSIZE      = 5;
static const int   MIN_AREA         = 500;
static const int   HOUGH_THRESHOLD  = 54;
static const int   HOUGH_MIN_LENGTH = 80;
static const int   HOUGH_MAX_GAP    = 36;
static const int   HISTORY_LEN      = 5;
static const int   CONFIRM_FRAMES   = 2;
static const float ANGLE_TOL_DEG    = 12.0f;
static const float DIST_TOL_PX      = 34.0f;
static const int   SAMPLE_DIST      = 20;
static const int   N_SAMPLES        = 20;
static const float GREEN_FRAC_THRESH = 0.3f;
static const float HUE_VAR_THRESH    = 15.0f;
static const float OSCILLATION_THRESH = 0.25f;

/* derived */
static float angle_tol_rad;

/* ── per-frame history entry ────────────────────────────────────── */
struct LineRecord {
    float angle;
    float mid_x, mid_y;
    cv::Point p1, p2;
};

static std::deque<std::vector<LineRecord>> history;

/* ── helpers ────────────────────────────────────────────────────── */

static void line_params(cv::Point p1, cv::Point p2,
                        float &angle, float &mid_x, float &mid_y)
{
    float dx = (float)(p2.x - p1.x);
    float dy = (float)(p2.y - p1.y);
    angle = std::atan2(std::fabs(dy), std::fabs(dx));
    mid_x = (p1.x + p2.x) * 0.5f;
    mid_y = (p1.y + p2.y) * 0.5f;
}

static bool lines_match(float a1, float mx1, float my1,
                         float a2, float mx2, float my2)
{
    float ad = std::fabs(a1 - a2);
    ad = std::min(ad, (float)CV_PI - ad);
    float dist = std::hypot(mx1 - mx2, my1 - my2);
    return ad < angle_tol_rad && dist < DIST_TOL_PX;
}

static void extend_line(cv::Point p1, cv::Point p2, int img_w,
                         cv::Point &ep1, cv::Point &ep2)
{
    if (p2.x == p1.x) {
        ep1 = cv::Point(p1.x, 0);
        ep2 = cv::Point(p1.x, 9999);
        return;
    }
    float slope = (float)(p2.y - p1.y) / (float)(p2.x - p1.x);
    int y_left  = (int)(p1.y + slope * (0 - p1.x));
    int y_right = (int)(p1.y + slope * (img_w - 1 - p1.x));
    ep1 = cv::Point(0, y_left);
    ep2 = cv::Point(img_w - 1, y_right);
}

/* Sample both sides of a line for green fraction & hue variance */
static void sample_sides(cv::Point p1, cv::Point p2,
                          const cv::Mat &mask, const cv::Mat &hsv,
                          float &left_frac, float &right_frac,
                          float &left_std, float &right_std)
{
    float dx = (float)(p2.x - p1.x);
    float dy = (float)(p2.y - p1.y);
    float length = std::hypot(dx, dy);
    if (length < 1.0f) {
        left_frac = right_frac = 0.0f;
        left_std  = right_std  = 999.0f;
        return;
    }
    float px = -dy / length;
    float py =  dx / length;
    int h = mask.rows, w = mask.cols;

    std::vector<int> lh, rh;
    int lhits = 0, rhits = 0, ln = 0, rn = 0;

    for (int i = 0; i < N_SAMPLES; i++) {
        float t = (float)i / std::max(N_SAMPLES - 1, 1);
        int lx = (int)(p1.x + t * dx);
        int ly = (int)(p1.y + t * dy);

        int lx_l = (int)(lx + px * SAMPLE_DIST);
        int ly_l = (int)(ly + py * SAMPLE_DIST);
        int lx_r = (int)(lx - px * SAMPLE_DIST);
        int ly_r = (int)(ly - py * SAMPLE_DIST);

        if (ly_l >= 0 && ly_l < h && lx_l >= 0 && lx_l < w) {
            ln++;
            if (mask.at<uint8_t>(ly_l, lx_l) > 0) lhits++;
            lh.push_back((int)hsv.at<cv::Vec3b>(ly_l, lx_l)[0]);
        }
        if (ly_r >= 0 && ly_r < h && lx_r >= 0 && lx_r < w) {
            rn++;
            if (mask.at<uint8_t>(ly_r, lx_r) > 0) rhits++;
            rh.push_back((int)hsv.at<cv::Vec3b>(ly_r, lx_r)[0]);
        }
    }

    left_frac  = ln > 0 ? (float)lhits / ln : 0.0f;
    right_frac = rn > 0 ? (float)rhits / rn : 0.0f;

    auto calc_std = [](const std::vector<int> &v) -> float {
        if (v.empty()) return 999.0f;
        float mean = 0;
        for (int x : v) mean += x;
        mean /= v.size();
        float var = 0;
        for (int x : v) var += (x - mean) * (x - mean);
        return std::sqrt(var / v.size());
    };
    left_std  = calc_std(lh);
    right_std = calc_std(rh);
}

/* Check if green side oscillates (mat stripes) */
static float green_side_oscillates(cv::Point p1, cv::Point p2,
                                    const cv::Mat &mask)
{
    float dx = (float)(p2.x - p1.x);
    float dy = (float)(p2.y - p1.y);
    float length = std::hypot(dx, dy);
    if (length < 1.0f) return 0.0f;

    float px = -dy / length;
    float py =  dx / length;
    int h = mask.rows, w = mask.cols;

    std::vector<float> left_vals, right_vals;

    for (int i = 0; i < N_SAMPLES; i++) {
        float t = (float)i / std::max(N_SAMPLES - 1, 1);
        int lx = (int)(p1.x + t * dx);
        int ly = (int)(p1.y + t * dy);

        int lx_l = (int)(lx + px * SAMPLE_DIST);
        int ly_l = (int)(ly + py * SAMPLE_DIST);
        int lx_r = (int)(lx - px * SAMPLE_DIST);
        int ly_r = (int)(ly - py * SAMPLE_DIST);

        if (ly_l >= 0 && ly_l < h && lx_l >= 0 && lx_l < w)
            left_vals.push_back(mask.at<uint8_t>(ly_l, lx_l) > 0 ? 1.0f : 0.0f);
        if (ly_r >= 0 && ly_r < h && lx_r >= 0 && lx_r < w)
            right_vals.push_back(mask.at<uint8_t>(ly_r, lx_r) > 0 ? 1.0f : 0.0f);
    }

    /* pick the greener side */
    auto mean_v = [](const std::vector<float> &v) -> float {
        if (v.empty()) return 0.0f;
        float s = 0;
        for (float x : v) s += x;
        return s / v.size();
    };
    const std::vector<float> &dom =
        mean_v(left_vals) >= mean_v(right_vals) ? left_vals : right_vals;

    if (dom.size() < 2) return 0.0f;
    float diffs = 0;
    for (size_t i = 0; i + 1 < dom.size(); i++)
        diffs += std::fabs(dom[i + 1] - dom[i]);
    return diffs / (dom.size() - 1);
}

/* Returns true if line should be rejected (mat edge) */
static bool is_mat_edge(cv::Point p1, cv::Point p2,
                         const cv::Mat &mask, const cv::Mat &hsv)
{
    float lf, rf, ls, rs;
    sample_sides(p1, p2, mask, hsv, lf, rf, ls, rs);

    /* both sides green → mat interior */
    if (lf > GREEN_FRAC_THRESH && rf > GREEN_FRAC_THRESH)
        return true;
    /* high hue variance on the green side → printed mat */
    if (lf > GREEN_FRAC_THRESH && ls > HUE_VAR_THRESH)
        return true;
    if (rf > GREEN_FRAC_THRESH && rs > HUE_VAR_THRESH)
        return true;
    /* oscillating green → mat stripes */
    float osc = green_side_oscillates(p1, p2, mask);
    if (osc > OSCILLATION_THRESH)
        return true;

    return false;
}

/* ================================================================
 * PUBLIC API
 * ================================================================ */

void ground_edge_node_init(void)
{
    angle_tol_rad = ANGLE_TOL_DEG * (float)CV_PI / 180.0f;
    history.clear();
}

void ground_edge_node_process(const uint8_t *bgr_data,
                              int width, int height,
                              GroundEdgeResult *out)
{
    /* zero output */
    memset(out, 0, sizeof(*out));

    /* wrap raw pointer into cv::Mat (no copy) */
    cv::Mat bgr(height, width, CV_8UC3, (void *)bgr_data);

    /* 1. Gaussian blur + HSV mask */
    cv::Mat blurred, hsv, mask;
    cv::GaussianBlur(bgr, blurred, cv::Size(BLUR_KSIZE, BLUR_KSIZE), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, HSV_LOWER, HSV_UPPER, mask);

    /* morphology close + open */
    cv::Mat kernel_close = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(MORPH_KSIZE, MORPH_KSIZE));
    cv::Mat kernel_open = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel_open);

    /* 2. boundary = mask - erode(mask) */
    cv::Mat eroded, boundary;
    cv::Mat erode_k = cv::Mat::ones(3, 3, CV_8U);
    cv::erode(mask, eroded, erode_k, cv::Point(-1, -1), 1);
    cv::subtract(mask, eroded, boundary);

    /* 3. HoughLinesP on boundary */
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(boundary, raw_lines, 1.0, CV_PI / 180.0,
                    HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP);

    /* build this frame's line records */
    std::vector<LineRecord> frame_lines;
    for (const auto &l : raw_lines) {
        LineRecord lr;
        lr.p1 = cv::Point(l[0], l[1]);
        lr.p2 = cv::Point(l[2], l[3]);
        line_params(lr.p1, lr.p2, lr.angle, lr.mid_x, lr.mid_y);
        frame_lines.push_back(lr);
    }

    /* push into temporal history ring buffer */
    history.push_back(frame_lines);
    if ((int)history.size() > HISTORY_LEN)
        history.pop_front();

    out->has_lines = !frame_lines.empty();

    /* 4. temporal confirmation + mat rejection */
    int n_conf = 0, n_rej = 0;

    for (const auto &fl : frame_lines) {
        /* count how many past frames contain a matching line */
        int match_count = 0;
        for (const auto &past : history) {
            for (const auto &pl : past) {
                if (lines_match(fl.angle, fl.mid_x, fl.mid_y,
                                pl.angle, pl.mid_x, pl.mid_y)) {
                    match_count++;
                    break;
                }
            }
        }
        if (match_count < CONFIRM_FRAMES)
            continue;

        cv::Point ep1, ep2;
        extend_line(fl.p1, fl.p2, width, ep1, ep2);

        if (is_mat_edge(fl.p1, fl.p2, mask, hsv)) {
            if (n_rej < MAX_REJECTED_LINES) {
                out->rejected[n_rej].ep1 = {ep1.x, ep1.y};
                out->rejected[n_rej].ep2 = {ep2.x, ep2.y};
                n_rej++;
            }
        } else {
            if (n_conf < MAX_CONFIRMED_LINES) {
                out->confirmed[n_conf].ep1     = {ep1.x, ep1.y};
                out->confirmed[n_conf].ep2     = {ep2.x, ep2.y};
                out->confirmed[n_conf].orig_p1 = {fl.p1.x, fl.p1.y};
                out->confirmed[n_conf].orig_p2 = {fl.p2.x, fl.p2.y};
                n_conf++;
            }
        }
    }
    out->n_confirmed = n_conf;
    out->n_rejected  = n_rej;
}
