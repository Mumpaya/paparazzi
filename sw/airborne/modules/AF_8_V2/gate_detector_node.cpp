

#include "gate_detector_node.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

/* ── configuration (mirrors Python CONF dict)*/
static const cv::Scalar BLUE_HSV_LO(79, 90, 120);
static const cv::Scalar BLUE_HSV_HI(137, 220, 255);
static const cv::Scalar BGR_LO(70,  17,  20);
static const cv::Scalar BGR_HI(255, 200, 150);

static const float MIN_BBOX_FRAC      = 0.002f;
static const float MIN_ASPECT_RATIO   = 2.27f;
static const int   OPEN_KERNEL_SZ     = 3;
static const int   CLOSE_KERNEL_SZ    = 11;

/* gate validation */
static const float GATE_MAX_VERT_DIFF  = 0.20f;
static const float GATE_MIN_HORIZ_DIST = 0.10f;
static const float GATE_ANGLE_THRESH   = 0.50f;

/* distance calibration */
static const float DIST_CALIB_M        = 0.79f;

/* ── internal types ─────────────────────────────────────────────── */
struct PillarBBox {
    int x, y, w, h;
    int cx;   /* centre X */
    int cy;   /* centre Y */
};

/* ── find the two pillars with greatest horizontal separation ───── */
static int find_blue_pillars(const cv::Mat &mask, int img_h, int img_w,
                              PillarBBox best_pair[2])
{
    float min_bbox_px = MIN_BBOX_FRAC * img_h * img_w;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<PillarBBox> accepted;
    for (auto &cnt : contours) {
        cv::Rect br = cv::boundingRect(cnt);
        float bbox_area  = (float)(br.width * br.height);
        float aspect     = br.width > 0 ? (float)br.height / br.width : 0;
        if (bbox_area > min_bbox_px && aspect >= MIN_ASPECT_RATIO) {
            PillarBBox pb;
            pb.x  = br.x;
            pb.y  = br.y;
            pb.w  = br.width;
            pb.h  = br.height;
            pb.cx = br.x + br.width / 2;
            pb.cy = br.y + br.height / 2;
            accepted.push_back(pb);
        }
    }

    /* pick pair with greatest horizontal separation */
    int best_dist = -1;
    int n_found = 0;
    for (size_t i = 0; i < accepted.size(); i++) {
        for (size_t j = i + 1; j < accepted.size(); j++) {
            int dist = std::abs(accepted[i].cx - accepted[j].cx);
            if (dist > best_dist) {
                best_dist = dist;
                /* ensure left < right */
                if (accepted[i].cx <= accepted[j].cx) {
                    best_pair[0] = accepted[i];
                    best_pair[1] = accepted[j];
                } else {
                    best_pair[0] = accepted[j];
                    best_pair[1] = accepted[i];
                }
                n_found = 2;
            }
        }
    }
    return n_found;
}

/* ================================================================
 * PUBLIC API
 * ================================================================ */

void gate_detector_node_init(void)
{
}

void gate_detector_node_process(const uint8_t *bgr_data,
                                int width, int height,
                                GateResult *out)
{
    memset(out, 0, sizeof(*out));
    out->detected = false;
    out->dist_m   = -1.0f;

    cv::Mat frame(height, width, CV_8UC3, (void *)bgr_data);
    int h = height, w = width;

    /* 1. colour masking ──────────────────────────────────────────── */
    cv::Mat bgr_mask, hsv, hsv_mask, blue_mask;
    cv::inRange(frame, BGR_LO, BGR_HI, bgr_mask);
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI, hsv_mask);
    cv::bitwise_and(bgr_mask, hsv_mask, blue_mask);

    /* morphology: open to remove specks, close to bridge vertical gaps */
    cv::Mat open_k = cv::Mat::ones(OPEN_KERNEL_SZ, OPEN_KERNEL_SZ, CV_8U);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, open_k);

    cv::Mat close_k = cv::Mat::ones(CLOSE_KERNEL_SZ, 1, CV_8U);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, close_k);

    /* 2. detection ───────────────────────────────────────────────── */
    PillarBBox pillars[2];
    int n_pillars = find_blue_pillars(blue_mask, h, w, pillars);
    if (n_pillars < 2) return;

    /* 3. gate validation ─────────────────────────────────────────── */
    int cy0 = pillars[0].cy;
    int cy1 = pillars[1].cy;
    int vert_diff  = std::abs(cy0 - cy1);
    int horiz_dist = std::abs(pillars[1].cx - pillars[0].cx);

    bool vert_ok  = vert_diff  < (int)(GATE_MAX_VERT_DIFF  * h);
    bool horiz_ok = horiz_dist > (int)(GATE_MIN_HORIZ_DIST * w);
    if (!(vert_ok && horiz_ok)) return;

    /* gate centre */
    int gate_cx = (pillars[0].cx + pillars[1].cx) / 2;
    int gate_cy = (cy0 + cy1) / 2;

    /* approach angle */
    int area0 = pillars[0].w * pillars[0].h;
    int area1 = pillars[1].w * pillars[1].h;
    int angle;
    if (area0 > (int)(area1 * (1.0f + GATE_ANGLE_THRESH)))
        angle = GATE_ANGLE_LEFT;
    else if (area1 > (int)(area0 * (1.0f + GATE_ANGLE_THRESH)))
        angle = GATE_ANGLE_RIGHT;
    else
        angle = GATE_ANGLE_GOOD;

    /* distance estimate (meaningful when angle == GOOD) */
    float dist_m = horiz_dist > 0 ? DIST_CALIB_M * w / (float)horiz_dist : -1.0f;

    /* projected gate square */
    (void)horiz_dist;
    int gate_half = horiz_dist / 2;
    (void)gate_half;

    /* fill output */
    out->detected = true;
    out->cx_px    = gate_cx;
    out->cy_px    = gate_cy;
    out->cx_pct   = 100.0f * gate_cx / w;
    out->cy_pct   = 100.0f * gate_cy / h;
    out->angle    = angle;
    out->dist_m   = dist_m;
}
