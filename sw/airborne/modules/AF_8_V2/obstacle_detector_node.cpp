/*
 * obstacle_detector_node.cpp — Multi-stage obstacle detector
 *
 * Port of obstacle_detector_v5.py (active version with GapCandidate).
 * Stages 1-6 + fusion + gap ranking, all in OpenCV C++.
 */

#include "obstacle_detector_node.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

/* ── tunable parameters ─────────────────────────────────────────── */
static const int   N_COLS             = OBSTACLE_N_COLS;  /* from detection_types.h */
static const float ROI_TOP_FRAC       = 0.10f;
static const float ROI_BOTTOM_FRAC    = 0.90f;
static const float CLAHE_CLIP         = 4.0f;
static const int   CLAHE_TILE         = 8;

/* flow */
static const float FLOW_THRESHOLD     = 0.7f;
static const float IIR_ALPHA          = 0.35f;
static const int   MAX_CORNERS        = 150;
static const int   MIN_FEATURE_DIST   = 4;
static const int   REFRESH_INTERVAL   = 8;
static const float FLOW_WEIGHT        = 1.5f;

/* hough */
static const float HOUGH_WEIGHT       = 1.5f;
static const int   HOUGH_BLUR_K       = 5;
static const int   HOUGH_CANNY_LO     = 50;
static const int   HOUGH_CANNY_HI     = 150;
static const int   HOUGH_THRESHOLD    = 30;
static const int   HOUGH_MIN_LEN      = 40;
static const int   HOUGH_MAX_GAP      = 100;
static const float HOUGH_VERT_TOL_DEG = 20.0f;
static const int   HOUGH_CLUSTER_GAP  = 35;
static const int   HOUGH_MAX_BOX_W    = 200;
static const float HOUGH_MIN_ASPECT   = 1.5f;
static const float HOUGH_MAX_ASPECT   = 8.0f;
static const int   HOUGH_BORDER_MARG  = 15;
static const float HOUGH_MAX_EDGE_DEN = 0.15f;

/* orange */
static const float ORANGE_WEIGHT      = 2.5f;
static const cv::Scalar ORANGE_HSV_LO(0, 40, 50);
static const cv::Scalar ORANGE_HSV_HI(25, 255, 255);
static const int   ORANGE_MIN_AREA    = 300;
static const float ORANGE_MIN_ASPECT  = 1.5f;
static const float ORANGE_MIN_CONF    = 0.2f;

/* ground */
static const float GROUND_WEIGHT      = 2.0f;
static const cv::Scalar GREEN_HSV_LO(18, 17, 124);
static const cv::Scalar GREEN_HSV_HI(76, 153, 255);
static const float GROUND_MIN_MEAN    = 0.05f;

/* gap safety weights */
static const float GAP_W_WIDTH        = 0.50f;
static const float GAP_W_CLEAR        = 0.40f;
static const float GAP_W_CENTER       = 0.10f;

/* ── persistent state ───────────────────────────────────────────── */
static cv::Ptr<cv::CLAHE> clahe;
static cv::Mat prev_roi;
static std::vector<cv::Point2f> prev_pts;
static float flow_scores_smooth[N_COLS];
static float fused_scores_smooth[N_COLS];
static int   frame_count;

/* ── LK + feature params ───────────────────────────────────────── */
static cv::TermCriteria lk_criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                    20, 0.01);
static const cv::Size lk_win(21, 21);
static const int lk_levels = 3;

/* ── helper: clamp ──────────────────────────────────────────────── */
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ================================================================
 * Stage 1 — rotate + CLAHE + ROI
 * ================================================================ */
static void stage1(const cv::Mat &bgr,
                   cv::Mat &enhanced, cv::Mat &rotated,
                   int &roi_top, int &roi_bottom, cv::Mat &roi_frame)
{
    cv::rotate(bgr, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    int h = rotated.rows;
    (void)h;
    cv::Mat gray;
    cv::cvtColor(rotated, gray, cv::COLOR_BGR2GRAY);
    clahe->apply(gray, enhanced);
    roi_top    = (int)(h * ROI_TOP_FRAC);
    roi_bottom = (int)(h * ROI_BOTTOM_FRAC);
    roi_frame  = enhanced(cv::Range(roi_top, roi_bottom), cv::Range::all());
}

/* ================================================================
 * Stage 2+3 — optical flow + ego-motion compensation
 * ================================================================ */
static void stage2_3(const cv::Mat &roi_frame, float *flow_scores)
{
    int roi_w = roi_frame.cols;
    float col_width = (float)roi_w / N_COLS;

    /* first frame or too few points → seed features */
    if (prev_roi.empty() || prev_pts.size() < 4) {
        roi_frame.copyTo(prev_roi);
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(roi_frame, corners, MAX_CORNERS, 0.005,
                                MIN_FEATURE_DIST, cv::noArray(), 5);
        prev_pts = corners;
        for (int c = 0; c < N_COLS; c++) flow_scores[c] = flow_scores_smooth[c];
        return;
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_roi, roi_frame, prev_pts, curr_pts,
                             status, err, lk_win, lk_levels, lk_criteria);

    /* keep only good matches */
    std::vector<cv::Point2f> prev_good, curr_good;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prev_good.push_back(prev_pts[i]);
            curr_good.push_back(curr_pts[i]);
        }
    }

    if (prev_good.empty()) {
        roi_frame.copyTo(prev_roi);
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(roi_frame, corners, MAX_CORNERS, 0.005,
                                MIN_FEATURE_DIST, cv::noArray(), 5);
        prev_pts = corners;
        for (int c = 0; c < N_COLS; c++) flow_scores[c] = flow_scores_smooth[c];
        return;
    }

    /* flow vectors + ego-motion compensation */
    std::vector<float> vx(prev_good.size()), vy(prev_good.size());
    float mean_vx = 0, mean_vy = 0;
    for (size_t i = 0; i < prev_good.size(); i++) {
        vx[i] = curr_good[i].x - prev_good[i].x;
        vy[i] = curr_good[i].y - prev_good[i].y;
        mean_vx += vx[i];
        mean_vy += vy[i];
    }
    mean_vx /= prev_good.size();
    mean_vy /= prev_good.size();
    for (size_t i = 0; i < prev_good.size(); i++) {
        vx[i] -= mean_vx;
        vy[i] -= mean_vy;
    }

    /* column voting */
    float col_scores_raw[N_COLS] = {};
    int   col_counts[N_COLS]     = {};

    for (size_t i = 0; i < prev_good.size(); i++) {
        float mag = std::hypot(vx[i], vy[i]);
        int col = clampi((int)(prev_good[i].x / col_width), 0, N_COLS - 1);
        col_scores_raw[col] += mag;
        col_counts[col]++;
    }
    for (int c = 0; c < N_COLS; c++)
        if (col_counts[c] > 0) col_scores_raw[c] /= col_counts[c];

    float max_f = *std::max_element(col_scores_raw, col_scores_raw + N_COLS);
    if (max_f > 0)
        for (int c = 0; c < N_COLS; c++) col_scores_raw[c] /= max_f;

    /* IIR smoothing */
    for (int c = 0; c < N_COLS; c++)
        flow_scores_smooth[c] = IIR_ALPHA * col_scores_raw[c]
                               + (1.0f - IIR_ALPHA) * flow_scores_smooth[c];

    for (int c = 0; c < N_COLS; c++) flow_scores[c] = flow_scores_smooth[c];

    /* update state */
    roi_frame.copyTo(prev_roi);
    frame_count++;
    if (frame_count % REFRESH_INTERVAL == 0 ||
        (int)curr_good.size() < MAX_CORNERS / 3) {
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(roi_frame, corners, MAX_CORNERS, 0.005,
                                MIN_FEATURE_DIST, cv::noArray(), 5);
        prev_pts = corners;
        frame_count = 0;
    } else {
        prev_pts = curr_good;
    }
}

/* ================================================================
 * Stage 4 — Hough vertical line detector
 * ================================================================ */
static void stage4_hough(const cv::Mat &roi_frame, float *hough_scores)
{
    int roi_w = roi_frame.cols;
    float col_width = (float)roi_w / N_COLS;

    cv::Mat blurred, edges;
    cv::GaussianBlur(roi_frame, blurred, cv::Size(HOUGH_BLUR_K, HOUGH_BLUR_K), 0);
    cv::Canny(blurred, edges, HOUGH_CANNY_LO, HOUGH_CANNY_HI);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0,
                    HOUGH_THRESHOLD, HOUGH_MIN_LEN, HOUGH_MAX_GAP);

    /* filter vertical */
    struct VLine { int x1, y1, x2, y2; float mid_x; };
    std::vector<VLine> vertical;
    for (auto &l : lines) {
        int dx = std::abs(l[2] - l[0]);
        int dy = std::abs(l[3] - l[1]);
        float ang = (float)(180.0 / CV_PI) * std::atan2((float)dx, std::max((float)dy, 1.0f));
        if (ang <= HOUGH_VERT_TOL_DEG)
            vertical.push_back({l[0], l[1], l[2], l[3],
                                (l[0] + l[2]) * 0.5f});
    }

    /* cluster by x-midpoint → raw boxes */
    struct Box4 { int x1, y1, x2, y2; };
    std::vector<Box4> raw_boxes;

    if (!vertical.empty()) {
        std::sort(vertical.begin(), vertical.end(),
                  [](const VLine &a, const VLine &b) { return a.mid_x < b.mid_x; });

        std::vector<std::vector<VLine>> clusters;
        std::vector<VLine> cur = {vertical[0]};
        for (size_t i = 1; i < vertical.size(); i++) {
            if (std::fabs(vertical[i].mid_x - cur.back().mid_x) <= HOUGH_CLUSTER_GAP) {
                cur.push_back(vertical[i]);
            } else {
                clusters.push_back(cur);
                cur = {vertical[i]};
            }
        }
        clusters.push_back(cur);

        for (auto &cl : clusters) {
            int xmin = INT_MAX, ymin = INT_MAX, xmax = 0, ymax = 0;
            for (auto &v : cl) {
                xmin = std::min({xmin, v.x1, v.x2});
                xmax = std::max({xmax, v.x1, v.x2});
                ymin = std::min({ymin, v.y1, v.y2});
                ymax = std::max({ymax, v.y1, v.y2});
            }
            if (xmax - xmin <= HOUGH_MAX_BOX_W) {
                raw_boxes.push_back({xmin, ymin, xmax, ymax});
            } else {
                int xmid = (xmin + xmax) / 2;
                /* split into two halves */
                for (int side = 0; side < 2; side++) {
                    int hxmin = INT_MAX, hymin = INT_MAX, hxmax = 0, hymax = 0;
                    bool any = false;
                    for (auto &v : cl) {
                        bool use = side == 0 ? (v.mid_x <= xmid) : (v.mid_x > xmid);
                        if (use) {
                            hxmin = std::min({hxmin, v.x1, v.x2});
                            hxmax = std::max({hxmax, v.x1, v.x2});
                            hymin = std::min({hymin, v.y1, v.y2});
                            hymax = std::max({hymax, v.y1, v.y2});
                            any = true;
                        }
                    }
                    if (any) raw_boxes.push_back({hxmin, hymin, hxmax, hymax});
                }
            }
        }
    }

    /* filter boxes */
    std::vector<Box4> filtered;
    for (auto &b : raw_boxes) {
        int bw = std::max(b.x2 - b.x1, 1);
        int bh = std::max(b.y2 - b.y1, 1);
        if (b.x1 < HOUGH_BORDER_MARG || b.x2 > roi_w - HOUGH_BORDER_MARG)
            continue;
        float ratio = (float)bh / bw;
        if (ratio < HOUGH_MIN_ASPECT || ratio > HOUGH_MAX_ASPECT)
            continue;
        /* interior edge density check */
        int shrink = 10;
        int ix1 = std::min(b.x1 + shrink, b.x2);
        int ix2 = std::max(b.x2 - shrink, b.x1);
        int iy1 = std::min(b.y1 + shrink, b.y2);
        int iy2 = std::max(b.y2 - shrink, b.y1);
        if (ix2 > ix1 && iy2 > iy1) {
            cv::Mat roi = edges(cv::Range(iy1, iy2), cv::Range(ix1, ix2));
            float density = (float)cv::countNonZero(roi) / ((ix2 - ix1) * (iy2 - iy1));
            if (density > HOUGH_MAX_EDGE_DEN) continue;
        }
        filtered.push_back(b);
    }

    /* column scoring */
    for (int c = 0; c < N_COLS; c++) hough_scores[c] = 0.0f;
    for (auto &b : filtered) {
        for (int c = 0; c < N_COLS; c++) {
            int cx1 = (int)(c * col_width);
            int cx2 = (int)((c + 1) * col_width);
            int overlap = std::max(0, std::min(b.x2, cx2) - std::max(b.x1, cx1));
            if (overlap > 0) {
                float s = (float)overlap / std::max(cx2 - cx1, 1);
                hough_scores[c] = std::max(hough_scores[c], s);
            }
        }
    }
}

/* ================================================================
 * Stage 5 — Orange HSV detector
 * ================================================================ */
static void stage5_orange(const cv::Mat &rotated, int roi_top, int roi_bottom,
                          float *orange_scores,
                          PoleDetection *out_boxes, int *n_boxes)
{
    int roi_w = rotated.cols;
    float col_width = (float)roi_w / N_COLS;
    cv::Mat roi_bgr = rotated(cv::Range(roi_top, roi_bottom), cv::Range::all());

    cv::Mat hsv, mask;
    cv::cvtColor(roi_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, ORANGE_HSV_LO, ORANGE_HSV_HI, mask);

    cv::Mat morph_k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  morph_k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, morph_k);

    /* find pole contours */
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    *n_boxes = 0;
    for (auto &cnt : contours) {
        float area = (float)cv::contourArea(cnt);
        if (area < ORANGE_MIN_AREA) continue;
        cv::Rect br = cv::boundingRect(cnt);
        float aspect = (float)br.height / std::max(br.width, 1);
        if (aspect < ORANGE_MIN_ASPECT) continue;
        cv::Mat roi_mask = mask(br);
        float conf = (float)cv::countNonZero(roi_mask) / std::max(br.area(), 1);
        if (conf < ORANGE_MIN_CONF) continue;
        if (*n_boxes < MAX_ORANGE_BOXES) {
            PoleDetection &pd = out_boxes[*n_boxes];
            pd.x    = br.x;
            pd.y    = br.y;
            pd.w    = br.width;
            pd.h    = br.height;
            pd.cx   = br.x + br.width / 2;
            pd.cy   = br.y + br.height / 2;
            pd.area = (int)area;
            pd.confidence = conf;
            (*n_boxes)++;
        }
    }

    /* column scoring */
    for (int c = 0; c < N_COLS; c++) {
        int x1 = (int)(c * col_width);
        int x2 = (int)((c + 1) * col_width);
        cv::Mat col_mask = mask(cv::Range::all(), cv::Range(x1, x2));
        orange_scores[c] = (float)cv::countNonZero(col_mask) / std::max((int)col_mask.total(), 1);
    }
    float max_o = *std::max_element(orange_scores, orange_scores + N_COLS);
    if (max_o > 0)
        for (int c = 0; c < N_COLS; c++) orange_scores[c] /= max_o;
}

/* ================================================================
 * Stage 6 — Green ground mask
 * ================================================================ */
static void stage6_ground(const cv::Mat &rotated, int roi_top, int roi_bottom,
                          float *ground_obstacle_scores)
{
    int roi_w = rotated.cols;
    float col_width = (float)roi_w / N_COLS;
    cv::Mat roi_bgr = rotated(cv::Range(roi_top, roi_bottom), cv::Range::all());

    cv::Mat hsv, green_mask;
    cv::cvtColor(roi_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI, green_mask);

    cv::Mat morph_k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN,  morph_k);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, morph_k);

    float green_scores[N_COLS];
    for (int c = 0; c < N_COLS; c++) {
        int x1 = (int)(c * col_width);
        int x2 = (int)((c + 1) * col_width);
        cv::Mat col = green_mask(cv::Range::all(), cv::Range(x1, x2));
        green_scores[c] = (float)cv::countNonZero(col) / std::max((int)col.total(), 1);
    }

    float mean_green = 0;
    for (int c = 0; c < N_COLS; c++) mean_green += green_scores[c];
    mean_green /= N_COLS;

    if (mean_green < GROUND_MIN_MEAN) {
        for (int c = 0; c < N_COLS; c++) ground_obstacle_scores[c] = 0.0f;
    } else {
        float max_o = 0;
        for (int c = 0; c < N_COLS; c++) {
            ground_obstacle_scores[c] = clampf(mean_green - green_scores[c], 0.0f, 10.0f);
            if (ground_obstacle_scores[c] > max_o) max_o = ground_obstacle_scores[c];
        }
        if (max_o > 0)
            for (int c = 0; c < N_COLS; c++) ground_obstacle_scores[c] /= max_o;
    }
}

/* ================================================================
 * Gap finder — top-N free corridors ranked by safety
 * ================================================================ */
static void find_gaps(const int *obstacle_cols, int n_obs,
                      const float *fused, int roi_w,
                      GapCandidate *out_gaps, int *n_gaps)
{
    float col_width = (float)roi_w / N_COLS;

    /* build a "is-obstacle" flag array */
    bool is_obs[OBSTACLE_N_COLS] = {};
    for (int i = 0; i < n_obs; i++)
        is_obs[obstacle_cols[i]] = true;

    /* normalise fused for clearness calc */
    float fused_norm[N_COLS] = {};
    float max_f = 0;
    for (int c = 0; c < N_COLS; c++) if (fused[c] > max_f) max_f = fused[c];
    if (max_f > 0)
        for (int c = 0; c < N_COLS; c++) fused_norm[c] = clampf(fused[c] / max_f, 0, 1);

    /* enumerate contiguous free runs */
    struct Run { int start, end; };
    std::vector<Run> runs;
    int rs = -1;
    for (int c = 0; c < N_COLS; c++) {
        if (!is_obs[c]) {
            if (rs < 0) rs = c;
        } else {
            if (rs >= 0) { runs.push_back({rs, c - 1}); rs = -1; }
        }
    }
    if (rs >= 0) runs.push_back({rs, N_COLS - 1});

    /* score each run */
    struct Scored { GapCandidate gc; float safety; };
    std::vector<Scored> scored;
    for (auto &r : runs) {
        int w_cols = r.end - r.start + 1;
        float w_sc = (float)w_cols / N_COLS;

        float c_sum = 0;
        for (int c = r.start; c <= r.end; c++) c_sum += (1.0f - fused_norm[c]);
        float c_sc = c_sum / w_cols;

        float gap_centre = (r.start + r.end) * 0.5f;
        float img_centre = (N_COLS - 1) * 0.5f;
        float cent_sc = 1.0f - 2.0f * std::fabs(gap_centre - img_centre) / N_COLS;

        float safety = clampf(GAP_W_WIDTH * w_sc + GAP_W_CLEAR * c_sc + GAP_W_CENTER * cent_sc,
                              0.0f, 1.0f);

        GapCandidate gc;
        gc.rank = 0;
        gc.center_x = (int)((r.start + w_cols * 0.5f) * col_width);
        gc.col_start = r.start;
        gc.col_end   = r.end;
        gc.width_cols = w_cols;
        gc.safety = safety;
        gc.width_score = w_sc;
        gc.clearness_score = c_sc;
        gc.centrality_score = cent_sc;
        scored.push_back({gc, safety});
    }

    /* sort descending by safety, take top 3 */
    std::sort(scored.begin(), scored.end(),
              [](const Scored &a, const Scored &b) { return a.safety > b.safety; });

    *n_gaps = 0;
    for (int i = 0; i < (int)scored.size() && i < MAX_GAP_CANDIDATES; i++) {
        out_gaps[i] = scored[i].gc;
        out_gaps[i].rank = i + 1;
        (*n_gaps)++;
    }
}

/* ================================================================
 * PUBLIC API
 * ================================================================ */

void obstacle_detector_node_init(void)
{
    clahe = cv::createCLAHE(CLAHE_CLIP, cv::Size(CLAHE_TILE, CLAHE_TILE));
    prev_roi = cv::Mat();
    prev_pts.clear();
    memset(flow_scores_smooth,  0, sizeof(flow_scores_smooth));
    memset(fused_scores_smooth, 0, sizeof(fused_scores_smooth));
    frame_count = 0;
}

void obstacle_detector_node_process(const uint8_t *bgr_data,
                                    int width, int height,
                                    ObstacleResult *out)
{
    memset(out, 0, sizeof(*out));
    out->n_cols = N_COLS;
    out->gap_center_x = -1;

    cv::Mat bgr(height, width, CV_8UC3, (void *)bgr_data);

    /* Stage 1 */
    cv::Mat enhanced, rotated, roi_frame;
    int roi_top, roi_bottom;
    stage1(bgr, enhanced, rotated, roi_top, roi_bottom, roi_frame);

    /* Stage 2+3: optical flow */
    float flow_scores[N_COLS];
    stage2_3(roi_frame, flow_scores);

    /* Stage 4: Hough poles */
    float hough_scores[N_COLS];
    stage4_hough(roi_frame, hough_scores);

    /* Stage 5: orange detection */
    float orange_scores[N_COLS];
    stage5_orange(rotated, roi_top, roi_bottom,
                  orange_scores, out->orange_boxes, &out->n_orange_boxes);

    /* Stage 6: green ground */
    float ground_obs_scores[N_COLS];
    stage6_ground(rotated, roi_top, roi_bottom, ground_obs_scores);

    /* Fusion */
    float fused_raw[N_COLS];
    for (int c = 0; c < N_COLS; c++) {
        fused_raw[c] = flow_scores[c]      * FLOW_WEIGHT
                     + hough_scores[c]      * HOUGH_WEIGHT
                     + orange_scores[c]     * ORANGE_WEIGHT
                     + ground_obs_scores[c] * GROUND_WEIGHT;
    }

    for (int c = 0; c < N_COLS; c++)
        fused_scores_smooth[c] = IIR_ALPHA * fused_raw[c]
                                + (1.0f - IIR_ALPHA) * fused_scores_smooth[c];

    /* threshold → obstacle columns */
    float mean_f = 0, std_f = 0;
    for (int c = 0; c < N_COLS; c++) mean_f += fused_scores_smooth[c];
    mean_f /= N_COLS;
    for (int c = 0; c < N_COLS; c++) {
        float d = fused_scores_smooth[c] - mean_f;
        std_f += d * d;
    }
    std_f = std::sqrt(std_f / N_COLS);
    float threshold = mean_f + FLOW_THRESHOLD * std_f;

    out->n_obstacle_cols = 0;
    for (int c = 0; c < N_COLS; c++) {
        out->fused_scores[c] = fused_scores_smooth[c];
        if (fused_scores_smooth[c] > threshold) {
            if (out->n_obstacle_cols < MAX_OBSTACLE_COLS)
                out->obstacle_cols[out->n_obstacle_cols++] = c;
        }
    }

    /* Gap finding */
    int roi_w = rotated.cols;
    find_gaps(out->obstacle_cols, out->n_obstacle_cols,
              fused_scores_smooth, roi_w,
              out->gaps, &out->n_gaps);

    out->gap_center_x = out->n_gaps > 0 ? out->gaps[0].center_x : -1;

    /* ── Fill controller-ready flyzone fields ────────────────────── */
    float col_width = (float)roi_w / N_COLS;
    out->n_flyzones = out->n_gaps;

    for (int i = 0; i < out->n_gaps; i++) {
        const GapCandidate &g = out->gaps[i];
        out->flyzone[i].left_px   = (int)(g.col_start * col_width);
        out->flyzone[i].right_px  = (int)((g.col_end + 1) * col_width);
        out->flyzone[i].center_px = g.center_x;
        out->flyzone[i].width_px  = out->flyzone[i].right_px - out->flyzone[i].left_px;
        out->flyzone[i].safety    = g.safety;
    }
}
