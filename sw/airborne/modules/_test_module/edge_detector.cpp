#include "edge_detector.h"
#include "opencv_image_functions.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdint>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

enum GroundCalibStatus {
  CALIB_IDLE = 0,
  CALIB_RUNNING = 1,
  CALIB_OK = 2,
  CALIB_FAIL = 3
};

static bool     g_calib_active = false;
static bool     g_calib_pending = false;
static int      g_calib_status = CALIB_IDLE;
static int      g_calib_frames = 0;
static int      g_calib_status_ttl = 0;
static const int g_calib_target_frames = 30;
static uint32_t g_hist_h[181];
static uint32_t g_hist_s[256];
static uint32_t g_hist_v[256];
static uint32_t g_calib_samples = 0;
static int      g_cal_h_lo = 10, g_cal_h_hi = 35;
static int      g_cal_s_lo = 5,  g_cal_s_hi = 80;
static int      g_cal_v_lo = 120, g_cal_v_hi = 255;

static void calib_clear_hist(void)
{
  memset(g_hist_h, 0, sizeof(g_hist_h));
  memset(g_hist_s, 0, sizeof(g_hist_s));
  memset(g_hist_v, 0, sizeof(g_hist_v));
  g_calib_samples = 0;
  g_calib_frames = 0;
}

static int hist_percentile(const uint32_t *hist, int n_bins,
                           uint32_t total, float p)
{
  if (total == 0 || n_bins <= 0) {
    return 0;
  }
  if (p < 0.f) p = 0.f;
  if (p > 1.f) p = 1.f;

  uint32_t target = (uint32_t)((float)(total - 1) * p);
  uint32_t acc = 0;
  for (int i = 0; i < n_bins; i++) {
    acc += hist[i];
    if (acc > target) {
      return i;
    }
  }
  return n_bins - 1;
}

static void enforce_span(int *lo, int *hi, int min_span, int lo_lim, int hi_lim)
{
  int span = *hi - *lo;
  if (span >= min_span) {
    return;
  }
  int center = (*hi + *lo) / 2;
  int half = min_span / 2;
  *lo = center - half;
  *hi = center + (min_span - half);
  if (*lo < lo_lim) {
    int d = lo_lim - *lo;
    *lo += d;
    *hi += d;
  }
  if (*hi > hi_lim) {
    int d = *hi - hi_lim;
    *lo -= d;
    *hi -= d;
  }
  *lo = max(*lo, lo_lim);
  *hi = min(*hi, hi_lim);
}

void obstacle_start_ground_calibration(void)
{
  calib_clear_hist();
  g_calib_active = true;
  g_calib_pending = false;
  g_calib_status = CALIB_RUNNING;
  g_calib_status_ttl = 90;
}

int obstacle_consume_calibrated_hsv(int *h_lo, int *h_hi,
                                    int *s_lo, int *s_hi,
                                    int *v_lo, int *v_hi)
{
  if (!g_calib_pending) {
    return 0;
  }
  if (!h_lo || !h_hi || !s_lo || !s_hi || !v_lo || !v_hi) {
    return 0;
  }

  *h_lo = g_cal_h_lo;
  *h_hi = g_cal_h_hi;
  *s_lo = g_cal_s_lo;
  *s_hi = g_cal_s_hi;
  *v_lo = g_cal_v_lo;
  *v_hi = g_cal_v_hi;
  g_calib_pending = false;
  return 1;
}

/* ====================================================================== */
/*  Helper: Euclidean segment length                                      */
/* ====================================================================== */
static inline float seg_length(int x1, int y1, int x2, int y2)
{
  float dx = (float)(x2 - x1);
  float dy = (float)(y2 - y1);
  return sqrtf(dx * dx + dy * dy);
}

static inline float clampf(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}


struct edge_result detect_edges(char *img, int width, int height,
                                int canny_low, int canny_high)
{
  struct edge_result res = {0.f, 0.f, 0.f, 0.f, 0};

  Mat yuv(height, width, CV_8UC2, img);
  Mat gray;
  cvtColor(yuv, gray, cv::COLOR_YUV2GRAY_Y422);
  GaussianBlur(gray, gray, Size(5, 5), 1.5);

  Mat edges;
  Canny(gray, edges, canny_low, canny_high);

  vector<Vec4i> lines;
  HoughLinesP(edges, lines, 1, CV_PI / 180, 30, 20, 10);

  int half_h = height / 2;
  int mid_x  = width / 2;
  float front_total = 0.f, front_left = 0.f, front_right = 0.f, longest = 0.f;
  int front_count = 0;

  for (size_t i = 0; i < lines.size(); i++) {
    int x1 = lines[i][0], y1 = lines[i][1];
    int x2 = lines[i][2], y2 = lines[i][3];
    float mx = (float)(x1 + x2) * 0.5f;
    float my = (float)(y1 + y2) * 0.5f;
    if (my < (float)half_h) {
      float len = seg_length(x1, y1, x2, y2);
      front_total += len;
      front_count++;
      if (mx < (float)mid_x) front_left  += len;
      else                    front_right += len;
      if (len > longest) longest = len;
    }
  }

  res.front_line_length       = front_total;
  res.front_left_line_length  = front_left;
  res.front_right_line_length = front_right;
  res.longest_line            = longest;
  res.line_count              = front_count;

  grayscale_opencv_to_yuv422(edges, img, width, height);
  return res;
}

/* ====================================================================== */
/*  Default configuration                                                 */
/* ====================================================================== */
struct ObstacleConfig obstacle_config_defaults(void)
{
  struct ObstacleConfig c;
  memset(&c, 0, sizeof(c));
// these will be recalibrated later anyways
c.hsv_h_lo = 10;   
c.hsv_h_hi = 35;  

c.hsv_s_lo = 5;    
c.hsv_s_hi = 80;   

c.hsv_v_lo = 120; 
c.hsv_v_hi = 255;

/* morphology for smoothing area */
c.morph_ksize = 5;


  /* Grid: 3 columns × 1 row covering front 2/3 */
  c.grid_cols = 3;
  c.grid_rows = 1;

  /* Scoring weights */
  c.w_ng = 0.50f;
  c.w_ed = 0.30f;
  c.w_ll = 0.20f;

  /* EMA */
  c.ema_alpha = 0.3f;

  /* Score threshold for "found" flag */
  c.score_threshold = 0.25f;

  /* Downscale for Canny/Hough */
  c.ds_width  = 160;
  c.ds_height = 120;

  /* Min contour area (full-res pixels) */
  c.min_contour_area = 200;

  /* Edge cleanup and gating for real-world noise */
  c.edge_open_ksize = 3;
  c.min_line_len_px = 20;
  c.min_non_green_for_edges = 0.06f;

  return c;
}

/* ====================================================================== */
/*  Cached structuring elements     */
/* ====================================================================== */
static Mat s_morph_kern;
static int s_morph_ksize = 0;
static Mat s_edge_kern;
static int s_edge_ksize = 0;

static Mat get_morph_kern(int ksize)
{
  if (ksize != s_morph_ksize || s_morph_kern.empty()) {
    s_morph_kern = getStructuringElement(MORPH_ELLIPSE, Size(ksize, ksize));
    s_morph_ksize = ksize;
  }
  return s_morph_kern;
}

static Mat get_edge_kern(int ksize)
{
  if (ksize != s_edge_ksize || s_edge_kern.empty()) {
    s_edge_kern = getStructuringElement(MORPH_ELLIPSE, Size(ksize, ksize));
    s_edge_ksize = ksize;
  }
  return s_edge_kern;
}

/* ====================================================================== */
/*  EMA state – file-static, persists across frames                       */
/* ====================================================================== */
static float ema_score    = 0.f;
static float ema_centroid = 0.f;
static bool  ema_init     = false;

/* ====================================================================== */
/*  Fused obstacle detection pipeline                                     */
/* ====================================================================== */
struct obstacle_result detect_obstacles(char *img, int width, int height,
                                        int canny_low, int canny_high,
                                        struct ObstacleConfig cfg)
{
  struct obstacle_result out;
  memset(&out, 0, sizeof(out));

  /* ── 1. Convert YUV422 → grayscale + downscale early ────────────────── */
  Mat yuv(height, width, CV_8UC2, img);

  int ds_w = cfg.ds_width;
  int ds_h = cfg.ds_height;
  float sx = (float)width  / (float)ds_w;
  float sy = (float)height / (float)ds_h;

  /* Grayscale at full res, then downscale */
  Mat gray;
  cvtColor(yuv, gray, cv::COLOR_YUV2GRAY_Y422);
  Mat gray_ds;
  resize(gray, gray_ds, Size(ds_w, ds_h), 0, 0, INTER_AREA);

  /* BGR + HSV at downscaled resolution (major speedup) */
  Mat bgr_full, bgr_ds, hsv_ds;
  cvtColor(yuv, bgr_full, cv::COLOR_YUV2BGR_Y422);
  resize(bgr_full, bgr_ds, Size(ds_w, ds_h), 0, 0, INTER_AREA);
  cvtColor(bgr_ds, hsv_ds, cv::COLOR_BGR2HSV);

  /* ── Calibration sampling (on downscaled HSV) ─────────────────────── */
  if (g_calib_active) {
    int x0 = (ds_w * 25) / 100;
    int x1 = (ds_w * 75) / 100;
    int y0 = (ds_h * 55) / 100;
    int y1 = (ds_h * 95) / 100;
    x0 = max(0, min(x0, ds_w - 1));
    x1 = max(x0 + 1, min(x1, ds_w));
    y0 = max(0, min(y0, ds_h - 1));
    y1 = max(y0 + 1, min(y1, ds_h));

    for (int r = y0; r < y1; r++) {
      const Vec3b *ph = hsv_ds.ptr<Vec3b>(r);
      for (int c = x0; c < x1; c++) {
        int h = max(0, min((int)ph[c][0], 180));
        int s = max(0, min((int)ph[c][1], 255));
        int v = max(0, min((int)ph[c][2], 255));
        g_hist_h[h]++;
        g_hist_s[s]++;
        g_hist_v[v]++;
        g_calib_samples++;
      }
    }
    g_calib_frames++;
    if (g_calib_frames >= g_calib_target_frames) {
      if (g_calib_samples < 500) {
        g_calib_status = CALIB_FAIL;
        g_calib_status_ttl = 120;
      } else {
        int h10 = hist_percentile(g_hist_h, 181, g_calib_samples, 0.10f);
        int h90 = hist_percentile(g_hist_h, 181, g_calib_samples, 0.90f);
        int s10 = hist_percentile(g_hist_s, 256, g_calib_samples, 0.10f);
        int s90 = hist_percentile(g_hist_s, 256, g_calib_samples, 0.90f);
        int v10 = hist_percentile(g_hist_v, 256, g_calib_samples, 0.10f);
        int v90 = hist_percentile(g_hist_v, 256, g_calib_samples, 0.90f);
        g_cal_h_lo = max(0, h10 - 6);  g_cal_h_hi = min(180, h90 + 6);
        g_cal_s_lo = max(0, s10 - 18); g_cal_s_hi = min(255, s90 + 18);
        g_cal_v_lo = max(0, v10 - 18); g_cal_v_hi = min(255, v90 + 18);
        enforce_span(&g_cal_h_lo, &g_cal_h_hi, 12, 0, 180);
        enforce_span(&g_cal_s_lo, &g_cal_s_hi, 30, 0, 255);
        enforce_span(&g_cal_v_lo, &g_cal_v_hi, 35, 0, 255);
        g_calib_pending = true;
        g_calib_status = CALIB_OK;
        g_calib_status_ttl = 160;
      }
      g_calib_active = false;
    }
  }

  /* ── 3. Green mask at downscaled resolution ────────────────────────── */
  Mat green_mask_ds;
  inRange(hsv_ds,
          Scalar(cfg.hsv_h_lo, cfg.hsv_s_lo, cfg.hsv_v_lo),
          Scalar(cfg.hsv_h_hi, cfg.hsv_s_hi, cfg.hsv_v_hi),
          green_mask_ds);

  /* Morphology with cached kernel */
  Mat kern = get_morph_kern(cfg.morph_ksize);
  morphologyEx(green_mask_ds, green_mask_ds, MORPH_OPEN,  kern);
  morphologyEx(green_mask_ds, green_mask_ds, MORPH_CLOSE, kern);

  /* Remove tiny components (at ds resolution, scale threshold) */
  if (cfg.min_contour_area > 0) {
    float area_scale = sx * sy;
    double min_area_ds = (double)cfg.min_contour_area / area_scale;
    vector<vector<Point>> contours;
    findContours(green_mask_ds.clone(), contours, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    green_mask_ds.setTo(0);
    for (size_t i = 0; i < contours.size(); i++) {
      if (contourArea(contours[i]) >= min_area_ds) {
        drawContours(green_mask_ds, contours, (int)i, Scalar(255), FILLED);
      }
    }
  }

  /* Green ratio */
  int total_ds = ds_w * ds_h;
  int green_ds = countNonZero(green_mask_ds);
  float green_ratio = (float)green_ds / (float)total_ds;

  /* ── 4. Non-green mask (ds) ────────────────────────────────────────── */
  Mat non_green_ds;
  bitwise_not(green_mask_ds, non_green_ds);

  /* ── 5. Canny + Hough (already at ds resolution) ───────────────────── */
  GaussianBlur(gray_ds, gray_ds, Size(5, 5), 1.5);
  Mat edges_ds;
  Canny(gray_ds, edges_ds, canny_low, canny_high);
  bitwise_and(edges_ds, non_green_ds, edges_ds);

  if (cfg.edge_open_ksize >= 3) {
    Mat ekern = get_edge_kern(cfg.edge_open_ksize);
    morphologyEx(edges_ds, edges_ds, MORPH_OPEN, ekern);
  }

  vector<Vec4i> raw_lines;
  HoughLinesP(edges_ds, raw_lines, 1, CV_PI / 180, 30, 15, 8);

  /* Pre-scale all line endpoints to full-res ONCE */
  struct ScaledLine { float x1, y1, x2, y2, mx, my, len; };
  vector<ScaledLine> lines;
  lines.reserve(raw_lines.size());
  for (size_t i = 0; i < raw_lines.size(); i++) {
    ScaledLine sl;
    sl.x1 = raw_lines[i][0] * sx;  sl.y1 = raw_lines[i][1] * sy;
    sl.x2 = raw_lines[i][2] * sx;  sl.y2 = raw_lines[i][3] * sy;
    sl.mx = (sl.x1 + sl.x2) * 0.5f;
    sl.my = (sl.y1 + sl.y2) * 0.5f;
    sl.len = sqrtf((sl.x2 - sl.x1) * (sl.x2 - sl.x1) +
                   (sl.y2 - sl.y1) * (sl.y2 - sl.y1));
    if (sl.len >= (float)cfg.min_line_len_px) {
      lines.push_back(sl);
    }
  }

  /* ── 5b. Legacy line metrics ───────────────────────────────────────── */
  int half_h = height / 2;
  int mid_x  = width  / 2;
  float front_total = 0.f, front_left = 0.f, front_right = 0.f, longest = 0.f;
  int front_count = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    if (lines[i].my < (float)half_h) {
      front_total += lines[i].len;
      front_count++;
      if (lines[i].mx < (float)mid_x) front_left  += lines[i].len;
      else                              front_right += lines[i].len;
      if (lines[i].len > longest) longest = lines[i].len;
    }
  }
  out.front_line_total       = front_total;
  out.front_left_line_length = front_left;
  out.front_right_line_length= front_right;
  out.longest_line           = longest;
  out.line_count             = front_count;
  out.green_ratio            = green_ratio;

  /* ── 6. Grid features (at downscaled resolution) ───────────────────── */
  int ds_front_h = (ds_h * 1) / 3;
  int cell_w  = ds_w    / cfg.grid_cols;
  int cell_h  = ds_front_h / cfg.grid_rows;
  /* Full-res front_h for line mapping */
  int front_h = (height * 1) / 3;

  float max_score_cell  = 0.f;
  float mean_score_sum  = 0.f, mean_score_wsum = 0.f;
  float score_cx_sum    = 0.f, score_weight_sum = 0.f;
  float diag = sqrtf((float)(width * width + front_h * front_h));

  const int MAX_CELLS = 16;
  Rect  cell_rois_full[MAX_CELLS];
  float cell_scores[MAX_CELLS];
  int   n_cells = 0;

  for (int gr = 0; gr < cfg.grid_rows; gr++) {
    float row_weight = 1.0f + 0.5f * (float)gr;
    for (int gc = 0; gc < cfg.grid_cols; gc++) {
      /* DS-space ROI */
      int dx0 = gc * cell_w;
      int dy0 = gr * cell_h;
      int dx1 = (gc == cfg.grid_cols - 1) ? ds_w      : (gc + 1) * cell_w;
      int dy1 = (gr == cfg.grid_rows - 1) ? ds_front_h : (gr + 1) * cell_h;
      Rect ds_roi(dx0, dy0, dx1 - dx0, dy1 - dy0);
      int cell_area = ds_roi.width * ds_roi.height;
      if (cell_area <= 0) continue;

      /* Full-res ROI for overlay and line matching */
      int fx0 = (int)(dx0 * sx), fy0 = (int)(dy0 * sy);
      int fx1 = (int)(dx1 * sx), fy1 = (int)(dy1 * sy);

      float cell_green = (float)countNonZero(green_mask_ds(ds_roi)) / (float)cell_area;
      float non_green_area = 1.f - cell_green;
      float edge_density = (float)countNonZero(edges_ds(ds_roi)) / (float)cell_area;

      /* Max line length in this cell (using pre-scaled endpoints) */
      float max_ll = 0.f;
      for (size_t li = 0; li < lines.size(); li++) {
        if (lines[li].mx >= fx0 && lines[li].mx < fx1 &&
            lines[li].my >= fy0 && lines[li].my < fy1) {
          if (lines[li].len > max_ll) max_ll = lines[li].len;
        }
      }
      float max_line_norm = clampf(max_ll / diag, 0.f, 1.f);

      /* Gate edge/line evidence on non-green fraction */
      float ev_scale = 1.f;
      if (non_green_area < cfg.min_non_green_for_edges && cfg.min_non_green_for_edges > 1e-6f)
        ev_scale = non_green_area / cfg.min_non_green_for_edges;
      ev_scale = clampf(ev_scale, 0.f, 1.f);
      edge_density  *= ev_scale;
      max_line_norm *= ev_scale;

      float sc = clampf(cfg.w_ng * non_green_area +
                         cfg.w_ed * edge_density +
                         cfg.w_ll * max_line_norm, 0.f, 1.f);

      if (n_cells < MAX_CELLS) {
        cell_rois_full[n_cells] = Rect(fx0, fy0, fx1 - fx0, fy1 - fy0);
        cell_scores[n_cells] = sc;
        n_cells++;
      }
      if (sc > max_score_cell) max_score_cell = sc;
      mean_score_sum += sc * row_weight;
      mean_score_wsum += row_weight;
      float cell_cx = ((float)(fx0 + fx1) * 0.5f / (float)width) * 2.f - 1.f;
      score_cx_sum    += sc * row_weight * cell_cx;
      score_weight_sum += sc * row_weight;
    }
  }

  /* ── 7. Aggregate global score & centroid ──────────────────────────── */
  float mean_score = (mean_score_wsum > 1e-6f) ? (mean_score_sum / mean_score_wsum) : 0.f;
  float raw_score = clampf(0.65f * max_score_cell + 0.35f * mean_score, 0.f, 1.f);
  float raw_cx    = (score_weight_sum > 1e-6f) ? (score_cx_sum / score_weight_sum) : 0.f;
  raw_cx = clampf(raw_cx, -1.f, 1.f);

  /* Non-green area fraction in front half (ds) */
  Rect ds_front_roi(0, 0, ds_w, ds_front_h);
  int ng_front = ds_front_roi.area() - countNonZero(green_mask_ds(ds_front_roi));
  float obstacle_size = (float)ng_front / (float)ds_front_roi.area();

  /* ── 8. EMA smoothing ─────────────────────────────────────────────── */
  float alpha = cfg.ema_alpha;
  if (!ema_init) {
    ema_score    = raw_score;
    ema_centroid = raw_cx;
    ema_init     = true;
  } else {
    ema_score    = alpha * raw_score + (1.f - alpha) * ema_score;
    ema_centroid = alpha * raw_cx    + (1.f - alpha) * ema_centroid;
  }
  out.obstacle_score      = ema_score;
  out.obstacle_centroid_x = ema_centroid;
  out.obstacle_size       = obstacle_size;

  /* ── 9. Debug overlay (skipped on AP builds for speed) ───────────────── */
#ifndef EDGE_DETECTOR_NO_OVERLAY
  {
  /* Upscale masks to full-res only for drawing */
  Mat green_mask, non_green, edges_full;
  resize(green_mask_ds, green_mask, Size(width, height), 0, 0, INTER_NEAREST);
  resize(non_green_ds,  non_green,  Size(width, height), 0, 0, INTER_NEAREST);
  resize(edges_ds,      edges_full, Size(width, height), 0, 0, INTER_NEAREST);

  Mat debug_bgr;
  debug_bgr = bgr_full * 0.4;

  /* (b) Bright green tint on detected floor */
  for (int r = 0; r < height; r++) {
    const uchar *pm = green_mask.ptr<uchar>(r);
    const uchar *po = bgr_full.ptr<uchar>(r);
    uchar *pd = debug_bgr.ptr<uchar>(r);
    for (int c = 0; c < width; c++) {
      if (pm[c]) {
        pd[3*c+0] = (uchar)(po[3*c+0] * 0.3f);
        pd[3*c+1] = (uchar)min(255, (int)(po[3*c+1] * 0.5f + 128));
        pd[3*c+2] = (uchar)(po[3*c+2] * 0.3f);
      }
    }
  }

  /* (c) Red/magenta tint on non-green areas in front half */
  for (int r = 0; r < front_h; r++) {
    const uchar *pm = non_green.ptr<uchar>(r);
    const uchar *po = bgr_full.ptr<uchar>(r);
    uchar *pd = debug_bgr.ptr<uchar>(r);
    for (int c = 0; c < width; c++) {
      if (pm[c]) {
        pd[3*c+0] = (uchar)(po[3*c+0] * 0.3f);
        pd[3*c+1] = (uchar)(po[3*c+1] * 0.2f);
        pd[3*c+2] = (uchar)min(255, (int)(po[3*c+2] * 0.4f + 100));
      }
    }
  }

  /* (d) Non-green contour outlines in magenta */
  {
    vector<vector<Point>> ng_contours;
    findContours(non_green.clone(), ng_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(debug_bgr, ng_contours, -1, Scalar(255, 0, 255), 1);
  }

  /* (e) Canny edges in red */
  for (int r = 0; r < height; r++) {
    const uchar *pe = edges_full.ptr<uchar>(r);
    uchar *pd = debug_bgr.ptr<uchar>(r);
    for (int c = 0; c < width; c++) {
      if (pe[c]) {
        pd[3*c+0] = 30;
        pd[3*c+1] = 30;
        pd[3*c+2] = 255;
      }
    }
  }

  /* (f) Hough line segments in cyan */
  for (size_t li = 0; li < lines.size(); li++) {
    Point p1((int)lines[li].x1, (int)lines[li].y1);
    Point p2((int)lines[li].x2, (int)lines[li].y2);
    cv::line(debug_bgr, p1, p2, Scalar(255, 255, 0), 1, LINE_AA);
  }

  /* (g) Grid overlay – cells coloured by score */
  for (int ci = 0; ci < n_cells; ci++) {
    Rect roi = cell_rois_full[ci];
    float sc = cell_scores[ci];
    int rb = (int)(sc * 255.f);
    int gb = (int)((1.f - sc) * 255.f);
    Scalar cell_color(0, gb, rb);
    Mat cell_region = debug_bgr(roi);
    Mat tint(roi.height, roi.width, CV_8UC3, cell_color);
    addWeighted(cell_region, 0.75, tint, 0.25, 0, cell_region);
    rectangle(debug_bgr, roi, Scalar(200, 200, 200), 1);
    char sc_txt[16];
    snprintf(sc_txt, sizeof(sc_txt), "%.2f", sc);
    putText(debug_bgr, sc_txt, Point(roi.x + 2, roi.y + 12),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 2);
    putText(debug_bgr, sc_txt, Point(roi.x + 2, roi.y + 12),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 1);
  }

  /* (h) Obstacle centroid crosshair */
  int cx_px = (int)(((ema_centroid + 1.f) * 0.5f) * (float)width);
  cx_px = max(0, min(cx_px, width - 1));
  int cy_px = front_h / 2;
  cv::line(debug_bgr, Point(cx_px - 15, cy_px), Point(cx_px + 15, cy_px), Scalar(0, 255, 255), 2);
  cv::line(debug_bgr, Point(cx_px, cy_px - 15), Point(cx_px, cy_px + 15), Scalar(0, 255, 255), 2);
  circle(debug_bgr, Point(cx_px, cy_px), 12, Scalar(0, 255, 255), 2);

  /* (i) Front-half boundary */
  for (int c = 0; c < width; c += 8) {
    int end_c = min(c + 4, width);
    cv::line(debug_bgr, Point(c, front_h), Point(end_c, front_h), Scalar(255, 255, 255), 1);
  }

  /* (j) Text HUD */
  char txt[128];
  Scalar hud_fg(255, 255, 255), hud_bg(0, 0, 0);
  int y_txt = height - 8;
  snprintf(txt, sizeof(txt), "SCORE %.2f  CX %.2f  SIZE %.2f", ema_score, ema_centroid, obstacle_size);
  putText(debug_bgr, txt, Point(4, y_txt), FONT_HERSHEY_SIMPLEX, 0.38, hud_bg, 2);
  putText(debug_bgr, txt, Point(4, y_txt), FONT_HERSHEY_SIMPLEX, 0.38, hud_fg, 1);
  y_txt -= 14;
  snprintf(txt, sizeof(txt), "GREEN %.0f%%  LINES %d  LONGEST %.0f", green_ratio * 100.f, front_count, longest);
  putText(debug_bgr, txt, Point(4, y_txt), FONT_HERSHEY_SIMPLEX, 0.38, hud_bg, 2);
  putText(debug_bgr, txt, Point(4, y_txt), FONT_HERSHEY_SIMPLEX, 0.38, hud_fg, 1);

  /* Danger/safe indicator */
  {
    const char *status = (ema_score > cfg.score_threshold) ? "DANGER" : "SAFE";
    Scalar scolor = (ema_score > cfg.score_threshold) ? Scalar(0, 0, 255) : Scalar(0, 200, 0);
    putText(debug_bgr, status, Point(width - 75, 18), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0), 3);
    putText(debug_bgr, status, Point(width - 75, 18), FONT_HERSHEY_SIMPLEX, 0.55, scolor, 2);
  }

  /* Ground-calibration status */
  {
    char ctxt[96];
    Scalar ccol(255, 255, 255);
    if (g_calib_active) {
      snprintf(ctxt, sizeof(ctxt), "CALIBRATING FLOOR %d/%d", g_calib_frames, g_calib_target_frames);
      ccol = Scalar(0, 255, 255);
    } else if (g_calib_status == CALIB_OK && g_calib_status_ttl > 0) {
      printf(ctxt, sizeof(ctxt), "CALIBRATED H[%d..%d] S[%d..%d] V[%d..%d]",
               g_cal_h_lo, g_cal_h_hi, g_cal_s_lo, g_cal_s_hi, g_cal_v_lo, g_cal_v_hi);
      ccol = Scalar(0, 220, 0);
      g_calib_status_ttl--;
    } else if (g_calib_status == CALIB_FAIL && g_calib_status_ttl > 0) {
      printf(ctxt, sizeof(ctxt), "CALIBRATION FAILED - RETRY");
      ccol = Scalar(0, 0, 255);
      g_calib_status_ttl--;
    } else {
      ctxt[0] = '\0';
    }
    if (ctxt[0] != '\0') {
      putText(debug_bgr, ctxt, Point(4, 18), FONT_HERSHEY_SIMPLEX, 0.40, Scalar(0, 0, 0), 3);
      putText(debug_bgr, ctxt, Point(4, 18), FONT_HERSHEY_SIMPLEX, 0.40, ccol, 1);
    }
  }

  colorbgr_opencv_to_yuv422(debug_bgr, img, width, height);
  } /* end overlay block */
#endif /* EDGE_DETECTOR_NO_OVERLAY */

  return out;
}
