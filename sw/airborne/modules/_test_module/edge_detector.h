/*
 * @file "modules/_test_module/edge_detector.h"
 * Combined green-floor segmentation + edge/line obstacle detector.
 * Keeps the legacy edge_result API and adds obstacle_result for the
 * new fused pipeline.
 */

#ifndef EDGE_DETECTOR_H
#define EDGE_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── Legacy result (kept for backwards compatibility) ───────────────────── */
struct edge_result {
  float front_line_length;       // total line length (px) in front half
  float front_left_line_length;  // total line length (px) in front-left
  float front_right_line_length; // total line length (px) in front-right
  float longest_line;            // length of the single longest segment
  int   line_count;              // number of line segments found in front
};

struct edge_result detect_edges(char *img, int width, int height,
                                int canny_low, int canny_high);

/* ── Configuration for the fused obstacle detector ──────────────────────── */
struct ObstacleConfig {
  /* HSV green bounds */
  int hsv_h_lo, hsv_h_hi;
  int hsv_s_lo, hsv_s_hi;
  int hsv_v_lo, hsv_v_hi;

  /* Grid layout (cols × rows in the front half) */
  int grid_cols, grid_rows;

  /* Per-cell score weights */
  float w_ng;   // non-green area weight
  float w_ed;   // edge density weight
  float w_ll;   // max line-length weight

  /* EMA smoothing factor (0..1, smaller = smoother) */
  float ema_alpha;

  /* Score threshold for "obstacle found" flag */
  float score_threshold;

  /* Downscale resolution for Canny/Hough (width, height) */
  int ds_width, ds_height;

  /* Morphology kernel size */
  int morph_ksize;

  /* Minimum contour area to keep (pixels in full-res) */
  int min_contour_area;

  /* Morphological cleanup for Canny edges (odd kernel size, 0 disables) */
  int edge_open_ksize;

  /* Minimum Hough segment length in full-resolution pixels */
  int min_line_len_px;

  /* If non-green is below this, edge/line evidence is down-weighted */
  float min_non_green_for_edges;
};

/* ── New fused result ───────────────────────────────────────────────────── */
struct obstacle_result {
  /* New fused outputs */
  float obstacle_score;       
  float obstacle_centroid_x;  
  float obstacle_size;     
  float green_ratio;        

  /* Legacy-compatible outputs */
  float front_line_total;
  float front_left_line_length;
  float front_right_line_length;
  float longest_line;
  int   line_count;
};

/*
 * Run the full fused pipeline on a YUV422 buffer.
 * Writes a debug overlay back into img for RTP visualisation.
 */
struct obstacle_result detect_obstacles(char *img, int width, int height,
                                        int canny_low, int canny_high,
                                        struct ObstacleConfig cfg);

/*
 * Start floor-colour calibration from the current bottom-camera view.
 * Calibration runs over multiple frames inside detect_obstacles().
 */
void obstacle_start_ground_calibration(void);

/*
 * Fetch the latest calibrated HSV bounds once.
 * Returns 1 when new bounds are available, 0 otherwise.
 */
int obstacle_consume_calibrated_hsv(int *h_lo, int *h_hi,
                                    int *s_lo, int *s_hi,
                                    int *v_lo, int *v_hi);

/*
 * Return a config struct filled with sensible defaults.
 */
struct ObstacleConfig obstacle_config_defaults(void);

#ifdef __cplusplus
}
#endif

#endif
