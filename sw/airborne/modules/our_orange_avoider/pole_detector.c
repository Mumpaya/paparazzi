// /*
//  * @file "modules/our_orange_avoider/pole_detector.c"
//  * Detects orange poles via YUV color thresholding on the front camera.
//  * Translated directly from Python OrangePoleDetector prototype.
//  *
//  * ABI message fields used:
//  *   pixel_x  = centroid x of largest orange blob (-1 if none found)
//  *   pixel_y  = centroid y of largest orange blob (-1 if none found)
//  *   quality  = total orange pixel count
//  *   extra    = 1 if pole detected, 0 if not
//  */

// #include "modules/our_orange_avoider/pole_detector.h"
// #include "modules/computer_vision/cv.h"
// #include "modules/core/abi.h"
// #include "generated/airframe.h"
// #include "modules/computer_vision/lib/vision/image.h"
// #include <stdio.h>
// #include <stdint.h>
// #include <string.h>

// #define POLE_DETECTOR_VERBOSE TRUE
// #define PRINT(string,...) fprintf(stderr, "[pole_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
// #if POLE_DETECTOR_VERBOSE
// #define VERBOSE_PRINT PRINT
// #else
// #define VERBOSE_PRINT(...)
// #endif

// // ============================================================
// // COLOR THRESHOLDS (YUV422)
// // SIMULATION (matches Python HSV (5,120,80)-(30,255,255)):
// #ifndef POLE_Y_MIN
// #define POLE_Y_MIN  41
// #endif
// #ifndef POLE_Y_MAX
// #define POLE_Y_MAX 183
// #endif
// #ifndef POLE_U_MIN
// #define POLE_U_MIN  53
// #endif
// #ifndef POLE_U_MAX
// #define POLE_U_MAX 121
// #endif
// #ifndef POLE_V_MIN
// #define POLE_V_MIN 134
// #endif
// #ifndef POLE_V_MAX
// #define POLE_V_MAX 249
// #endif
// // CYBERZOO — override in airframe .xml:
// // POLE_Y_MIN=30, POLE_Y_MAX=220
// // POLE_U_MIN=60, POLE_U_MAX=130
// // POLE_V_MIN=125, POLE_V_MAX=255
// // ============================================================

// #ifndef POLE_MIN_PIXEL_COUNT
// #define POLE_MIN_PIXEL_COUNT 120
// #endif

// #ifndef POLE_DETECTOR_CAMERA
// #define POLE_DETECTOR_CAMERA front_camera
// #endif

// #ifndef POLE_DETECTOR_FPS
// #define POLE_DETECTOR_FPS 0
// #endif

// #ifndef POLE_DETECTOR_ABI_ID
// #define POLE_DETECTOR_ABI_ID 1
// #endif

// // Grid: each cell must have enough orange pixels to "survive" morphological open
// // Equivalent to Python morph_kernel_size=(5,5) open+close
// #define NUM_COLS 40
// #define NUM_ROWS 40
// // A cell is "orange" only if it has at least this many orange pixels
// // This approximates morphological opening (removes isolated noise cells)
// #define CELL_MIN_PIXELS 1

// // Python: min_aspect_ratio=2.5, min_confidence=0.3
// // NOTE: aspect ratio is on the DISPLAY image which is rotated.
// // In the RAW buffer (before counterclockwise flip), width and height are swapped.
// // So a tall pole in display = wide blob in raw buffer → w/h > 2.5 in raw
// #define MIN_ASPECT_RATIO_x10  25   // w/h > 2.5 in raw buffer
// #define MIN_CONFIDENCE_x100   30   // 30% fill

// // Grid flood fill uses a small stack
// #define STACK_SIZE 2048

// // ── Shared result ─────────────────────────────────────────────────────────────
// static int16_t pole_cx    = -1;
// static int16_t pole_cy    = -1;
// static int32_t pole_count =  0;
// static int16_t pole_found =  0;

// // UYVY draw colors: {U, Y, V, Y}
// static uint8_t green_color[4] = {0,   255, 0,   255};
// static uint8_t red_color[4]   = {255, 76,  255, 76};

// // ── Vision callback ───────────────────────────────────────────────────────────
// static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
// {
//   if (img->type != IMAGE_YUV422) return NULL;

//   uint8_t  *buf    = img->buf;
//   uint32_t  width  = img->w;
//   uint32_t  height = img->h;

//   // ── Step 1: build binary orange mask on the grid ──────────────────────────
//   // grid_count[r][c] = number of orange pixels in that cell
//   static uint32_t grid_count[NUM_ROWS][NUM_COLS];
//   static uint8_t  grid_mask[NUM_ROWS][NUM_COLS];   // 1=orange cell, 0=not
//   static uint8_t  grid_visited[NUM_ROWS][NUM_COLS];
//   memset(grid_count,   0, sizeof(grid_count));

//   int64_t sum_x = 0, sum_y = 0;
//   int32_t total_orange = 0;

//   for (uint32_t y = 0; y < height; y++) {
//     uint32_t row = (y * NUM_ROWS) / height;
//     for (uint32_t x = 0; x < width; x += 2) {
//       uint32_t idx = (y * width + x) * 2;
//       uint8_t u    = buf[idx];
//       uint8_t lum1 = buf[idx + 1];
//       uint8_t v    = buf[idx + 2];
//       uint8_t lum2 = buf[idx + 3];

//       if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
//           u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
//           v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
//         grid_count[row][(x * NUM_COLS) / width]++;
//         sum_x += x; sum_y += y; total_orange++;
//       }
//       if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
//           u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
//           v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
//         grid_count[row][((x + 1) * NUM_COLS) / width]++;
//         sum_x += (x + 1); sum_y += y; total_orange++;
//       }
//     }
//   }

//   if (total_orange < POLE_MIN_PIXEL_COUNT) {
//     pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
//     return img;
//   }

//   // ── Step 2: threshold grid cells (approximates morphological open) ────────
//   for (uint32_t r = 0; r < NUM_ROWS; r++)
//     for (uint32_t c = 0; c < NUM_COLS; c++)
//       grid_mask[r][c] = (grid_count[r][c] >= CELL_MIN_PIXELS) ? 1 : 0;

//   // ── Step 3: find largest connected blob via BFS on grid ───────────────────
//   memset(grid_visited, 0, sizeof(grid_visited));

//   // BFS stack: encode (row, col) as row*NUM_COLS+col
//   static uint16_t stack[STACK_SIZE];

//   uint32_t best_blob_size  = 0;
//   uint32_t best_c_min = 0, best_c_max = 0;
//   uint32_t best_r_min = 0, best_r_max = 0;
//   uint32_t best_orange_cells = 0;

//   for (uint32_t seed_r = 0; seed_r < NUM_ROWS; seed_r++) {
//     for (uint32_t seed_c = 0; seed_c < NUM_COLS; seed_c++) {
//       if (!grid_mask[seed_r][seed_c] || grid_visited[seed_r][seed_c]) continue;

//       // BFS from this seed
//       uint32_t sp = 0;
//       stack[sp++] = (uint16_t)(seed_r * NUM_COLS + seed_c);
//       grid_visited[seed_r][seed_c] = 1;

//       uint32_t blob_size    = 0;
//       uint32_t c_min = NUM_COLS, c_max = 0;
//       uint32_t r_min = NUM_ROWS, r_max = 0;

//       while (sp > 0 && sp < STACK_SIZE) {
//         uint16_t node = stack[--sp];
//         uint32_t r = node / NUM_COLS;
//         uint32_t c = node % NUM_COLS;

//         blob_size++;
//         if (c < c_min) c_min = c;
//         if (c > c_max) c_max = c;
//         if (r < r_min) r_min = r;
//         if (r > r_max) r_max = r;

//         // 4-connected neighbours
//         const int dr[4] = {-1, 1,  0, 0};
//         const int dc[4] = { 0, 0, -1, 1};
//         for (int d = 0; d < 4; d++) {
//           int nr = (int)r + dr[d];
//           int nc = (int)c + dc[d];
//           if (nr < 0 || nr >= NUM_ROWS || nc < 0 || nc >= NUM_COLS) continue;
//           if (!grid_mask[nr][nc] || grid_visited[nr][nc]) continue;
//           grid_visited[nr][nc] = 1;
//           stack[sp++] = (uint16_t)(nr * NUM_COLS + nc);
//         }
//       }

//       if (blob_size > best_blob_size) {
//         best_blob_size    = blob_size;
//         best_c_min = c_min; best_c_max = c_max;
//         best_r_min = r_min; best_r_max = r_max;
//         best_orange_cells = blob_size;
//       }
//     }
//   }

//   if (best_blob_size == 0) {
//     pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
//     return img;
//   }

//   // ── Step 4: compute bounding box in pixels ────────────────────────────────
//   uint32_t box_x_min = (best_c_min        * width)  / NUM_COLS;
//   uint32_t box_x_max = ((best_c_max + 1)  * width)  / NUM_COLS;
//   uint32_t box_y_min = (best_r_min        * height) / NUM_ROWS;
//   uint32_t box_y_max = ((best_r_max + 1)  * height) / NUM_ROWS;

//   uint32_t box_w = box_x_max - box_x_min;
//   uint32_t box_h = box_y_max - box_y_min;

//   // If >40% of the central strip is orange, we're too close: just report centroid
//   uint32_t center_c_min = NUM_COLS / 4;
//   uint32_t center_c_max = 3 * NUM_COLS / 4;
//   uint32_t center_cells = 0, center_orange = 0;
//   for (uint32_t r = 0; r < NUM_ROWS; r++)
//     for (uint32_t c = center_c_min; c < center_c_max; c++) {
//       center_cells++;
//       if (grid_mask[r][c]) center_orange++;
//     }

//   if (center_cells > 0 && center_orange * 100 > center_cells * 40) {
//     // Pole is very close — skip shape filters, just report it
//     pole_cx    = (int16_t)(sum_x / total_orange);
//     pole_cy    = (int16_t)(sum_y / total_orange);
//     pole_count = total_orange;
//     pole_found = 1;
//     image_draw_rectangle(img, (int)box_x_min, (int)box_x_max,
//                               (int)box_y_min, (int)box_y_max, green_color);
//     struct point_t centroid = {.x = (uint32_t)pole_cx, .y = (uint32_t)pole_cy};
//     image_draw_crosshair(img, &centroid, red_color, 10);
//     VERBOSE_PRINT("close pole override: center_orange=%d/%d\n",
//                   (int)center_orange, (int)center_cells);
//     return img;
//   }


//   // ── Step 5: aspect ratio filter ───────────────────────────────────────────
//   // Raw buffer is rotated CCW vs display. Pole is tall in display = wide in raw.
//   // So check w/h > 2.5 (i.e. w*10 > h*25)
//   if (box_h == 0 || box_w * 10 < box_h * MIN_ASPECT_RATIO_x10) {
//     VERBOSE_PRINT("rejected aspect: w=%d h=%d\n", (int)box_w, (int)box_h);
//     pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
//     return img;
//   }

//   // ── Step 6: confidence filter — orange cells / total box cells ────────────
//   uint32_t box_total_cells = (best_r_max - best_r_min + 1) *
//                              (best_c_max - best_c_min + 1);
//   if (box_total_cells == 0 ||
//       best_orange_cells * 100 < box_total_cells * MIN_CONFIDENCE_x100) {
//     VERBOSE_PRINT("rejected confidence: %d/%d cells\n",
//                   (int)best_orange_cells, (int)box_total_cells);
//     pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
//     return img;
//   }

//   // ── All filters passed ────────────────────────────────────────────────────
//   pole_cx    = (int16_t)(sum_x / total_orange);
//   pole_cy    = (int16_t)(sum_y / total_orange);
//   pole_count = total_orange;
//   pole_found = 1;

//   image_draw_rectangle(img, (int)box_x_min, (int)box_x_max,
//                             (int)box_y_min, (int)box_y_max, green_color);

//   struct point_t centroid;
//   centroid.x = (uint32_t)pole_cx;
//   centroid.y = (uint32_t)pole_cy;
//   image_draw_crosshair(img, &centroid, red_color, 10);

//   VERBOSE_PRINT("found: cx=%d cy=%d count=%d box=(%d,%d)-(%d,%d) w=%d h=%d\n",
//                 pole_cx, pole_cy, pole_count,
//                 (int)box_x_min, (int)box_y_min, (int)box_x_max, (int)box_y_max,
//                 (int)box_w, (int)box_h);

//   return img;
// }

// // ── Module init ───────────────────────────────────────────────────────────────
// void pole_detector_init(void)
// {
//   cv_add_to_device(&POLE_DETECTOR_CAMERA, detect_orange, POLE_DETECTOR_FPS, 0);
//   VERBOSE_PRINT("Pole detector initialised.\n");
// }

// // ── Module periodic — publishes ABI message ───────────────────────────────────
// void pole_detector_periodic(void)
// {
//   AbiSendMsgVISUAL_DETECTION(
//     POLE_DETECTOR_ABI_ID,
//     pole_cx,
//     pole_cy,
//     0,
//     0,
//     pole_count,
//     pole_found
//   );

//   VERBOSE_PRINT("periodic: found=%d cx=%d cy=%d count=%d\n",
//                 pole_found, pole_cx, pole_cy, pole_count);
// }

/*
 * @file "modules/our_orange_avoider/pole_detector.c"
 * Detects orange poles via YUV color thresholding on the front camera.
 * Finds all blobs, filters by shape, reports the largest non-marginal one.
 *
 * ABI message fields used:
 *   pixel_x  = centroid x of best pole (-1 if none found)
 *   pixel_y  = centroid y of best pole (-1 if none found)
 *   quality  = total orange pixel count
 *   extra    = 1 if pole detected, 0 if not
 */

#include "modules/our_orange_avoider/pole_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "generated/airframe.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define POLE_DETECTOR_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[pole_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if POLE_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// ============================================================
// COLOR THRESHOLDS (YUV422)
// SIMULATION (matches Python HSV (5,120,80)-(30,255,255)):
#ifndef POLE_Y_MIN
#define POLE_Y_MIN  41
#endif
#ifndef POLE_Y_MAX
#define POLE_Y_MAX 183
#endif
#ifndef POLE_U_MIN
#define POLE_U_MIN  53
#endif
#ifndef POLE_U_MAX
#define POLE_U_MAX 121
#endif
#ifndef POLE_V_MIN
#define POLE_V_MIN 134
#endif
#ifndef POLE_V_MAX
#define POLE_V_MAX 249
#endif
// CYBERZOO override in airframe .xml:
// POLE_Y_MIN=30, POLE_Y_MAX=220
// POLE_U_MIN=60, POLE_U_MAX=130
// POLE_V_MIN=125, POLE_V_MAX=255
// ============================================================

#ifndef POLE_MIN_PIXEL_COUNT
#define POLE_MIN_PIXEL_COUNT 150
#endif

#ifndef POLE_DETECTOR_CAMERA
#define POLE_DETECTOR_CAMERA front_camera
#endif

#ifndef POLE_DETECTOR_FPS
#define POLE_DETECTOR_FPS 0
#endif

#ifndef POLE_DETECTOR_ABI_ID
#define POLE_DETECTOR_ABI_ID 1
#endif

#define NUM_COLS 80
#define NUM_ROWS 80

// A grid cell needs at least this many orange pixels to count
#define CELL_MIN_PIXELS 1

// Blob shape filters
// Raw buffer is rotated CCW vs display — pole tall in display = wide in raw
#define MIN_ASPECT_RATIO_x10  20
#define MIN_CONFIDENCE_x100   25   

// Margin: ignore blobs whose bounding box touches within this many grid cells
// of the image edge — they are likely partial poles
#define MARGIN_CELLS 2

// Close-pole threshold: if >40% of central strip is orange, pole is too close
#define CLOSE_POLE_CENTER_PCT 40

#define STACK_SIZE 4096
#define MAX_BLOBS  16

// ── Shared result ─────────────────────────────────────────────────────────────
static int16_t pole_cx    = -1;
static int16_t pole_cy    = -1;
static int32_t pole_count =  0;
static int16_t pole_found =  0;

// UYVY draw colors: {U, Y, V, Y}
static uint8_t green_color[4] = {0,   255, 0,   255};
static uint8_t red_color[4]   = {255, 76,  255, 76};
static uint8_t yellow_color[4]= {128, 210, 128, 210}; // discarded blobs

// ── Blob descriptor ───────────────────────────────────────────────────────────
typedef struct {
  uint32_t c_min, c_max, r_min, r_max;  // grid coords
  uint32_t size;                         // number of orange cells in blob
} Blob;

// ── Vision callback ───────────────────────────────────────────────────────────
static struct image_t *detect_orange(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (img->type != IMAGE_YUV422) return NULL;

  uint8_t  *buf    = img->buf;
  uint32_t  width  = img->w;
  uint32_t  height = img->h;

  static uint32_t grid_count[NUM_ROWS][NUM_COLS];
  static uint8_t  grid_mask[NUM_ROWS][NUM_COLS];
  static uint8_t  grid_visited[NUM_ROWS][NUM_COLS];
  memset(grid_count,   0, sizeof(grid_count));
  memset(grid_visited, 0, sizeof(grid_visited));

  int64_t sum_x_all = 0, sum_y_all = 0;
  int32_t total_orange = 0;

  // ── Pass 1: classify pixels, fill grid ───────────────────────────────────
  for (uint32_t y = 0; y < height; y++) {
    uint32_t row = (y * NUM_ROWS) / height;
    for (uint32_t x = 0; x < width; x += 2) {
      uint32_t idx = (y * width + x) * 2;
      uint8_t u    = buf[idx];
      uint8_t lum1 = buf[idx + 1];
      uint8_t v    = buf[idx + 2];
      uint8_t lum2 = buf[idx + 3];

      if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        grid_count[row][(x * NUM_COLS) / width]++;
        sum_x_all += x; sum_y_all += y; total_orange++;
      }
      if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        grid_count[row][((x + 1) * NUM_COLS) / width]++;
        sum_x_all += (x + 1); sum_y_all += y; total_orange++;
      }
    }
  }

  if (total_orange < POLE_MIN_PIXEL_COUNT) {
    pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
    return img;
  }

  // ── Step 2: threshold cells ───────────────────────────────────────────────
  for (uint32_t r = 0; r < NUM_ROWS; r++)
    for (uint32_t c = 0; c < NUM_COLS; c++)
      grid_mask[r][c] = (grid_count[r][c] >= CELL_MIN_PIXELS) ? 1 : 0;

  // ── Step 3: close-pole override ───────────────────────────────────────────
  // If the central 50% of columns has >40% orange cells, pole fills frame
  uint32_t center_c_min = NUM_COLS / 4;
  uint32_t center_c_max = 3 * NUM_COLS / 4;
  uint32_t center_total = 0, center_orange = 0;
  for (uint32_t r = 0; r < NUM_ROWS; r++)
    for (uint32_t c = center_c_min; c < center_c_max; c++) {
      center_total++;
      if (grid_mask[r][c]) center_orange++;
    }

  if (center_total > 0 && center_orange * 100 > center_total * CLOSE_POLE_CENTER_PCT) {
    pole_cx    = (int16_t)(sum_x_all / total_orange);
    pole_cy    = (int16_t)(sum_y_all / total_orange);
    pole_count = total_orange;
    pole_found = 1;
    // Draw full-width box to indicate close pole
    image_draw_rectangle(img, 0, (int)width, 0, (int)height, green_color);
    struct point_t centroid = {.x = (uint32_t)pole_cx, .y = (uint32_t)pole_cy};
    image_draw_crosshair(img, &centroid, red_color, 10);
    VERBOSE_PRINT("close pole override: %d/%d center cells orange\n",
                  (int)center_orange, (int)center_total);
    return img;
  }

  // ── Step 4: BFS to find all blobs ─────────────────────────────────────────
  static uint16_t stack[STACK_SIZE];
  Blob blobs[MAX_BLOBS];
  uint32_t num_blobs = 0;

  const int dr[4] = {-1, 1,  0, 0};
  const int dc[4] = { 0, 0, -1, 1};

  for (uint32_t seed_r = 0; seed_r < NUM_ROWS; seed_r++) {
    for (uint32_t seed_c = 0; seed_c < NUM_COLS; seed_c++) {
      if (!grid_mask[seed_r][seed_c] || grid_visited[seed_r][seed_c]) continue;
      if (num_blobs >= MAX_BLOBS) break;

      uint32_t sp = 0;
      stack[sp++] = (uint16_t)(seed_r * NUM_COLS + seed_c);
      grid_visited[seed_r][seed_c] = 1;

      uint32_t blob_size = 0;
      uint32_t c_min = NUM_COLS, c_max = 0;
      uint32_t r_min = NUM_ROWS, r_max = 0;

      while (sp > 0 && sp < STACK_SIZE) {
        uint16_t node = stack[--sp];
        uint32_t r = node / NUM_COLS;
        uint32_t c = node % NUM_COLS;

        blob_size++;
        if (c < c_min) c_min = c;
        if (c > c_max) c_max = c;
        if (r < r_min) r_min = r;
        if (r > r_max) r_max = r;

        for (int d = 0; d < 4; d++) {
          int nr = (int)r + dr[d];
          int nc = (int)c + dc[d];
          if (nr < 0 || nr >= (int)NUM_ROWS || nc < 0 || nc >= (int)NUM_COLS) continue;
          if (!grid_mask[nr][nc] || grid_visited[nr][nc]) continue;
          grid_visited[nr][nc] = 1;
          stack[sp++] = (uint16_t)(nr * NUM_COLS + nc);
        }
      }

      blobs[num_blobs].c_min = c_min;
      blobs[num_blobs].c_max = c_max;
      blobs[num_blobs].r_min = r_min;
      blobs[num_blobs].r_max = r_max;
      blobs[num_blobs].size  = blob_size;
      num_blobs++;
    }
  }

  // ── Step 5: filter blobs, pick largest non-marginal valid one ────────────
  Blob *best = NULL;

  for (uint32_t b = 0; b < num_blobs; b++) {
    Blob *bl = &blobs[b];

    uint32_t box_x_min = (bl->c_min        * width)  / NUM_COLS;
    uint32_t box_x_max = ((bl->c_max + 1)  * width)  / NUM_COLS;
    uint32_t box_y_min = (bl->r_min        * height) / NUM_ROWS;
    uint32_t box_y_max = ((bl->r_max + 1)  * height) / NUM_ROWS;
    uint32_t box_w = box_x_max - box_x_min;
    uint32_t box_h = box_y_max - box_y_min;

    // Draw all blobs in yellow for debug
    image_draw_rectangle(img, (int)box_x_min, (int)box_x_max,
                              (int)box_y_min,  (int)box_y_max, yellow_color);

    // Filter 1: margin — skip blobs touching image edge
    // if (bl->c_min < MARGIN_CELLS || bl->c_max > NUM_COLS - 1 - MARGIN_CELLS ||
    //     bl->r_min < MARGIN_CELLS || bl->r_max > NUM_ROWS - 1 - MARGIN_CELLS) {
    //   VERBOSE_PRINT("blob %d rejected: margin\n", (int)b);
    //   continue;
    // }

    // Filter 2: minimum size
    if (bl->size < 8) {
      VERBOSE_PRINT("blob %d rejected: too small (%d cells)\n", (int)b, (int)bl->size);
      continue;
    }

    // Filter 3: aspect ratio — w/h > 1.5 in raw buffer (pole tall in display)
    if (box_h == 0 || box_w * 10 < box_h * MIN_ASPECT_RATIO_x10) {
      VERBOSE_PRINT("blob %d rejected: aspect w=%d h=%d\n", (int)b, (int)box_w, (int)box_h);
      continue;
    }

    // Filter 4: confidence — orange cells / box cells > 0.25
    uint32_t box_total_cells  = (bl->r_max - bl->r_min + 1) *
                                (bl->c_max - bl->c_min + 1);
    if (box_total_cells == 0 ||
        bl->size * 100 < box_total_cells * MIN_CONFIDENCE_x100) {
      VERBOSE_PRINT("blob %d rejected: confidence %d/%d\n",
                    (int)b, (int)bl->size, (int)box_total_cells);
      continue;
    }

    // Pick largest passing blob
    if (best == NULL || bl->size > best->size) {
      best = bl;
    }
  }

  if (best == NULL) {
    pole_cx = -1; pole_cy = -1; pole_count = total_orange; pole_found = 0;
    return img;
  }

  // ── Step 6: report best blob ──────────────────────────────────────────────
  uint32_t box_x_min = (best->c_min        * width)  / NUM_COLS;
  uint32_t box_x_max = ((best->c_max + 1)  * width)  / NUM_COLS;
  uint32_t box_y_min = (best->r_min        * height) / NUM_ROWS;
  uint32_t box_y_max = ((best->r_max + 1)  * height) / NUM_ROWS;

  // Centroid from best blob's grid region
  int64_t sum_x = 0, sum_y = 0;
  int32_t cnt = 0;
  uint32_t px_c_min = box_x_min, px_c_max = box_x_max;
  uint32_t px_r_min = box_y_min, px_r_max = box_y_max;
  for (uint32_t y = px_r_min; y < px_r_max; y++) {
    for (uint32_t x = px_c_min; x < px_c_max; x += 2) {
      uint32_t idx = (y * width + x) * 2;
      uint8_t u    = buf[idx];
      uint8_t lum1 = buf[idx + 1];
      uint8_t v    = buf[idx + 2];
      uint8_t lum2 = buf[idx + 3];
      if (lum1 >= POLE_Y_MIN && lum1 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += x; sum_y += y; cnt++;
      }
      if (lum2 >= POLE_Y_MIN && lum2 <= POLE_Y_MAX &&
          u    >= POLE_U_MIN && u    <= POLE_U_MAX  &&
          v    >= POLE_V_MIN && v    <= POLE_V_MAX) {
        sum_x += (x + 1); sum_y += y; cnt++;
      }
    }
  }

  pole_cx    = (cnt > 0) ? (int16_t)(sum_x / cnt) : (int16_t)(sum_x_all / total_orange);
  pole_cy    = (cnt > 0) ? (int16_t)(sum_y / cnt) : (int16_t)(sum_y_all / total_orange);
  pole_count = total_orange;
  pole_found = 1;

  // Draw best blob in green
  image_draw_rectangle(img, (int)box_x_min, (int)box_x_max,
                            (int)box_y_min,  (int)box_y_max, green_color);
  struct point_t centroid = {.x = (uint32_t)pole_cx, .y = (uint32_t)pole_cy};
  image_draw_crosshair(img, &centroid, red_color, 10);

  VERBOSE_PRINT("best blob: cx=%d cy=%d size=%d box=(%d,%d)-(%d,%d)\n",
                pole_cx, pole_cy, (int)best->size,
                (int)box_x_min, (int)box_y_min, (int)box_x_max, (int)box_y_max);

  return img;
}

// ── Module init ───────────────────────────────────────────────────────────────
void pole_detector_init(void)
{
  cv_add_to_device(&POLE_DETECTOR_CAMERA, detect_orange, POLE_DETECTOR_FPS, 0);
  VERBOSE_PRINT("Pole detector initialised.\n");
}

// ── Module periodic — publishes ABI message ───────────────────────────────────
void pole_detector_periodic(void)
{
  AbiSendMsgVISUAL_DETECTION(
    POLE_DETECTOR_ABI_ID,
    pole_cx,
    pole_cy,
    0,
    0,
    pole_count,
    pole_found
  );

  VERBOSE_PRINT("periodic: found=%d cx=%d cy=%d count=%d\n",
                pole_found, pole_cx, pole_cy, pole_count);
}