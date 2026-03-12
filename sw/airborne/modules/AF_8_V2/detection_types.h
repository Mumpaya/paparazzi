/*
 * detection_types.h — shared data structures for all CV nodes
 *
 * Every node (ground-edge, obstacle, gate) writes into its own
 * result struct.  The central FrameResults aggregates them all
 * so that the controller can read one flat object per frame.
 */
#ifndef DETECTION_TYPES_H
#define DETECTION_TYPES_H

#include <stdint.h>
#include <stdbool.h>

/* ── tunables ──────────────────────────────────────────────────── */
#define MAX_CONFIRMED_LINES   16
#define MAX_REJECTED_LINES    16
#define MAX_RAW_LINES         64
#define MAX_OBSTACLE_COLS     32
#define MAX_GAP_CANDIDATES     3
#define MAX_ORANGE_BOXES      16
#define MAX_HOUGH_BOXES       32
#define OBSTACLE_N_COLS       22

/* ── simple 2-D integer point ─────────────────────────────────── */
typedef struct {
    int x;
    int y;
} Point2i;

/* ================================================================
 * GROUND-EDGE NODE
 * ================================================================ */

/* A confirmed or rejected line (extended endpoints + original seg) */
typedef struct {
    Point2i ep1;        /* extended endpoint 1 (full-width)       */
    Point2i ep2;        /* extended endpoint 2                    */
    Point2i orig_p1;    /* original detected segment start        */
    Point2i orig_p2;    /* original detected segment end          */
} GroundEdgeLine;

typedef struct {
    Point2i ep1;
    Point2i ep2;
    /* reason string not transferred — just flagged as rejected   */
} RejectedLine;

typedef struct {
    /* confirmed real ground-boundary lines (after temporal + mat filter) */
    GroundEdgeLine confirmed[MAX_CONFIRMED_LINES];
    int            n_confirmed;

    /* temporally confirmed but rejected as mat-edge */
    RejectedLine   rejected[MAX_REJECTED_LINES];
    int            n_rejected;

    /* were any lines detected at all this frame? */
    bool           has_lines;
} GroundEdgeResult;


/* ================================================================
 * OBSTACLE DETECTOR NODE
 * ================================================================ */

typedef struct {
    int   x, y, w, h;
    int   cx, cy;
    int   area;
    float confidence;
} PoleDetection;

typedef struct {
    int   rank;               /* 1 = best                              */
    int   center_x;           /* pixel x in ROI-width space            */
    int   col_start;
    int   col_end;
    int   width_cols;
    float safety;             /* composite [0,1]                       */
    float width_score;
    float clearness_score;
    float centrality_score;
} GapCandidate;

typedef struct {
    /* fused per-column obstacle scores (n_cols entries) */
    float fused_scores[OBSTACLE_N_COLS];

    /* obstacle columns (indices into 0..n_cols-1) */
    int   obstacle_cols[MAX_OBSTACLE_COLS];
    int   n_obstacle_cols;

    /* top-N free-corridor gaps, sorted best-first */
    GapCandidate gaps[MAX_GAP_CANDIDATES];
    int          n_gaps;

    /* convenience: best gap centre or -1 if none */
    int   gap_center_x;

    /* orange pole detections inside the ROI */
    PoleDetection orange_boxes[MAX_ORANGE_BOXES];
    int           n_orange_boxes;

    int   n_cols;             /* actual column count used              */

    /* ── CONTROLLER-READY VALUES ─────────────────────────────────
     * All pixel values are in the rotated-frame coordinate space.
     *
     * Multiple flyzones per frame are supported (up to MAX_GAP_CANDIDATES = 3).
     * They are stored in flyzone[] sorted best-first (rank 1 = safest).
     * n_flyzones tells you how many are valid this frame (0 = fully blocked). */

    int  n_flyzones;          /* how many valid flyzones found (0–3)   */

    struct {
        /* Horizontal pixel boundaries of this flyable corridor.
         * Use left/right to know exactly where the gap is.          */
        int   left_px;        /* left  edge (pixels)                  */
        int   right_px;       /* right edge (pixels)                  */

        /* Centre of this corridor — direct input to heading control:
         *   error = center_px - (frame_width / 2)                   */
        int   center_px;

        /* Width of this corridor in pixels.
         * Wider = more room to manoeuvre.                            */
        int   width_px;

        /* Composite safety score [0.0 – 1.0].
         * Combines: width (50%) + clearness (40%) + centrality (10%) */
        float safety;

    } flyzone[MAX_GAP_CANDIDATES];  /* [0] = best, [1] = 2nd, [2] = 3rd */

} ObstacleResult;


/* ================================================================
 * GATE DETECTOR NODE
 * ================================================================ */

typedef struct {
    /* ── CONTROLLER-READY VALUES ─────────────────────────────────
     * These are the values you directly use in a control script.   */

    /* Was a valid gate found this frame? */
    bool  detected;

    /* Gate centre — pixels (valid when detected == true) */
    int   cx_px;              /* horizontal centre in pixels           */
    int   cy_px;              /* vertical   centre in pixels           */

    /* Gate centre — normalised [0–100] % of frame size.
     * Use cx_pct to drive a heading controller:
     *   error = cx_pct - 50.0   (negative = gate is to the left)   */
    float cx_pct;
    float cy_pct;

    /* Approach angle — are we flying straight at the gate?
     *   GATE_ANGLE_GOOD  (0) = centred, use dist_m for approach
     *   GATE_ANGLE_LEFT  (1) = gate to our left  → yaw left
     *   GATE_ANGLE_RIGHT (2) = gate to our right → yaw right        */
    int   angle;

    /* Estimated distance to gate in metres.
     * Meaningful only when angle == GATE_ANGLE_GOOD.
     * Based on horizontal pillar separation (linear model).         */
    float dist_m;

} GateResult;

#define GATE_ANGLE_GOOD   0
#define GATE_ANGLE_LEFT   1
#define GATE_ANGLE_RIGHT  2


/* ================================================================
 * CENTRAL FRAME RESULTS — one per frame, read by controller
 * ================================================================ */

typedef struct {
    /* set to true once the main loop has filled all fields */
    bool valid;

    GroundEdgeResult  ground_edge;
    ObstacleResult    obstacle;
    GateResult        gate;
} FrameResults;


#endif /* DETECTION_TYPES_H */
