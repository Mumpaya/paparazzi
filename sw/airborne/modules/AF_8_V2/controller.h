/*
 * controller.h — AF_8_V2 heading/velocity controller
 *
 * Reads one FrameResults per frame and produces a ControlOutput
 * (desired heading delta in radians + forward velocity in m/s).
 *
 * ── State machine ─────────────────────────────────────────────────
 *
 *  NORMAL  (default)
 *   • Nothing seen            → fly straight, V_STD
 *   • Flyzones only           → yaw towards best-flyzone centre
 *   • Gate only               → yaw towards gate centre (if GOOD),
 *                               else yaw LEFT/RIGHT to recover GOOD;
 *                               ignore gate if not in a wide-enough flyzone
 *   • Gate + flyzones         → same as gate-only but check flyzone overlap
 *   Transition to GATELOCK    → GATE_ANGLE_GOOD for GATELOCK_TRIGGER_FRAMES
 *                               consecutive frames AND gate centre not in an
 *                               obstacle column region
 *
 *  GATELOCK  (high-confidence straight approach)
 *   • Ignore obstacle detector entirely
 *   • Track gate heading only
 *   • Transition back to NORMAL after gate disappears for
 *     GATELOCK_COAST_FRAMES frames  (fly straight during coast)
 *
 * ── Heading update ────────────────────────────────────────────────
 *   delta_yaw = K_YAW * error_pct / 100.0          (radians)
 *   where error_pct = desired_x_pct - 50.0
 *   sign convention: positive → yaw right, negative → yaw left
 *
 * ── Flyzone overlap check ─────────────────────────────────────────
 *   Gate centre (px) must lie within a flyzone whose width_px ≥
 *   GATE_MIN_FLYZONE_WIDTH_PX.  If no such flyzone exists, ignore gate.
 */

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "detection_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── tunables ──────────────────────────────────────────────────── */

/** Yaw gain: radians of heading change per 1% of frame-width error.
 *  With a 50%-wide error (gate at far left) this gives 0.05 rad ≈ 2.9°.
 *  Keep it small so the UAV steers gradually rather than aggressively.  */
#define K_YAW                   0.001f   /* rad / (% error)               */

/** Forward velocity when flying straight (m/s). */
#define V_STD                   0.2f

/** Forward velocity during gate approach (GATELOCK mode). */
#define V_GATE                  0.3f

/** Number of consecutive GOOD-angle + unobstructed gate frames needed
 *  to enter GATELOCK mode. */
#define GATELOCK_TRIGGER_FRAMES 7

/** Number of frames after gate disappears before leaving GATELOCK mode
 *  (the UAV is assumed to be flying through the gate during this coast). */
#define GATELOCK_COAST_FRAMES   20

/** Gate centre must lie inside a flyzone of at least this width (px)
 *  for the gate to be used as the heading target in NORMAL mode.
 *  If no qualifying flyzone exists, the gate is ignored for heading. */
#define GATE_MIN_FLYZONE_WIDTH_PX 60

/** The minimum gate distance (m) before reducing approach velocity. */
#define GATE_SLOW_DIST_M        1.5f

/* ── smoothing (rolling window) ─────────────────────────────────
 * Apply a small moving-average window to the controller outputs to
 * reduce frame-to-frame jitter before they are published.
 */
#ifndef CTRL_SMOOTH_WINDOW
#define CTRL_SMOOTH_WINDOW 10
#endif

/* ── controller state ───────────────────────────────────────────── */

typedef enum {
    CTRL_MODE_NORMAL   = 0,   /**< obstacle-aware navigation           */
    CTRL_MODE_GATELOCK = 1    /**< locked onto gate, ignore obstacles  */
} CtrlMode;

/** What the controller decided to do this frame. */
typedef enum {
    CTRL_ACTION_STRAIGHT        = 0,  /**< no info — fly straight          */
    CTRL_ACTION_FLYZONE_STEER   = 1,  /**< steering toward flyzone centre  */
    CTRL_ACTION_GATE_STEER      = 2,  /**< steering toward gate (normal)   */
    CTRL_ACTION_GATE_RECOVER    = 3,  /**< rotating to regain GOOD angle   */
    CTRL_ACTION_GATELOCK_TRACK  = 4,  /**< GATELOCK tracking gate          */
    CTRL_ACTION_GATELOCK_COAST  = 5   /**< GATELOCK coast (gate lost)      */
} CtrlAction;

/** The controller outputs one of these per frame. */
typedef struct {
    CtrlMode   mode;           /**< current state machine mode          */
    CtrlAction action;         /**< what the controller is doing        */
    float      delta_yaw_rad;  /**< heading change this frame (rad)     */
    float      forward_vel;    /**< forward velocity command (m/s)      */

    /* diagnostic — what target was used */
    float  target_x_pct;       /**< desired heading x as % of frame     */
    float  heading_error_pct;  /**< target_x_pct - 50.0  (signed)       */

    /* GATELOCK counters (informational) */
    int    good_streak;        /**< consecutive GOOD frames so far       */
    int    coast_frames_left;  /**< >0 while coasting after gate loss    */

    /* flyzone overlap result */
    bool   gate_in_flyzone;    /**< gate centre overlaps a valid flyzone */
} ControlOutput;

/* ── internal persistent state ─────────────────────────────────── */

/** Opaque controller state — zero-initialise before first call. */
typedef struct {
    CtrlMode mode;
    int      good_streak;      /**< consecutive GOOD+unobstructed frames */
    int      coast_frames;     /**< frames remaining in GATELOCK coast   */
    /* history buffers for simple moving-average smoothing */
    float    yaw_hist[CTRL_SMOOTH_WINDOW];
    float    vel_hist[CTRL_SMOOTH_WINDOW];
    int      hist_idx;        /* next index to overwrite (circular) */
    int      hist_len;        /* number of valid entries (<= window) */
} ControllerState;

/* ── public API ─────────────────────────────────────────────────── */

/**
 * Reset the controller to its initial state.
 * Call once at startup.
 */
void controller_init(ControllerState *cs);

/**
 * Run one frame of the controller.
 *
 * @param cs    Persistent state (modified in place)
 * @param fr    Detection results for this frame
 * @param img_w Camera image width (pixels, in rotated space)
 * @return      Control command for this frame
 */
ControlOutput controller_update(ControllerState *cs,
                                const FrameResults *fr,
                                int img_w);

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_H */
