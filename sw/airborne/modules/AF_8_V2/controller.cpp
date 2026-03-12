/*
 * controller.cpp — AF_8_V2 heading/velocity controller
 *
 * See controller.h for the full design notes.
 *
 * ── Decision tree (called once per frame) ────────────────────────
 *
 *  1. Check gate in flyzone overlap → gate_in_flyzone flag
 *  2. Evaluate GATELOCK entry / exit conditions
 *
 *  GATELOCK mode:
 *    a. Gate visible → track gate heading, V_GATE
 *    b. Gate gone    → coast (GATELOCK_COAST_FRAMES), fly straight
 *    c. Coast done   → return to NORMAL
 *
 *  NORMAL mode (priority order):
 *    A. Gate detected AND gate_in_flyzone AND angle==GOOD  → steer to gate
 *    B. Gate detected AND gate_in_flyzone AND angle==LEFT  → yaw left  (recover)
 *    C. Gate detected AND gate_in_flyzone AND angle==RIGHT → yaw right (recover)
 *    D. Gate detected AND NOT gate_in_flyzone  → ignore gate, fall through
 *    E. Flyzones available → steer to best flyzone centre
 *    F. Nothing            → fly straight, V_STD
 *
 *  When both gate (in flyzone) and flyzones are present, the gate takes
 *  priority (cases A–C), because a gate target is more precise and we have
 *  already confirmed it sits in navigable space via the overlap check.
 */

#include "controller.h"
#include <cmath>
#include <cstring>
#include <algorithm>

/* ── helpers ──────────────────────────────────────────────────── */

/**
 * Convert a pixel x position to a "% of frame width" value (0–100).
 */
static inline float px_to_pct(int px, int img_w)
{
    if (img_w <= 0) return 50.0f;
    float p = (float)px / (float)img_w * 100.0f;
    if (p < 0.0f)   p = 0.0f;
    if (p > 100.0f) p = 100.0f;
    return p;
}

/**
 * Check whether the gate centre (gate.cx_px) falls inside any flyzone
 * whose width_px >= GATE_MIN_FLYZONE_WIDTH_PX.
 *
 * Returns true if at least one qualifying flyzone contains the gate.
 */
static bool gate_overlaps_flyzone(const GateResult *gate,
                                  const ObstacleResult *obs)
{
    if (!gate->detected)          return false;
    if (obs->n_flyzones == 0)     return false;

    int gcx = gate->cx_px;
    for (int i = 0; i < obs->n_flyzones; i++) {
        const auto &fz = obs->flyzone[i];
        if (fz.width_px < GATE_MIN_FLYZONE_WIDTH_PX) continue;
        if (gcx >= fz.left_px && gcx <= fz.right_px)
            return true;
    }
    return false;
}

/**
 * True if the gate has a GOOD angle AND its centre is not sitting on an
 * obstacle column — used for GATELOCK qualification.
 */
static bool is_gatelock_eligible(const FrameResults *fr, int img_w)
{
    const GateResult    *gate = &fr->gate;
    const ObstacleResult *obs = &fr->obstacle;

    if (!gate->detected)                   return false;
    if (gate->angle != GATE_ANGLE_GOOD)    return false;

    /* Check that the gate cx is NOT landing on a known obstacle column.
     * We do this by looking at the fused_scores at the column index
     * corresponding to gate.cx_px.  If that column is an obstacle column,
     * the path is actually blocked → don't lock. */
    int n_cols = obs->n_cols > 0 ? obs->n_cols : OBSTACLE_N_COLS;
    float col_w = (float)img_w / n_cols;
    if (col_w > 0) {
        int gate_col = (int)(gate->cx_px / col_w);
        gate_col = std::max(0, std::min(gate_col, n_cols - 1));
        for (int i = 0; i < obs->n_obstacle_cols; i++) {
            if (obs->obstacle_cols[i] == gate_col)
                return false;  /* obstacle at gate position */
        }
    }
    return true;
}

/* ── public API ─────────────────────────────────────────────────── */

void controller_init(ControllerState *cs)
{
    memset(cs, 0, sizeof(*cs));
    cs->mode         = CTRL_MODE_NORMAL;
    cs->good_streak  = 0;
    cs->coast_frames = 0;
}

ControlOutput controller_update(ControllerState *cs,
                                const FrameResults *fr,
                                int img_w)
{
    ControlOutput out;
    memset(&out, 0, sizeof(out));

    /* ── fill persistent-state diagnostics ──────────────────── */
    out.good_streak      = cs->good_streak;
    out.coast_frames_left = cs->coast_frames;
    out.mode             = cs->mode;

    /* ── gate-in-flyzone check ───────────────────────────────── */
    bool has_gate     = fr->valid && fr->gate.detected;
    bool has_flyzones = fr->valid && (fr->obstacle.n_flyzones > 0);

    bool gate_in_fz = false;
    if (has_gate && has_flyzones) {
        gate_in_fz = gate_overlaps_flyzone(&fr->gate, &fr->obstacle);
    } else if (has_gate && !has_flyzones) {
        /* No obstacle data at all — treat gate as unobstructed */
        gate_in_fz = true;
    }
    out.gate_in_flyzone = gate_in_fz;

    /* ── GATELOCK entry / maintenance ──────────────────────────
     *
     * Entry:  GOOD angle + gate centre not on obstacle col,
     *         for GATELOCK_TRIGGER_FRAMES consecutive frames.
     * Stay:   As long as the gate is visible in GATELOCK mode.
     * Coast:  After gate disappears, coast for GATELOCK_COAST_FRAMES.
     * Exit:   Coast expires → NORMAL.
     */
    bool eligible = fr->valid && is_gatelock_eligible(fr, img_w);

    if (cs->mode == CTRL_MODE_NORMAL) {
        if (eligible) {
            cs->good_streak++;
            if (cs->good_streak >= GATELOCK_TRIGGER_FRAMES) {
                cs->mode         = CTRL_MODE_GATELOCK;
                cs->coast_frames = 0;
            }
        } else {
            cs->good_streak = 0;   /* reset on any non-qualifying frame */
        }
    } else {
        /* GATELOCK mode */
        if (has_gate) {
            cs->coast_frames = 0;  /* gate visible, reset coast */
            /* refresh good_streak so we know we're still tracking */
            cs->good_streak++;
        } else {
            /* gate lost — start / continue coast */
            if (cs->coast_frames == 0)
                cs->coast_frames = GATELOCK_COAST_FRAMES;
            else
                cs->coast_frames--;

            if (cs->coast_frames <= 0) {
                cs->mode        = CTRL_MODE_NORMAL;
                cs->good_streak = 0;
                cs->coast_frames = 0;
            }
        }
    }

    /* refresh diagnostics after update */
    out.good_streak       = cs->good_streak;
    out.coast_frames_left = cs->coast_frames;
    out.mode              = cs->mode;

    /* ══════════════════════════════════════════════════════════
     * GATELOCK BRANCH
     * ═════════════════════════════════════════════════════════ */
    if (cs->mode == CTRL_MODE_GATELOCK) {

        if (has_gate) {
            /* Track gate heading */
            float target_pct   = fr->gate.cx_pct;
            float error_pct    = target_pct - 50.0f;
            float delta_yaw    = K_YAW * error_pct;

            out.action          = CTRL_ACTION_GATELOCK_TRACK;
            out.target_x_pct    = target_pct;
            out.heading_error_pct = error_pct;
            out.delta_yaw_rad   = delta_yaw;

            /* Slow down as we get very close */
            float v = V_GATE;
            if (fr->gate.dist_m > 0 && fr->gate.dist_m < GATE_SLOW_DIST_M)
                v = V_STD + (V_GATE - V_STD) * (fr->gate.dist_m / GATE_SLOW_DIST_M);
            out.forward_vel = v;

        } else {
            /* Coast — fly straight, no yaw change */
            out.action          = CTRL_ACTION_GATELOCK_COAST;
            out.target_x_pct    = 50.0f;
            out.heading_error_pct = 0.0f;
            out.delta_yaw_rad   = 0.0f;
            out.forward_vel     = V_GATE;   /* keep speed up through gate */
        }

        return out;
    }

    /* ══════════════════════════════════════════════════════════
     * NORMAL BRANCH
     * ═════════════════════════════════════════════════════════ */

    /* Priority A / B / C — gate detected and in a valid flyzone */
    if (has_gate && gate_in_fz) {

        float target_pct, error_pct, delta_yaw;

        if (fr->gate.angle == GATE_ANGLE_GOOD) {
            /* A — steer to gate centre */
            target_pct = fr->gate.cx_pct;
            error_pct  = target_pct - 50.0f;
            delta_yaw  = K_YAW * error_pct;
            out.action = CTRL_ACTION_GATE_STEER;

        } else {
            /* B / C — recover heading to find GOOD angle
             * We apply a fixed small rotation in the direction needed.
             * K_YAW * 20 gives ~0.02 rad per frame — gentle recovery. */
            float sign = (fr->gate.angle == GATE_ANGLE_LEFT) ? -1.0f : 1.0f;
            delta_yaw  = K_YAW * 20.0f * sign;
            target_pct = 50.0f + sign * 20.0f;   /* notional target */
            error_pct  = target_pct - 50.0f;
            out.action = CTRL_ACTION_GATE_RECOVER;
        }

        out.target_x_pct      = target_pct;
        out.heading_error_pct = error_pct;
        out.delta_yaw_rad     = delta_yaw;
        out.forward_vel       = V_STD;
        return out;
    }

    /* Priority E — flyzones but no usable gate */
    if (has_flyzones) {
        /* Use the best (rank-0) flyzone centre as heading target */
        const auto &best_fz = fr->obstacle.flyzone[0];
        float target_pct  = px_to_pct(best_fz.center_px, img_w);
        float error_pct   = target_pct - 50.0f;
        float delta_yaw   = K_YAW * error_pct;

        out.action            = CTRL_ACTION_FLYZONE_STEER;
        out.target_x_pct      = target_pct;
        out.heading_error_pct = error_pct;
        out.delta_yaw_rad     = delta_yaw;
        out.forward_vel       = V_STD;
        return out;
    }

    /* Priority F — nothing found */
    out.action            = CTRL_ACTION_STRAIGHT;
    out.target_x_pct      = 50.0f;
    out.heading_error_pct = 0.0f;
    out.delta_yaw_rad     = 0.0f;
    out.forward_vel       = V_STD;
    return out;
}
