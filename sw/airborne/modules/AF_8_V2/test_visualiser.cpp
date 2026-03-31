/*
 * test_visualiser.cpp — Standalone playground test for AF_8_V2 CV pipeline
 *
 * Reads all .jpg images from Playground/, runs the three detector nodes
 * and the controller on each frame, and renders two windows:
 *
 * MAIN WINDOW (3 panels):
 *   LEFT   panel  — obstacle detector overlay (flyable zones + column scores)
 *   MIDDLE panel  — gate detector overlay (gate square, centre, angle, distance)
 *   RIGHT  panel  — controller action panel (mode, heading target, velocity)
 *
 * BOTTOM CAMERA WINDOW (separate popup):
 *   — bottom camera detection visualization with arrow and centroid overlay
 *   — runs in parallel with main 3-panel display
 *   — loads bottom camera images from Playground_Bottom folder
 *
 * Simulated controller output is printed in the top-left corner:
 *   • obstacle: n_flyzones, each zone left/center/right px + safety
 *   • gate:     detected, cx%, cy%, angle, dist_m
 *   • ctrl:     mode, action, delta_yaw, velocity, GATELOCK counters
 *
 * Controls:
 *   SPACE  — pause / resume
 *   A / D  — step backward / forward one frame (when paused)
 *   Q      — quit
 */

#include "detection_types.h"
#include "ground_edge_node.h"
#include "obstacle_detector_node.h"
#include "gate_detector_node.h"
#include "bottom_cam_node.h"
#include "controller.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

/* ── helpers ────────────────────────────────────────────────────── */
static void txt(cv::Mat &img, const std::string &s, cv::Point pt,
                cv::Scalar col, double scale = 0.45, int thick = 1)
{
    // Intentionally empty: visualization mode with no text overlays.
    (void)img; (void)s; (void)pt; (void)col; (void)scale; (void)thick;
}

/* Small helper to draw text only on the controller panel. We keep the
 * global txt() as a no-op so other panels remain text-free, but this
 * function draws using OpenCV directly for the controller UI. */
static void ctrl_txt(cv::Mat &img, const std::string &s, cv::Point pt,
                     cv::Scalar col, double scale = 0.45, int thick = 1)
{
    cv::putText(img, s, pt, cv::FONT_HERSHEY_SIMPLEX, scale, col, thick, cv::LINE_AA);
}

/* collect & sort images from folder */
static std::vector<std::string> collect_images(const std::string &dir)
{
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "Cannot open folder: %s\n", dir.c_str()); return out; }
    struct dirent *e;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() > 4) {
            std::string ext = n.substr(n.size() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp")
                out.push_back(dir + "/" + n);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

/* ── draw obstacle overlay on a copy of the rotated frame ───────── */
static cv::Mat draw_obstacle(const cv::Mat &rotated, const ObstacleResult &obs)
{
    cv::Mat out = rotated.clone();
    int h = out.rows, w = out.cols;
    int n_cols = obs.n_cols > 0 ? obs.n_cols : OBSTACLE_N_COLS;
    float col_w = (float)w / n_cols;

    /* semi-transparent column tint: red = obstacle, dark-green = free */
    for (int c = 0; c < n_cols; c++) {
        bool is_obs = false;
        for (int i = 0; i < obs.n_obstacle_cols; i++)
            if (obs.obstacle_cols[i] == c) { is_obs = true; break; }

        int x1 = (int)(c * col_w);
        int x2 = (int)((c + 1) * col_w);
        cv::Mat roi = out(cv::Rect(x1, 0, x2 - x1, h));
        cv::Mat tint = roi.clone();
        if (is_obs)
            tint = cv::Scalar(0, 0, 120);       /* red tint */
        else
            tint = cv::Scalar(0, 20, 0);        /* subtle green tint */
        cv::addWeighted(roi, 0.7, tint, 0.3, 0, roi);

        /* thin column divider */
        cv::line(out, {x2, 0}, {x2, h}, {40,40,40}, 1);
    }

    /* flyzone overlays — ranked colours: green / cyan / blue */
    static const cv::Scalar ZONE_COL[3] = {
        {0, 255, 80}, {0, 220, 255}, {80, 120, 255}};

    for (int i = 0; i < obs.n_flyzones; i++) {
        const auto &fz = obs.flyzone[i];
        cv::Scalar col = ZONE_COL[i];

        /* bright vertical band */
        cv::Mat band = out(cv::Rect(fz.left_px, 0,
                                    std::max(fz.right_px - fz.left_px, 1), h));
        cv::Mat tint2(band.size(), band.type(), col * 0.3);
        cv::addWeighted(band, 0.7, tint2, 0.3, 0, band);

        /* left / right boundary lines */
        cv::line(out, {fz.left_px,  0}, {fz.left_px,  h}, col, 2);
        cv::line(out, {fz.right_px, 0}, {fz.right_px, h}, col, 2);

        /* centre arrow */
        int cx = fz.center_px;
        cv::arrowedLine(out, {cx, h/2 + 30}, {cx, h/2 - 30}, col, 3, cv::LINE_AA, 0, 0.35);

    /* no textual labels on flyzones in left panel (visual-only) */
    }

    /* Do not draw textual 'NO FLYZONE' message; keep visualization minimal */

    /* fused score bar at bottom */
    int bar_max_h = 40;
    float max_s = 0;
    for (int c = 0; c < n_cols; c++) max_s = std::max(max_s, obs.fused_scores[c]);
    if (max_s > 0) {
        for (int c = 0; c < n_cols; c++) {
            int x1 = (int)(c * col_w);
            int x2 = (int)((c + 1) * col_w);
            int bh  = (int)(obs.fused_scores[c] / max_s * bar_max_h);
            bool is_obs = false;
            for (int i = 0; i < obs.n_obstacle_cols; i++)
                if (obs.obstacle_cols[i] == c) { is_obs = true; break; }
            cv::rectangle(out, {x1, h - bh}, {x2-1, h},
                          is_obs ? cv::Scalar(0,60,255) : cv::Scalar(0,180,80), -1);
        }
    }

    return out;
}

/* ── draw gate overlay on a copy of the rotated frame ───────────── */
static cv::Mat draw_gate(const cv::Mat &rotated, const GateResult &gate)
{
    cv::Mat out = rotated.clone();
    int h = out.rows, w = out.cols;

    if (!gate.detected) {
        txt(out, "No Gate", {w/2 - 40, h/2}, {0,0,255}, 0.7, 2);
        return out;
    }

    int cx = gate.cx_px;
    int cy = gate.cy_px;

    /* crosshair at gate centre */
    cv::drawMarker(out, {cx, cy}, {0,255,255}, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);

    /* vertical centre line */
    cv::line(out, {cx, 0}, {cx, h}, {0,255,255}, 1, cv::LINE_AA);

    /* approach angle label */
    static const char *angle_str[] = {"GOOD", "LEFT", "RIGHT"};
    static const cv::Scalar angle_col[] = {{0,255,0},{0,165,255},{0,165,255}};
    cv::Scalar acol = angle_col[gate.angle];

    // No textual labels for gate — visual-only

    /* error bar — how far from centre */
    float error_pct = gate.cx_pct - 50.0f;   /* negative = gate left */
    int bar_x2 = w / 2 + (int)(error_pct / 50.0f * (w / 4));
    cv::line(out, {w/2, h - 12}, {bar_x2, h - 12},
             gate.angle == GATE_ANGLE_GOOD ? cv::Scalar(0,255,0) : cv::Scalar(0,165,255), 4);
    cv::line(out, {w/2, h - 18}, {w/2, h - 6}, {200,200,200}, 1);
    // heading error visualized by bar; no text label

    return out;
}

/* ── bottom camera ground edge panel ────────────────────────────── */
static cv::Mat draw_bottom_cam(const cv::Mat &bgr, const BottomCamResult &bc, int fw)
{
    cv::Mat out = bgr.clone();
    int h = out.rows, w = out.cols;
    
    if (!bc.detected) {
        // no text when no detection — keep bottom view clean
        return out;
    }
    
    /* Frame center */
    int fcx = w / 2, fcy = h / 2;
    cv::circle(out, {fcx, fcy}, 4, {255, 0, 0}, -1, cv::LINE_AA);
    
    /* Centroid */
    cv::circle(out, {bc.cx_px, bc.cy_px}, 8, {0, 255, 255}, 2, cv::LINE_AA);
    
    /* Arrow from center to centroid */
    cv::Scalar arrow_col = bc.over_edge ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::arrowedLine(out, {fcx, fcy}, {bc.cx_px, bc.cy_px},
                    arrow_col, 3, cv::LINE_AA, 0, 0.3);
    
    // No textual diagnostics on bottom camera view
    
    return out;
}

/* ── controller visualisation panel ─────────────────────────────── */
/*
 * Renders a dark panel (same size as one camera frame) showing:
 *  • Mode banner   — NORMAL (green) or GATELOCK (orange)
 *  • GATELOCK entry streak bar (0 → GATELOCK_TRIGGER_FRAMES)
 *  • Coast countdown bar (when coasting)
 *  • Centred compass-style heading-error indicator
 *    (needle angle proportional to heading error)
 *  • Action label + forward velocity bar
 *  • Gate-in-flyzone flag
 */
static cv::Mat draw_ctrl_panel(const ControlOutput &ctrl,
                               const FrameResults  &fr,
                               int w, int h)
{
    cv::Mat panel(h, w, CV_8UC3, cv::Scalar(18, 18, 18));

    /* ── mode banner ────────────────────────────────────────── */
    bool gatelock = (ctrl.mode == CTRL_MODE_GATELOCK);
    bool edge = (ctrl.mode == CTRL_MODE_EDGE);
    
    cv::Scalar mode_col;
    const char *mode_str;
    
    if (edge) {
        mode_col = cv::Scalar(255, 0, 0);  /* BLUE */
        mode_str = "EDGE ALIGN";
    } else if (gatelock) {
        mode_col = cv::Scalar(0, 165, 255);  /* orange */
        mode_str = "GATELOCK";
    } else {
        mode_col = cv::Scalar(0, 220, 60);  /* green */
        mode_str = "NORMAL";
    }

    /* filled banner bar (no text) */
    cv::rectangle(panel, {0, 0}, {w, 34}, mode_col * 0.35, -1);
    cv::rectangle(panel, {0, 0}, {w, 34}, mode_col, 2);

    /* ── action label ───────────────────────────────────────── */
    static const char *action_names[] = {
        "FLY STRAIGHT",
        "FLYZONE STEER",
        "GATE STEER",
        "GATE RECOVER",
        "GATELOCK TRACK",
        "GATELOCK COAST",
        "EDGE ALIGN"
    };
    static const cv::Scalar action_cols[] = {
        {180,180,180},   /* STRAIGHT       */
        {0, 220,  60},   /* FLYZONE STEER  */
        {0, 255, 200},   /* GATE STEER     */
        {0, 165, 255},   /* GATE RECOVER   */
        {0, 200, 255},   /* GATELOCK TRACK */
        {80, 80, 255},   /* GATELOCK COAST */
        {255, 0, 0}      /* EDGE ALIGN     (blue) */
    };
    int ai = (int)ctrl.action;
    cv::Scalar acol = action_cols[ai];
    {
        char abuf[64];
        snprintf(abuf, sizeof(abuf), "Action: %s", action_names[ai]);
        cv::putText(panel, abuf, {8, 56},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, acol, 1, cv::LINE_AA);
    }

    /* ── GATELOCK streak / coast progress bars ───────────────── */
    int by = 70;
    /* streak bar (fills towards GATELOCK) */
    {
        int bar_w = w - 16;
        int fill  = (int)((float)ctrl.good_streak / GATELOCK_TRIGGER_FRAMES * bar_w);
        fill = std::min(fill, bar_w);
        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 14},
                      cv::Scalar(40,40,40), -1);
        if (fill > 0)
            cv::rectangle(panel, {8, by}, {8 + fill, by + 14},
                          gatelock ? cv::Scalar(0,165,255) : cv::Scalar(0,200,80), -1);
        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 14},
                      cv::Scalar(80,80,80), 1);
    char sbuf[48];
    snprintf(sbuf, sizeof(sbuf), "GOOD streak: %d / %d",
         ctrl.good_streak, GATELOCK_TRIGGER_FRAMES);
    ctrl_txt(panel, sbuf, {12, by + 11}, {180,180,180}, 0.38);
        by += 20;
    }

    /* coast countdown bar */
    if (ctrl.coast_frames_left > 0) {
        int bar_w = w - 16;
        int fill  = (int)((float)ctrl.coast_frames_left
                          / GATELOCK_COAST_FRAMES * bar_w);
        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 14},
                      cv::Scalar(40,40,40), -1);
        cv::rectangle(panel, {8, by}, {8 + fill, by + 14},
                      cv::Scalar(80,80,255), -1);
        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 14},
                      cv::Scalar(80,80,80), 1);
    char cbuf[48];
    snprintf(cbuf, sizeof(cbuf), "Coast frames left: %d",
         ctrl.coast_frames_left);
    ctrl_txt(panel, cbuf, {12, by + 11}, {100,100,255}, 0.38);
        by += 20;
    }

    /* ── compass / heading needle ────────────────────────────── */
    by += 8;
    int compass_cx = w / 2;
    int compass_cy = by + 90;
    int compass_r  = 75;

    /* outer ring */
    cv::circle(panel, {compass_cx, compass_cy}, compass_r,
               cv::Scalar(70,70,70), 2, cv::LINE_AA);
    /* centre dot */
    cv::circle(panel, {compass_cx, compass_cy}, 4,
               cv::Scalar(200,200,200), -1, cv::LINE_AA);
    /* 0-error vertical tick */
    cv::line(panel,
             {compass_cx, compass_cy - compass_r - 6},
             {compass_cx, compass_cy - compass_r + 6},
             cv::Scalar(120,120,120), 1, cv::LINE_AA);

    /* heading needle: error_pct in [-50, +50] → angle in [-90°, +90°]
     * pointing from centre outward.  Needle at top = 0 error.          */
    float ang_deg  = ctrl.heading_error_pct * 1.8f;   /* 50% → 90° */
    float ang_rad2 = (float)((ang_deg - 90.0f) * M_PI / 180.0f);
    int needle_x   = compass_cx + (int)(compass_r * 0.85f * cosf(ang_rad2));
    int needle_y   = compass_cy + (int)(compass_r * 0.85f * sinf(ang_rad2));

    cv::arrowedLine(panel, {compass_cx, compass_cy}, {needle_x, needle_y},
                    acol, 3, cv::LINE_AA, 0, 0.25);

    /* labels */
    /* keep compass labels off to avoid clutter; controller panel text
     * is drawn below in a focused area using ctrl_txt() */

    /* no textual heading diagnostics; needle is shown visually */

    by = compass_cy + compass_r + 35;

    /* ── forward velocity bar ────────────────────────────────── */
    {
        float v_max = V_GATE + 0.05f;
        int bar_w   = w - 16;
        int fill    = (int)(ctrl.forward_vel / v_max * bar_w);
        fill = std::min(fill, bar_w);

        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 18},
                      cv::Scalar(30,30,30), -1);
        cv::rectangle(panel, {8, by}, {8 + fill, by + 18},
                      gatelock ? cv::Scalar(0,165,255) : cv::Scalar(0,200,80), -1);
        cv::rectangle(panel, {8, by}, {8 + bar_w, by + 18},
                      cv::Scalar(80,80,80), 1);

    /* velocity bar only — no text */
        by += 28;
    }

    /* ── gate / flyzone status flags ─────────────────────────── */
    {
        bool gd  = fr.gate.detected;
        bool gfz = ctrl.gate_in_flyzone;

        cv::Scalar gd_col  = gd  ? cv::Scalar(0,255,180) : cv::Scalar(80,80,80);
        cv::Scalar gfz_col = gfz ? cv::Scalar(0,255,100) : cv::Scalar(80,80,80);

        char buf[80];
        snprintf(buf, sizeof(buf), "Gate: %s", gd ? "DETECTED" : "not seen");
        ctrl_txt(panel, buf, {8, by}, gd_col, 0.42);
        by += 18;

        snprintf(buf, sizeof(buf), "Gate in flyzone: %s", gfz ? "YES" : "NO");
        ctrl_txt(panel, buf, {8, by}, gfz_col, 0.42);
        by += 18;

        if (gd) {
            static const char *ang_str[] = {"GOOD", "LEFT", "RIGHT"};
            static const cv::Scalar ang_col[] = {{0,255,0},{0,165,255},{0,165,255}};
            snprintf(buf, sizeof(buf), "Gate angle: %s  dist: %.2fm",
                     ang_str[fr.gate.angle], fr.gate.dist_m);
            ctrl_txt(panel, buf, {8, by}, ang_col[fr.gate.angle], 0.42);
            by += 18;
        }

        snprintf(buf, sizeof(buf), "Flyzones: %d", fr.obstacle.n_flyzones);
        ctrl_txt(panel, buf, {8, by}, {80,220,80}, 0.42);
        by += 18;

        /* show per-flyzone brief info */
        for (int i = 0; i < fr.obstacle.n_flyzones; i++) {
            const auto &fz = fr.obstacle.flyzone[i];
            char fbuf[96];
            snprintf(fbuf, sizeof(fbuf), "[%d] cx=%d w=%d safe=%.2f",
                     i+1, fz.center_px, fz.width_px, fz.safety);
            cv::Scalar col = (i==0) ? cv::Scalar(0,255,80) : (i==1) ? cv::Scalar(0,220,255) : cv::Scalar(80,120,255);
            ctrl_txt(panel, fbuf, {12, by}, col, 0.38);
            by += 16;
        }
        
        /* draw mode text on banner */
        if (mode_str) {
            ctrl_txt(panel, std::string("MODE: ") + mode_str, {w/2 - 40, 22}, {240,240,240}, 0.55, 2);
        }
    }

    return panel;
}

/* ── simulated controller HUD ────────────────────────────────────── */
static void draw_hud(cv::Mat &canvas, const FrameResults &fr,
                     const ControlOutput &ctrl,
                     int idx, int total, bool paused)
{
    /* Place HUD inside the right-most panel so it doesn't overlay the left (flyzone) panel */
    int panel_w = canvas.cols / 3;
    int hud_x = panel_w * 2 + 8;
    int y = 16;
    auto hud_line = [&](const std::string &s, cv::Scalar col = {220,220,220}) {
        txt(canvas, s, {hud_x, y}, col, 0.42);
        y += 16;
    };

    char buf[128];
    snprintf(buf, sizeof(buf), "Frame %d / %d%s", idx+1, total, paused ? "  [PAUSED]" : "");
    hud_line(buf, {255,255,100});

    /* gate */
    if (fr.gate.detected) {
        snprintf(buf, sizeof(buf), "GATE  cx=%.0f%%  cy=%.0f%%  %s  %.2fm",
                 fr.gate.cx_pct, fr.gate.cy_pct,
                 fr.gate.angle==0?"GOOD":fr.gate.angle==1?"LEFT":"RIGHT",
                 fr.gate.dist_m);
        hud_line(buf, {0,255,200});
    } else {
        hud_line("GATE  not detected", {100,100,100});
    }

    /* obstacle */
    snprintf(buf, sizeof(buf), "OBSTACLE  flyzones=%d  obs_cols=%d/%d",
             fr.obstacle.n_flyzones, fr.obstacle.n_obstacle_cols, fr.obstacle.n_cols);
    hud_line(buf, {80,220,80});

    for (int i = 0; i < fr.obstacle.n_flyzones; i++) {
        snprintf(buf, sizeof(buf),
                 "  [%d] left=%d  cx=%d  right=%d  w=%dpx  safe=%.2f",
                 i+1,
                 fr.obstacle.flyzone[i].left_px,
                 fr.obstacle.flyzone[i].center_px,
                 fr.obstacle.flyzone[i].right_px,
                 fr.obstacle.flyzone[i].width_px,
                 fr.obstacle.flyzone[i].safety);
        hud_line(buf, {0, (double)(200 - i*40), 80.0});
    }

    /* controller */
    static const char *action_names[] = {
        "STRAIGHT","FLYZONE_STEER","GATE_STEER",
        "GATE_RECOVER","GL_TRACK","GL_COAST"
    };
    bool gl = (ctrl.mode == CTRL_MODE_GATELOCK);
    snprintf(buf, sizeof(buf), "CTRL  %s  %s  dYaw=%+.4frad  v=%.2fm/s",
             gl ? "GATELOCK" : "NORMAL",
             action_names[(int)ctrl.action],
             ctrl.delta_yaw_rad, ctrl.forward_vel);
    hud_line(buf, gl ? cv::Scalar(0,165,255) : cv::Scalar(0,220,60));

    snprintf(buf, sizeof(buf), "  streak=%d  coast=%d  gateInFZ=%s",
             ctrl.good_streak, ctrl.coast_frames_left,
             ctrl.gate_in_flyzone ? "YES" : "NO");
    hud_line(buf, {160,160,160});

    hud_line("SPC=pause  A/D=step  Q=quit", {120,120,120});
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(int argc, char *argv[])
{
    /* path to playground — can override via argv[1] */
    std::string folder = argc > 1 ? argv[1]
        : "sw/airborne/modules/AF_8_V2/Playground";

    auto images = collect_images(folder);
    if (images.empty()) {
        fprintf(stderr, "No images found in %s\n", folder.c_str());
        return 1;
    }
    printf("Found %zu images in %s\n", images.size(), folder.c_str());

    /* bottom camera is disabled for this run: visualiser-only mode
     * (keep code changes confined to this file so detectors/controllers
     * are not modified). */
        std::vector<std::string> bottom_images; /* empty -> no bottom processing */
        bottom_images.clear();

    /* init nodes */
    ground_edge_node_init();
    obstacle_detector_node_init();
    gate_detector_node_init();
    bottom_cam_node_init();

    /* init controller */
    ControllerState ctrl_state;
    controller_init(&ctrl_state);

    /* main window — 3 panels only */
    cv::namedWindow("AF_8_V2 — CV Pipeline Test (3 panels)", cv::WINDOW_NORMAL);
    cv::resizeWindow("AF_8_V2 — CV Pipeline Test (3 panels)", 1400, 620);

    /* bottom camera window — separate popup */
    if (!bottom_images.empty()) {
        cv::namedWindow("Bottom Camera Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Bottom Camera Detection", 650, 600);
    }

    int idx         = 0;
    int bottom_idx  = 0;
    bool paused     = false;
    int total       = (int)images.size();
    int bottom_total = (int)bottom_images.size();

    while (true) {
        /* ═══════════════════════════════════════════════
         * MAIN WINDOW: load + rotate frame
         * ══════════════════════════════════════════════ */
        cv::Mat raw = cv::imread(images[idx]);
        if (raw.empty()) { idx = (idx + 1) % total; continue; }

        cv::Mat frame;
        cv::rotate(raw, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        int fw = frame.cols, fh = frame.rows;

        /* run pipeline (3 detectors) */
        FrameResults fr;
        memset(&fr, 0, sizeof(fr));

        ground_edge_node_process(frame.data, fw, fh, &fr.ground_edge);
        /* obstacle node takes original (rotates internally) */
        obstacle_detector_node_process(raw.data, raw.cols, raw.rows, &fr.obstacle);
        gate_detector_node_process(frame.data, fw, fh, &fr.gate);
        
        /* Also process bottom camera in sync (from Playground_Bottom) */
        if (!bottom_images.empty()) {
            cv::Mat bottom_raw = cv::imread(bottom_images[bottom_idx]);
            if (!bottom_raw.empty()) {
                bottom_cam_node_process(bottom_raw.data, bottom_raw.cols, bottom_raw.rows, &fr.bottom_cam);
            }
        }
        
        fr.valid = true;

        /* run controller */
        ControlOutput ctrl = controller_update(&ctrl_state, &fr, fw);

        /* build THREE side-by-side panels */
        cv::Mat left_panel   = draw_obstacle(frame, fr.obstacle);
        cv::Mat middle_panel = draw_gate(frame, fr.gate);
        cv::Mat right_panel  = draw_ctrl_panel(ctrl, fr, fw, fh);

        /* ground-edge confirmed lines overlay on left panel */
        for (int i = 0; i < fr.ground_edge.n_confirmed; i++) {
            auto &l = fr.ground_edge.confirmed[i];
            cv::line(left_panel,
                     {l.orig_p1.x, l.orig_p1.y}, {l.orig_p2.x, l.orig_p2.y},
                     {0, 255, 0}, 2, cv::LINE_AA);
        }

    /* no textual panel labels to keep panels clean */

        /* combine 3 panels */
        cv::Mat divider(fh, 3, CV_8UC3, cv::Scalar(60,60,60));
        cv::Mat canvas;
        cv::hconcat(std::vector<cv::Mat>{left_panel, divider,
                                         middle_panel, divider.clone(),
                                         right_panel}, canvas);

        /* HUD on top of combined canvas */
        draw_hud(canvas, fr, ctrl, idx, total, paused);

        cv::imshow("AF_8_V2 — CV Pipeline Test (3 panels)", canvas);

        /* ═══════════════════════════════════════════════
         * BOTTOM CAMERA WINDOW (parallel): load frame
         * ══════════════════════════════════════════════ */
        if (!bottom_images.empty()) {
            cv::Mat bottom_raw = cv::imread(bottom_images[bottom_idx]);
            if (!bottom_raw.empty()) {
                /* run bottom camera detector */
                BottomCamResult bcam;
                memset(&bcam, 0, sizeof(bcam));
                bottom_cam_node_process(bottom_raw.data, bottom_raw.cols, bottom_raw.rows, &bcam);

                /* visualize */
                cv::Mat bottom_vis = draw_bottom_cam(bottom_raw, bcam, bottom_raw.cols);

                /* add frame counter */
                // No textual bottom info; show visual-only

                cv::imshow("Bottom Camera Detection", bottom_vis);
            }

            /* advance bottom camera frame in sync */
            if (!paused) bottom_idx = (bottom_idx + 1) % bottom_total;
        }

        /* advance main frame */
        if (!paused) idx = (idx + 1) % total;

        int key = cv::waitKey(40) & 0xFF;   /* ~25 fps */
        if (key == 'q' || key == 27) break;
        if (key == ' ') paused = !paused;
        if (key == 'd') { paused = true;  idx = (idx + 1) % total;  bottom_idx = (bottom_idx + 1) % bottom_total; }
        if (key == 'a') { paused = true;  idx = (idx - 1 + total) % total;  bottom_idx = (bottom_idx - 1 + bottom_total) % bottom_total; }
    }

    cv::destroyAllWindows();
    return 0;
}
