/*
 * cv_main.cpp — Central computer-vision pipeline for AF_8_V2
 *
 * One camera callback → converts YUV422 to BGR → runs:
 *   1. ground_edge_node       (on rotated frame)
 *   2. obstacle_detector_node (handles rotation internally)
 *   3. gate_detector_node     (on rotated frame)
 *   4. controller_update()    (heading / velocity state machine)
 *
 * Results land in the globals `frame_results` and `ctrl_output`
 * which can be read by any flight-plan or navigation module.
 *
 * ── Live video overlay ────────────────────────────────────────────
 * When compiled with AF_8_V2_DRAW=1 (the default) the function
 * draw_overlay() annotates the BGR frame with:
 *   • Obstacle column tints + flyzone bands
 *   • Gate crosshair + angle + distance text
 *   • Controller mode banner + compass needle
 * The annotated frame is then converted back to YUV422 and written
 * into img->buf so the Paparazzi video stream carries the overlay.
 */

#include "cv_main.h"
#include "ground_edge_node.h"
#include "obstacle_detector_node.h"
#include "gate_detector_node.h"
#include "bottom_cam_node.h"
#include "controller.h"

/* Paparazzi C headers */
extern "C" {
    #include "modules/computer_vision/cv.h"
    #include "firmwares/rotorcraft/guidance/guidance_h.h"
    #include "state.h"
}


#include "generated/airframe.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>

/* ── compile-time draw toggle ───────────────────────────────────── */
#ifndef AF_8_V2_DRAW
#define AF_8_V2_DRAW 1
#endif

/* ── global results ─────────────────────────────────────────────── */
volatile FrameResults  frame_results;
volatile ControlOutput ctrl_output;

/* ── persistent controller state ───────────────────────────────── */
static ControllerState s_ctrl;
static float s_heading_sp = 0.0f;

/* ── datalink-tunable settings ──────── */
int   af8_draw_overlay = 0;
float af8_k_yaw        = K_YAW;
float af8_v_std        = V_STD;
float af8_v_gate       = V_GATE;
float af8_bottom_cam_steer_threshold = 80.0f;  

/* ════════════════════════════════════════════════════════════════
 * OVERLAY DRAWING  (compiled only when AF_8_V2_DRAW == 1)
 * ═══════════════════════════════════════════════════════════════ */
#if AF_8_V2_DRAW

static void ov_txt(cv::Mat &img, const char *s, cv::Point pt,
                   cv::Scalar col, double scale = 0.42, int thick = 1)
{
    cv::putText(img, s, pt, cv::FONT_HERSHEY_SIMPLEX,
                scale, cv::Scalar(0,0,0), thick+2, cv::LINE_AA);
    cv::putText(img, s, pt, cv::FONT_HERSHEY_SIMPLEX,
                scale, col, thick, cv::LINE_AA);
}

static void draw_overlay(cv::Mat &canvas,
                         const FrameResults  &fr,
                         const ControlOutput &ctrl)
{
    int h = canvas.rows;
    int w = canvas.cols;

    /* ── obstacle: column tints ────────────────────────────────── */
    int n_cols = fr.obstacle.n_cols > 0 ? fr.obstacle.n_cols : OBSTACLE_N_COLS;
    float col_w = (float)w / n_cols;

    for (int c = 0; c < n_cols; c++) {
        bool is_obs = false;
        for (int i = 0; i < fr.obstacle.n_obstacle_cols; i++)
            if (fr.obstacle.obstacle_cols[i] == c) { is_obs = true; break; }

        int x1 = (int)(c * col_w);
        int x2 = (int)((c + 1) * col_w);
        cv::Mat roi = canvas(cv::Rect(x1, 0, x2 - x1, h));
        cv::Mat tint = roi.clone();
        tint = is_obs ? cv::Scalar(0, 0, 100) : cv::Scalar(0, 15, 0);
        cv::addWeighted(roi, 0.78, tint, 0.22, 0, roi);
    }

    /* ── obstacle: flyzone bands ────────────────────────────────── */
    static const cv::Scalar FZ_COL[3] = {
        {0,255,80}, {0,220,255}, {80,120,255}};

    for (int i = 0; i < fr.obstacle.n_flyzones; i++) {
        const auto &fz = fr.obstacle.flyzone[i];
        cv::Scalar col = FZ_COL[i];
        int bw = std::max(fz.right_px - fz.left_px, 1);
        cv::Mat band = canvas(cv::Rect(fz.left_px, 0, bw, h));
        cv::Mat tint2(band.size(), band.type(), col * 0.25);
        cv::addWeighted(band, 0.75, tint2, 0.25, 0, band);
        cv::line(canvas, {fz.left_px,  0}, {fz.left_px,  h}, col, 2);
        cv::line(canvas, {fz.right_px, 0}, {fz.right_px, h}, col, 2);
        cv::arrowedLine(canvas, {fz.center_px, h/2+24},
                        {fz.center_px, h/2-24}, col, 2, cv::LINE_AA, 0, 0.3);
        char fbuf[32];
        snprintf(fbuf, sizeof(fbuf), "#%d %.2f", i+1, fz.safety);
        ov_txt(canvas, fbuf, {fz.left_px+3, h/2+40+i*14}, col, 0.35);
    }

    /* ── ground-edge confirmed lines ───────────────────────────── */
    for (int i = 0; i < fr.ground_edge.n_confirmed; i++) {
        auto &l = fr.ground_edge.confirmed[i];
        cv::line(canvas,
                 {l.orig_p1.x, l.orig_p1.y},
                 {l.orig_p2.x, l.orig_p2.y},
                 {0, 220, 60}, 2, cv::LINE_AA);
    }

    /* ── gate crosshair + labels ────────────────────────────────── */
    if (fr.gate.detected) {
        int cx = fr.gate.cx_px;
        int cy = fr.gate.cy_px;
        cv::drawMarker(canvas, {cx, cy}, {0,255,255},
                       cv::MARKER_CROSS, 26, 2, cv::LINE_AA);
        cv::line(canvas, {cx, 0}, {cx, h}, {0,255,255}, 1, cv::LINE_AA);

        static const char *ang_str[] = {"GOOD","LEFT","RIGHT"};
        static const cv::Scalar ang_col[] = {{0,255,0},{0,165,255},{0,165,255}};
        ov_txt(canvas, ang_str[fr.gate.angle],
               {cx+6, cy-8}, ang_col[fr.gate.angle], 0.5, 1);
        if (fr.gate.dist_m > 0) {
            char dbuf[24];
            snprintf(dbuf, sizeof(dbuf), "%.2fm", fr.gate.dist_m);
            ov_txt(canvas, dbuf, {cx+6, cy+14}, {0,255,255}, 0.45);
        }
    }

    /* ── controller overlay ─────────────────────────────────────── */
    bool gl = (ctrl.mode == CTRL_MODE_GATELOCK);
    cv::Scalar mode_col = gl ? cv::Scalar(0,165,255) : cv::Scalar(0,200,60);

    /* small mode banner at bottom */
    cv::rectangle(canvas, {0, h-22}, {w, h}, mode_col*0.3, -1);
    static const char *action_short[] = {
        "STRAIGHT","FZ-STEER","GATE-STEER",
        "GATE-RECOV","GL-TRACK","GL-COAST"};
    char mbuf[64];
    snprintf(mbuf, sizeof(mbuf), "%s | %s | dYaw:%+.3f | v:%.2f",
             gl ? "GATELOCK" : "NORMAL",
             action_short[(int)ctrl.action],
             ctrl.delta_yaw_rad,
             ctrl.forward_vel);
    ov_txt(canvas, mbuf, {4, h-6}, mode_col, 0.38);

    /* compass needle — top-left corner */
    int cr = 28, ccx = 34, ccy = 38;
    cv::circle(canvas, {ccx, ccy}, cr, {60,60,60}, 1, cv::LINE_AA);
    float ang_r = (float)((ctrl.heading_error_pct * 1.8f - 90.0f) * M_PI / 180.0f);
    int nx = ccx + (int)(cr * 0.85f * cosf(ang_r));
    int ny = ccy + (int)(cr * 0.85f * sinf(ang_r));
    static const cv::Scalar action_cols[] = {
        {180,180,180},{0,220,60},{0,255,200},{0,165,255},{0,200,255},{80,80,255}};
    cv::arrowedLine(canvas, {ccx,ccy}, {nx,ny},
                    action_cols[(int)ctrl.action], 2, cv::LINE_AA, 0, 0.3);

    /* GATELOCK streak pip row */
    if (!gl) {
        int pip_y = ccy + cr + 8;
        for (int i = 0; i < GATELOCK_TRIGGER_FRAMES; i++) {
            cv::Scalar pc = (i < ctrl.good_streak)
                            ? cv::Scalar(0,200,60) : cv::Scalar(40,40,40);
            cv::circle(canvas, {4 + i*8, pip_y}, 3, pc, -1, cv::LINE_AA);
        }
    } else if (ctrl.coast_frames_left > 0) {
        char cbuf[24];
        snprintf(cbuf, sizeof(cbuf), "coast:%d", ctrl.coast_frames_left);
        ov_txt(canvas, cbuf, {4, ccy + cr + 12}, {80,80,255}, 0.38);
    }
}

#endif /* AF_8_V2_DRAW */

/* ════════════════════════════════════════════════════════════════
 * CAMERA CALLBACK
 * ═══════════════════════════════════════════════════════════════ */
static struct image_t *cv_main_cb(struct image_t *img,
                                  uint8_t camera_id __attribute__((unused)))
{
    if (!img || img->type != IMAGE_YUV422) return img;

    cv::Mat yuv(img->h, img->w, CV_8UC2, img->buf);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YUYV);

    cv::Mat rotated;
    cv::rotate(bgr, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);

    int rot_w = rotated.cols;
    int rot_h = rotated.rows;

    /* ── Run detectors ──────────────────────────────────────── */
    FrameResults local;
    memset(&local, 0, sizeof(local));

    ground_edge_node_process(rotated.data, rot_w, rot_h, &local.ground_edge);
    obstacle_detector_node_process(bgr.data, bgr.cols, bgr.rows, &local.obstacle);
    gate_detector_node_process(rotated.data, rot_w, rot_h, &local.gate);
    bottom_cam_node_process(bgr.data, bgr.cols, bgr.rows, &local.bottom_cam);
    local.valid = true;

    /* ── Run controller ─────────────────────────────────────── */
    ControlOutput cmd = controller_update(&s_ctrl, &local, rot_w);

    /* ── Publish globals ────────────────────────────────────── */
    *((FrameResults  *)&frame_results) = local;
    *((ControlOutput *)&ctrl_output)   = cmd;

#if AF_8_V2_DRAW
    draw_overlay(rotated, local, cmd);

    cv::Mat annotated_bgr;
    cv::rotate(rotated, annotated_bgr, cv::ROTATE_90_CLOCKWISE);

    cv::Mat yuv_tmp;
    cv::cvtColor(annotated_bgr, yuv_tmp, cv::COLOR_BGR2YUV);
    cv::Mat yuv_out(annotated_bgr.rows, annotated_bgr.cols, CV_8UC2);
    for (int r = 0; r < annotated_bgr.rows; r++) {
        for (int c = 0; c < annotated_bgr.cols; c += 2) {
            uint8_t y0 = yuv_tmp.at<cv::Vec3b>(r, c)[0];
            uint8_t u  = yuv_tmp.at<cv::Vec3b>(r, c)[1];
            uint8_t y1 = (c + 1 < annotated_bgr.cols)
                         ? yuv_tmp.at<cv::Vec3b>(r, c + 1)[0] : y0;
            uint8_t v  = yuv_tmp.at<cv::Vec3b>(r, c)[2];
            yuv_out.at<cv::Vec2b>(r, c)     = {y0, u};
            yuv_out.at<cv::Vec2b>(r, c + 1) = {y1, v};
        }
    }
    memcpy(img->buf, yuv_out.data,
           (size_t)img->w * (size_t)img->h * 2u);
#endif

    return img;  
}

/* ════════════════════════════════════════════════════════════════
 * PUBLIC API — called by Paparazzi module system
 * ═══════════════════════════════════════════════════════════════ */
extern "C" {

void cv_main_init(void)
{
    memset((void *)&frame_results, 0, sizeof(frame_results));
    memset((void *)&ctrl_output,   0, sizeof(ctrl_output));

    ground_edge_node_init();
    obstacle_detector_node_init();
    gate_detector_node_init();
    bottom_cam_node_init();
    controller_init(&s_ctrl);

    s_heading_sp = stateGetNedToBodyEulers_f()->psi;

    cv_add_to_device(&front_camera, cv_main_cb, 0, 0);
}

void cv_main_periodic(void)
{
    /* Only send guidance commands while already in guided mode */
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        return;  /* commands take effect next cycle */
    }

    ControlOutput cmd = *((ControlOutput *)&ctrl_output);

    s_heading_sp += cmd.delta_yaw_rad;
    FLOAT_ANGLE_NORMALIZE(s_heading_sp);

    guidance_h_set_heading(s_heading_sp);
    guidance_h_set_body_vel(cmd.forward_vel, 0.0f);
}


} /* extern "C" */
