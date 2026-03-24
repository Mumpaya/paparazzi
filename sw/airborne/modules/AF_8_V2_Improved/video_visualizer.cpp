/*
 * video_visualizer.cpp — Real detector output visualization
 *
 * Reads test images from Playground/ folder
 * Calls actual detector nodes (obstacle, gate, bottom_cam)
 * Displays real detector outputs overlayed on video frames
 *
 * Compile:  make video_visualizer
 * Run:      ./video_visualizer
 *
 * Controls:
 *   SPACE — pause/play
 *   A/D   — previous/next frame (when paused)
 *   Q     — quit
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "detection_types.h"
#include "obstacle_detector_node.h"
#include "gate_detector_node.h"
#include "ground_edge_node.h"
#include "bottom_cam_node.h"

#include <algorithm>
#include <dirent.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

/* Get list of image files from directory */
std::vector<std::string> getImageFiles(const std::string &dir_path)
{
    std::vector<std::string> images;
    
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) {
        return images;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() > 4) {
            std::string ext = filename.substr(filename.size() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png") {
                images.push_back(dir_path + "/" + filename);
            }
        }
    }
    closedir(dir);
    
    std::sort(images.begin(), images.end());
    return images;
}

/* Draw actual obstacle detector output */
void drawObstacleOutput(cv::Mat &img, const ObstacleResult &obs)
{
    if (img.empty()) return;
    int h = img.rows, w = img.cols;
    int n_cols = obs.n_cols > 0 ? obs.n_cols : OBSTACLE_N_COLS;
    float col_w = (float)w / n_cols;

    /* semi-transparent column tint: red = obstacle, green = free */
    for (int c = 0; c < n_cols; c++) {
        bool is_obs = false;
        for (int i = 0; i < obs.n_obstacle_cols; i++)
            if (obs.obstacle_cols[i] == c) { is_obs = true; break; }

        int x1 = (int)(c * col_w);
        int x2 = (int)((c + 1) * col_w);
        cv::Mat roi = img(cv::Rect(x1, 0, x2 - x1, h));
        cv::Mat tint = roi.clone();
        if (is_obs)
            tint = cv::Scalar(0, 0, 120);       /* red tint */
        else
            tint = cv::Scalar(0, 20, 0);        /* subtle green tint */
        cv::addWeighted(roi, 0.7, tint, 0.3, 0, roi);

        /* column divider */
        cv::line(img, {x2, 0}, {x2, h}, {40,40,40}, 1);
    }

    /* flyzones overlays */
    static const cv::Scalar ZONE_COL[3] = {
        {0, 255, 80}, {0, 220, 255}, {80, 120, 255}};

    for (int i = 0; i < obs.n_flyzones; i++) {
        const auto &fz = obs.flyzone[i];
        cv::Scalar col = ZONE_COL[i];

        /* bright vertical band */
        cv::Mat band = img(cv::Rect(fz.left_px, 0,
                                    std::max(fz.right_px - fz.left_px, 1), h));
        cv::Mat tint2(band.size(), band.type(), col * 0.3);
        cv::addWeighted(band, 0.7, tint2, 0.3, 0, band);

        /* boundary lines */
        cv::line(img, {fz.left_px,  0}, {fz.left_px,  h}, col, 2);
        cv::line(img, {fz.right_px, 0}, {fz.right_px, h}, col, 2);

        /* centre arrow */
        int cx = fz.center_px;
        cv::arrowedLine(img, {cx, h/2 + 30}, {cx, h/2 - 30}, col, 3, cv::LINE_AA, 0, 0.35);
    }
}

/* Draw actual gate detector output */
void drawGateOutput(cv::Mat &img, const GateResult &gate)
{
    if (img.empty()) return;
    int h = img.rows, w = img.cols;

    if (!gate.detected) {
        cv::putText(img, "No Gate", {w/2 - 40, h/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,255}, 2, cv::LINE_AA);
        return;
    }

    int cx = gate.cx_px;
    int cy = gate.cy_px;

    /* crosshair at gate centre */
    cv::drawMarker(img, {cx, cy}, {0,255,255}, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);

    /* vertical centre line */
    cv::line(img, {cx, 0}, {cx, h}, {0,255,255}, 1, cv::LINE_AA);

    /* approach angle */
    static const char *angle_str[] = {"GOOD", "LEFT", "RIGHT"};
    static const cv::Scalar angle_col[] = {{0,255,0},{0,165,255},{0,165,255}};
    cv::Scalar acol = angle_col[gate.angle];

    char buf[64];
    snprintf(buf, sizeof(buf), "%s", angle_str[gate.angle]);
    cv::putText(img, buf, {cx + 8, cy - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.55, acol, 1, cv::LINE_AA);

    if (gate.angle == GATE_ANGLE_GOOD && gate.dist_m > 0) {
        snprintf(buf, sizeof(buf), "%.2f m", gate.dist_m);
        cv::putText(img, buf, {cx + 8, cy + 16}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,255}, 1, cv::LINE_AA);
    }
}

/* Draw ground edge detector output */
void drawGroundEdgeOutput(cv::Mat &img, const GroundEdgeResult &ge)
{
    if (img.empty()) return;

    for (int i = 0; i < ge.n_confirmed; i++) {
        auto &l = ge.confirmed[i];
        cv::line(img, {l.orig_p1.x, l.orig_p1.y}, {l.orig_p2.x, l.orig_p2.y},
                 {0, 255, 0}, 2, cv::LINE_AA);
    }
}

int main(int argc, char *argv[])
{
    std::string playground_path = argc > 1 ? argv[1] : "sw/airborne/modules/AF_8_V2_Improved/Playground";
    
    std::cout << "\n╔═══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  AF_8_V2 Real Detector Output Visualizer    ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════╝\n" << std::endl;
    
    /* Get image files */
    std::vector<std::string> images = getImageFiles(playground_path);
    
    if (images.empty()) {
        std::cerr << "❌ No images found in: " << playground_path << std::endl;
        return 1;
    }
    
    std::cout << "✅ Found " << images.size() << " images\n" << std::endl;
    
    /* Disable Qt backend to avoid threading issues in headless mode */
    setenv("QT_QPA_PLATFORM", "offscreen", 1);
    
    /* Initialize detector nodes */
    ground_edge_node_init();
    obstacle_detector_node_init();
    gate_detector_node_init();
    
    /* Check if running in GUI mode or headless */
    /* FORCE HEADLESS MODE - GUI display is not functional */
    bool headless_mode = true;  /* FORCED headless */
    
    int max_frames_headless = 10;
    int frames_saved = 0;
    
    if (headless_mode) {
        std::cout << "📡 Headless mode - will save first " << max_frames_headless << " frames as PNG files\n" << std::endl;
    }
    
    /* Main visualization loop */
    int idx = 0;
    bool paused = false;
    int total = images.size();
    
    while (true) {
        /* Load image */
        cv::Mat raw = cv::imread(images[idx]);
        if (raw.empty()) {
            idx = (idx + 1) % total;
            continue;
        }
        
        /* Rotate image */
        cv::Mat frame;
        cv::rotate(raw, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        int fw = frame.cols, fh = frame.rows;
        
        /* Run detectors on actual data */
        GroundEdgeResult ground_edge;
        ObstacleResult obstacle;
        GateResult gate;
        
        memset(&ground_edge, 0, sizeof(ground_edge));
        memset(&obstacle, 0, sizeof(obstacle));
        memset(&gate, 0, sizeof(gate));
        
        /* Call actual detector nodes */
        ground_edge_node_process(frame.data, fw, fh, &ground_edge);
        obstacle_detector_node_process(raw.data, raw.cols, raw.rows, &obstacle);
        gate_detector_node_process(frame.data, fw, fh, &gate);
        
        /* Create visualization by drawing real detector outputs */
        cv::Mat vis_frame = frame.clone();
        
        drawObstacleOutput(vis_frame, obstacle);
        drawGateOutput(vis_frame, gate);
        drawGroundEdgeOutput(vis_frame, ground_edge);
        
        /* HUD info */
        cv::Mat overlay = vis_frame.clone();
        cv::rectangle(overlay, {0, 0}, {400, 90}, cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame);
        
        char frame_info[256];
        snprintf(frame_info, sizeof(frame_info),
                 "Frame %d/%d | Obstacle: %d flyzones, %d cols | Gate: %s | Edge: %d lines",
                 idx + 1, total,
                 obstacle.n_flyzones, obstacle.n_obstacle_cols,
                 gate.detected ? "DETECTED" : "not seen",
                 ground_edge.n_confirmed);
        
        cv::putText(vis_frame, frame_info, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {0, 255, 255}, 1, cv::LINE_AA);
        
        if (!headless_mode) {
            cv::putText(vis_frame, paused ? "[PAUSED]" : "[PLAYING]", {8, 52},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        paused ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            
            cv::putText(vis_frame, "SPACE=pause  A/D=step  Q=quit", {8, 80},
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, {180, 180, 180}, 1, cv::LINE_AA);
            
            /* Display in GUI mode */
            cv::imshow("AF_8_V2 Real Detector Output", vis_frame);
            
            /* Keyboard input */
            int key = cv::waitKey(paused ? 0 : 30) & 0xFF;
            
            if (key == 'q' || key == 27) break;           /* quit */
            if (key == ' ') paused = !paused;             /* pause/play */
            if (key == 'd' && paused) idx = (idx + 1) % total;  /* next frame */
            if (key == 'a' && paused) idx = (idx - 1 + total) % total;  /* prev frame */
            
            if (!paused) idx = (idx + 1) % total;
        } else {
            /* Headless mode: save first N frames */
            if (frames_saved < max_frames_headless) {
                char filename[128];
                snprintf(filename, sizeof(filename), "viz_frame_real_%04d.png", frames_saved + 1);
                cv::imwrite(filename, vis_frame);
                std::cout << "✓ Frame " << (frames_saved + 1) << "/" << max_frames_headless 
                          << " → " << filename << std::endl;
                frames_saved++;
            }
            
            idx = (idx + 1) % total;
            
            /* Exit after saving N frames */
            if (frames_saved >= max_frames_headless) break;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "\n✅ Visualization complete!\n" << std::endl;
    
    return 0;
}
