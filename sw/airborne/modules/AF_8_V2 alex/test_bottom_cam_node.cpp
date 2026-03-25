/*
 * test_bottom_cam_node.cpp — Standalone test for bottom camera detector
 *
 * Reads all .jpg images from Playground_Bottom/, runs the bottom_cam_node detector
 * and visualizes the results with arrow/centroid/magnitude overlay.
 *
 * Controls:
 *   q / ESC       — quit
 *   SPACE         — pause/resume
 *   d             — step forward (when paused)
 *   a             — step backward (when paused)
 */

#include "bottom_cam_node.h"
#include "detection_types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#define DIR_SEP "\\"
#else
#include <dirent.h>
#define DIR_SEP "/"
#endif

/* ══════════════════════════════════════════════════════════════════
 * Utility: Collect all .jpg images from a folder
 * ════════════════════════════════════════════════════════════════ */
static std::vector<std::string> collect_images(const std::string &folder)
{
    std::vector<std::string> images;

#ifdef _WIN32
    WIN32_FIND_DATAA ffd;
    HANDLE hFind = FindFirstFileA((folder + "\\*.jpg").c_str(), &ffd);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                images.push_back(folder + "\\" + ffd.cFileName);
            }
        } while (FindNextFileA(hFind, &ffd));
        FindClose(hFind);
    }
#else
    DIR *dir = opendir(folder.c_str());
    if (!dir) return images;

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.length() >= 4 && name.substr(name.length() - 4) == ".jpg") {
            images.push_back(folder + "/" + name);
        }
    }
    closedir(dir);
#endif

    std::sort(images.begin(), images.end());
    return images;
}

/* ══════════════════════════════════════════════════════════════════
 * Utility: Simple text rendering
 * ════════════════════════════════════════════════════════════════ */
static void txt(cv::Mat &canvas, const std::string &text, cv::Point pt,
                cv::Scalar color, double fontScale)
{
    cv::putText(canvas, text, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, 1, cv::LINE_AA);
}

/* ══════════════════════════════════════════════════════════════════
 * Draw bottom camera detection visualization
 * ════════════════════════════════════════════════════════════════ */
static cv::Mat draw_bottom_cam_test(const cv::Mat &frame,
                                    const BottomCamResult &bcam,
                                    int fw, int fh)
{
    cv::Mat canvas = frame.clone();

    /* frame center (blue dot) */
    cv::Point frame_center(fw / 2, fh / 2);
    cv::circle(canvas, frame_center, 5, {255, 0, 0}, -1, cv::LINE_AA);

    if (bcam.detected && bcam.magnitude > 0.1f) {
        /* detected centroid (cyan circle) */
        cv::Point centroid(bcam.cx_px, bcam.cy_px);
        cv::circle(canvas, centroid, 8, {255, 255, 0}, 2, cv::LINE_AA);

        /* arrow from frame center to centroid */
        cv::Scalar arrow_color = bcam.over_edge ? cv::Scalar(0, 0, 255)      /* red if over edge */
                                                 : cv::Scalar(0, 255, 0);    /* green if OK */

        cv::arrowedLine(canvas, frame_center, centroid, arrow_color, 2,
                        cv::LINE_AA, 0, 0.2);

        /* status text */
        char status[256];
        snprintf(status, sizeof(status),
                 "cx=%.1f cy=%.1f | mag=%.1f | good=%.2f | edge=%s",
                 (float)bcam.cx_px, (float)bcam.cy_px,
                 bcam.magnitude, bcam.goodness,
                 bcam.over_edge ? "YES" : "NO");

        txt(canvas, status, {10, 30}, {0, 255, 0}, 0.4);
    } else {
        txt(canvas, "NO DETECTION", {10, 30}, {100, 100, 100}, 0.4);
    }

    return canvas;
}

/* ══════════════════════════════════════════════════════════════════
 * HUD overlay (frame number, pause state)
 * ════════════════════════════════════════════════════════════════ */
static void draw_hud_bottom_cam(cv::Mat &canvas, int idx, int total, bool paused)
{
    std::string frame_info = "Frame: " + std::to_string(idx + 1) + " / " + std::to_string(total);
    txt(canvas, frame_info, {10, (int)canvas.rows - 20}, {100, 100, 255}, 0.5);

    if (paused) {
        txt(canvas, "PAUSED", {(int)canvas.cols - 150, (int)canvas.rows - 20},
            {0, 0, 255}, 0.6);
    }

    /* controls hint */
    txt(canvas, "q/ESC:quit | SPACE:pause | d:next | a:prev", {10, 20},
        {150, 150, 150}, 0.35);
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* path to bottom camera playground — can override via argv[1] */
    std::string folder = argc > 1 ? argv[1]
        : "sw/airborne/modules/AF_8_V2/Playground_Bottom/20260313-105015";

    auto images = collect_images(folder);
    if (images.empty()) {
        fprintf(stderr, "No images found in %s\n", folder.c_str());
        return 1;
    }
    printf("Found %zu images in %s\n", images.size(), folder.c_str());

    /* init bottom camera node */
    bottom_cam_node_init();

    cv::namedWindow("Bottom Camera Node Test", cv::WINDOW_NORMAL);
    cv::resizeWindow("Bottom Camera Node Test", 800, 600);

    int idx    = 0;
    bool paused = false;
    int total   = (int)images.size();

    while (true) {
        /* load frame */
        cv::Mat raw = cv::imread(images[idx]);
        if (raw.empty()) {
            fprintf(stderr, "Failed to load: %s\n", images[idx].c_str());
            idx = (idx + 1) % total;
            continue;
        }

        int fw = raw.cols, fh = raw.rows;

        /* run bottom camera detector */
        BottomCamResult bcam;
        memset(&bcam, 0, sizeof(bcam));

        bottom_cam_node_process(raw.data, fw, fh, &bcam);

        /* visualize result */
        cv::Mat canvas = draw_bottom_cam_test(raw, bcam, fw, fh);

        /* add HUD */
        draw_hud_bottom_cam(canvas, idx, total, paused);

        cv::imshow("Bottom Camera Node Test", canvas);

        /* advance frame */
        if (!paused) idx = (idx + 1) % total;

        int key = cv::waitKey(40) & 0xFF;   /* ~25 fps */
        if (key == 'q' || key == 27) break;
        if (key == ' ') paused = !paused;
        if (key == 'd') { paused = true;  idx = (idx + 1) % total; }
        if (key == 'a') { paused = true;  idx = (idx - 1 + total) % total; }
    }

    cv::destroyAllWindows();
    return 0;
}
