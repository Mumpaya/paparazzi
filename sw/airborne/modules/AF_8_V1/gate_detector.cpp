#include "gate_detector.h"
#include <opencv2/opencv.hpp>
extern "C" {
#include "modules/computer_vision/cv.h"
}
#include "modules/core/abi.h"

#ifndef GATE_DETECTOR_ABI_ID
#define GATE_DETECTOR_ABI_ID 2
#endif

#ifndef GATE_DETECTOR_CAMERA
#define GATE_DETECTOR_CAMERA front_camera
#endif

namespace Config {
    float H_MIN = 2,   H_MAX = 17;
    float S_MIN = 134, V_MIN = 126;
    float ALPHA  = 0.2f;
    float GATE_W = 1.0f, GATE_H = 1.0f;
}

static float s_dist = 0, s_ox = 0, s_oy = 0;
static int16_t gate_found = 0;

static struct image_t *gate_detect_cb(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    if (!img || img->type != IMAGE_YUV422) return NULL;

    // Convert YUV422 buffer to OpenCV BGR
    cv::Mat yuv(img->h, img->w, CV_8UC2, img->buf);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YUYV);

    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv,
        cv::Scalar(Config::H_MIN, Config::S_MIN, Config::V_MIN),
        cv::Scalar(Config::H_MAX, 255, 255),
        mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    gate_found = 0;

    for (auto& cnt : contours) {
        if (cv::contourArea(cnt) < 700) continue;

        std::vector<cv::Point2f> approx;
        double peri = cv::arcLength(cnt, true);
        cv::approxPolyDP(cnt, approx, 0.013 * peri, true);

        if (approx.size() == 4) {
            std::vector<cv::Point3f> obj_pts = {
                {-Config::GATE_W/2,  Config::GATE_H/2, 0},
                { Config::GATE_W/2,  Config::GATE_H/2, 0},
                { Config::GATE_W/2, -Config::GATE_H/2, 0},
                {-Config::GATE_W/2, -Config::GATE_H/2, 0}
            };

            cv::Mat rvec, tvec;
            cv::Mat cam_matrix = (cv::Mat_<double>(3,3)
                << img->w, 0, img->w/2.0,
                   0, img->w, img->h/2.0,
                   0, 0, 1);

            if (cv::solvePnP(obj_pts, approx, cam_matrix, cv::Mat(), rvec, tvec)) {
                float raw_dist = (float)cv::norm(tvec);
                float raw_ox   = (float)tvec.at<double>(0);
                float raw_oy   = (float)tvec.at<double>(1);

                s_dist = Config::ALPHA * raw_dist + (1.f - Config::ALPHA) * s_dist;
                s_ox   = Config::ALPHA * raw_ox   + (1.f - Config::ALPHA) * s_ox;
                s_oy   = Config::ALPHA * raw_oy   + (1.f - Config::ALPHA) * s_oy;

                gate_found = 1;
                break;
            }
        }
    }

    return NULL;
}

extern "C" {

void gate_detector_init(void)
{
    cv_add_to_device(&GATE_DETECTOR_CAMERA, gate_detect_cb, 0, 0);
}

void gate_detector_periodic(void)
{
    // pixel_x = lateral offset (s_ox scaled to pixels)
    // quality = distance in mm (s_dist * 1000)
    // extra   = gate_found flag
    int16_t px_x = (int16_t)(s_ox * 100.f);
    int16_t px_y = (int16_t)(s_oy * 100.f);
    int32_t qual = (int32_t)(s_dist * 1000.f);
    AbiSendMsgVISUAL_DETECTION(GATE_DETECTOR_ABI_ID,
                                px_x, px_y, 0, 0,
                                qual, gate_found);
}

} // extern "C"