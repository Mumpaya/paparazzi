#include "gate_detector.h"
#include <opencv2/opencv.hpp>
#include "modules/computer_vision/cv.h" // Paparazzi CV wrapper
#include "abi.h"

// Your Hardcoded Python Params
namespace Config {
    float H_MIN = 2, H_MAX = 17, S_MIN = 134, V_MIN = 126;
    float ALPHA = 0.2f;
    float GATE_W = 1.0f, GATE_H = 1.0f;
}

// State variables (Smoothing)
float s_dist = 0, s_ox = 0, s_oy = 0, s_yaw = 0;

extern "C" {

void cv_gate_detect_init(void) {
    // Initialization logic
}

void cv_gate_detect_periodic(void) {
    // 1. Get image from Paparazzi video thread
    struct image_t* img = cv_get_new_image(); 
    if (!img) return;

    // 2. Convert to OpenCV Mat (Paparazzi usually provides YUV or RGB)
    cv::Mat frame(img->h, img->w, CV_8UC3, img->buf);
    
    // --- YOUR PYTHON LOGIC START ---
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(Config::H_MIN, Config::S_MIN, Config::V_MIN), 
                     cv::Scalar(Config::H_MAX, 255, 255), mask);

    // Finding contours and ApproxPolyDP (Just like your Python script)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours) {
        if (cv::contourArea(cnt) < 700) continue;
        
        std::vector<cv::Point2f> approx;
        double peri = cv::arcLength(cnt, true);
        cv::approxPolyDP(cnt, approx, 0.013 * peri, true);

        if (approx.size() == 4) {
            // Re-implementing your solve_gate_spatial (SolvePnP)
            std::vector<cv::Point3f> obj_pts = {
                {-Config::GATE_W/2,  Config::GATE_H/2, 0}, 
                { Config::GATE_W/2,  Config::GATE_H/2, 0}, 
                { Config::GATE_W/2, -Config::GATE_H/2, 0}, 
                {-Config::GATE_W/2, -Config::GATE_H/2, 0}
            };
            
            cv::Mat rvec, tvec;
            cv::Mat cam_matrix = (cv::Mat_<double>(3,3) << img->w, 0, img->w/2, 0, img->w, img->h/2, 0, 0, 1);
            
            if (cv::solvePnP(obj_pts, approx, cam_matrix, cv::Mat(), rvec, tvec)) {
                float raw_dist = cv::norm(tvec);
                
                // Alpha Smoothing
                s_dist = (Config::ALPHA * raw_dist) + (1.0f - Config::ALPHA) * s_dist;
                
                // 3. BROADCAST TO C-CORE (Using ABI)
                // This "shouts" the data so the C navigation scripts can hear it
                AbiSendMsgVISUAL_DETECTION(0, s_dist, s_ox, s_yaw);
                break; 
            }
        }
    }
}
}