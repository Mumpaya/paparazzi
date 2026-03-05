#include "modules/AF_8_V1/controller.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "modules/core/abi.h"
#include "state.h"
#include "generated/airframe.h"
#include <cmath>

// Default Values
float max_speed = 0.4f;
float heading_rate = 0.6f;
float danger_threshold = 8000.0f;
float gate_target_dist = 1.2f;

#define ABI_ID_POLE 1
#define ABI_ID_GATE 2

enum class FlightState { SEARCHING, FORWARD, AVOIDING_POLE, APPROACHING_GATE };
static FlightState current_state = FlightState::SEARCHING;

struct DetectedObj {
    int16_t x;
    int32_t quality;
    bool active;
};

static DetectedObj last_pole = {0, 0, false};
static DetectedObj last_gate = {0, 0, false};

// ABI Callbacks
static abi_event pole_ev, gate_ev;

static void pole_callback(uint8_t sender_id, int16_t x, int16_t y, int16_t w, int16_t h, int32_t qual, int16_t extra) {
    if (sender_id == ABI_ID_POLE) {
        last_pole = {x, qual, (extra > 0)};
    }
}

static void gate_callback(uint8_t sender_id, int16_t x, int16_t y, int16_t w, int16_t h, int32_t qual, int16_t extra) {
    if (sender_id == ABI_ID_GATE) {
        last_gate = {x, qual, (extra > 0)};
    }
}

extern "C" {

void controller_init(void) {
    AbiBindMsgVISUAL_DETECTION(ABI_ID_POLE, &pole_ev, pole_callback);
    AbiBindMsgVISUAL_DETECTION(ABI_ID_GATE, &gate_ev, gate_callback);
    current_state = FlightState::SEARCHING;
}

void controller_periodic(void) {
    // Only run if the pilot has switched to GUIDED mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) return;

    float img_w = (float)front_camera.output_size.w;
    bool pole_is_close = (last_pole.active && last_pole.quality > danger_threshold);

    // --- PRIORITY LOGIC (Subsumption) ---
    if (pole_is_close) {
        current_state = FlightState::AVOIDING_POLE;
    } else if (last_gate.active) {
        current_state = FlightState::APPROACHING_GATE;
    } else {
        current_state = FlightState::FORWARD;
    }

    // --- STEERING COMMANDS ---
    switch (current_state) {
        case FlightState::AVOIDING_POLE: {
            // Pole on left? Turn right. Pole on right? Turn left.
            float dir = ((float)last_pole.x / img_w < 0.5f) ? 1.0f : -1.0f;
            guidance_h_set_body_vel(0, 0); // Stop moving forward to avoid collision
            guidance_h_set_heading_rate(dir * heading_rate);
            break;
        }

        case FlightState::APPROACHING_GATE: {
            // P-Controller: Center the gate in the camera view
            float error_x = ((float)last_gate.x / img_w) - 0.5f;
            guidance_h_set_heading_rate(-error_x * 1.5f); // Turn toward gate
            guidance_h_set_body_vel(max_speed, 0);       // Fly toward it
            break;
        }

        case FlightState::FORWARD:
            guidance_h_set_body_vel(max_speed, 0);
            guidance_h_set_heading_rate(0);
            break;

        default:
            guidance_h_set_body_vel(0, 0);
            guidance_h_set_heading_rate(heading_rate); // Spin to find something
            break;
    }
}

void orange_avoider_guided_retreat(void) {
    // stub — retreat handled by controller_periodic state machine
}

} // extern "C"

