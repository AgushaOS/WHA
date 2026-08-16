#pragma once
#include <vector>
#include <cstdint>

static constexpr uint32_t SAME_MODE_PRED_MAX_AGE = 4;

struct ModeState {
    bool     ready  = false;
    uint32_t last_bi = 0;

    std::vector<uint32_t> prev_scale0;
    std::vector<uint32_t> prev_scale1;
    std::vector<uint8_t>  prev_active0;
    std::vector<uint8_t>  prev_active1;
    std::vector<uint32_t> prev_sbr_rms;
    std::vector<uint8_t>  prev_sbr_noise;
    std::vector<uint8_t>  prev_sbr_valid;

    void ensure(int bands) {
        if ((int)prev_scale0.size() != bands) {
            prev_scale0.assign(bands, 0);
            prev_scale1.assign(bands, 0);
            prev_active0.assign(bands, 0);
            prev_active1.assign(bands, 0);
            prev_sbr_rms.assign(bands, 0);
            prev_sbr_noise.assign(bands, 0);
            prev_sbr_valid.assign(bands, 0);
            ready = false;
        }
    }
};

struct PredContext {
    ModeState long_state;
    ModeState short_state;
};