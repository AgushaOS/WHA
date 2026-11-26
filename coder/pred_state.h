#pragma once
#include <vector>
#include <cstdint>
#include <array>

static constexpr uint32_t SAME_MODE_PRED_MAX_AGE = 4;

struct ModeState {
    bool     ready  = false;
    uint32_t last_bi = 0;
    bool prev_scale_k_valid = false;
    uint8_t prev_scale_k0 = 0;
    uint8_t prev_scale_k1 = 0;

    std::vector<uint8_t> prev_k_sub0;
    std::vector<uint8_t> prev_k_sub1;
    std::vector<uint8_t> prev_mode_ms;

    std::vector<uint32_t> prev_scale0;
    std::vector<uint32_t> prev_scale1;
    std::vector<uint8_t>  prev_active0;
    std::vector<uint8_t>  prev_active1;
    std::vector<uint32_t> prev_sbr_rms;
    std::vector<uint8_t>  prev_sbr_noise;
    std::vector<uint8_t>  prev_sbr_shape;
    std::vector<uint8_t>  prev_sbr_valid;
    std::vector<std::array<int8_t, 4>> prev_is_coeff;
    std::vector<uint8_t> prev_is_valid;

    void ensure(int bands) {
        if ((int)prev_scale0.size() != bands) {
            prev_scale0.assign(bands, 0);
            prev_scale1.assign(bands, 0);
            prev_active0.assign(bands, 0);
            prev_active1.assign(bands, 0);
            prev_k_sub0.assign(bands, 0);
            prev_k_sub1.assign(bands, 0);
            prev_mode_ms.assign(bands, 0);
            prev_sbr_rms.assign(bands, 0);
            prev_sbr_noise.assign(bands, 0);
            prev_sbr_shape.assign(bands, 0);
            prev_sbr_valid.assign(bands, 0);
            prev_is_coeff.assign(bands, std::array<int8_t, 4>{});
            prev_is_valid.assign(bands, 0);
            prev_scale_k_valid = false;
            ready = false;
        }
    }
};

struct PredContext {
    ModeState long_state;
    ModeState short_state;
};