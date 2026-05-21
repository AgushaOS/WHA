#ifndef JOINT_STEREO_H
#define JOINT_STEREO_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

inline int get_is_start_band(float target_kbps) {
    if (target_kbps < 128.0f) return 3;
    if (target_kbps < 160.0f) return 4;
    if (target_kbps < 190.0f) return 8;
    return 999;
}

inline std::pair<std::vector<float>, std::vector<float>> mid_side(const std::vector<float>& left, const std::vector<float>& right) {
    size_t n = left.size();
    std::vector<float> mid(n), side(n);
    for (size_t i = 0; i < n; ++i) {
        mid[i] = (left[i] + right[i]) * 0.5f;
        side[i] = (left[i] - right[i]) * 0.5f;
    }
    return {mid, side};
}

inline bool use_mid_side(float El, float Er, float Em, float Es, bool enable_ms) {
    if (!enable_ms) return false;
    const float eps = 1e-12f;
    float prod_lr = El * Er;
    float prod_ms = Em * Es;
    return prod_ms < prod_lr - eps;
}

inline float compute_is_scale(const std::vector<std::vector<float>>& left_coeffs,
                              const std::vector<std::vector<float>>& right_coeffs,
                              int start_band, int band_count) {
    float El = 0.0f, Er = 0.0f;
    for (int i = start_band; i < band_count; ++i) {
        for (float v : left_coeffs[i]) El += v*v;
        for (float v : right_coeffs[i]) Er += v*v;
    }
    const float eps = 1e-12f;
    float scale = std::sqrt(Er / (El + eps));
    return std::clamp(scale, 0.0f, 4.0f);
}

inline uint8_t quantize_is_scale(float scale) {
    float idx_f = ((scale - 0.0f) / 4.0f) * 255.0f;
    return (uint8_t)std::clamp(std::round(idx_f), 0.0f, 255.0f);
}

#endif // JOINT_STEREO_H