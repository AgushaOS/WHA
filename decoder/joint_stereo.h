#ifndef JOINT_STEREO_H
#define JOINT_STEREO_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

inline void ms_to_lr(const std::vector<float>& mid, const std::vector<float>& side,
                     std::vector<float>& left, std::vector<float>& right) {
    size_t n = mid.size();
    left.resize(n);
    right.resize(n);
    for (size_t i = 0; i < n; ++i) {
        left[i] = mid[i] + side[i];
        right[i] = mid[i] - side[i];
    }
}

inline int get_is_start_band(float target_kbps) {
    if (target_kbps < 128.0f) return 4;
    if (target_kbps < 160.0f) return 4;
    if (target_kbps < 190.0f) return 8;
    return 999;
}

inline int get_r_bits(int band_idx) {
    if (band_idx < 4) return 6;
    if (band_idx < 8) return 5;
    return 4;
}

inline void compute_is_parameters(const std::vector<float>& left,
                                  const std::vector<float>& right,
                                  std::vector<float>& Y,
                                  float& r,
                                  float& sqrtE) {
    size_t n = left.size();
    double eL = 0.0, eR = 0.0;
    for (size_t i = 0; i < n; ++i) {
        eL += left[i] * left[i];
        eR += right[i] * right[i];
    }
    double total = eL + eR;
    if (total < 1e-12) {
        r = 0.5f;
        sqrtE = 0.0f;
        Y.assign(n, 0.0f);
        return;
    }
    r = static_cast<float>(eR / total);
    sqrtE = static_cast<float>(std::sqrt(total));
    double sqrt_eL = std::sqrt(eL);
    double sqrt_eR = std::sqrt(eR);
    double inv_sqrt_total = 1.0 / std::sqrt(total);

    double eX = 0.0;
    std::vector<float> X(n);
    for (size_t i = 0; i < n; ++i) {
        X[i] = static_cast<float>((sqrt_eL * left[i] + sqrt_eR * right[i]) * inv_sqrt_total);
        eX += X[i] * X[i];
    }

    if (eX > 1e-12) {
        double inv_sqrt_eX = 1.0 / std::sqrt(eX);
        Y.resize(n);
        for (size_t i = 0; i < n; ++i)
            Y[i] = static_cast<float>(X[i] * inv_sqrt_eX * sqrtE);
    } else {
        Y.assign(n, 0.0f);
    }
}

inline void apply_is(const std::vector<float>& Y, float r,
                     std::vector<float>& left, std::vector<float>& right) {
    size_t n = Y.size();
    left.resize(n);
    right.resize(n);
    float gl = std::sqrt(1.0f - r);
    float gr = std::sqrt(r);
    for (size_t i = 0; i < n; ++i) {
        left[i] = Y[i] * gl;
        right[i] = Y[i] * gr;
    }
}

inline uint32_t quantize_r(float r, int bits) {
    int max_val = (1 << bits) - 1;
    return static_cast<uint32_t>(std::clamp(r, 0.0f, 1.0f) * max_val + 0.5f);
}

inline float dequantize_r(uint32_t q, int bits) {
    return q / (float)((1 << bits) - 1);
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

#endif // JOINT_STEREO_H