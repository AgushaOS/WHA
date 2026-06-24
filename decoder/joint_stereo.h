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
    if (target_kbps < 128.0f) return 2;
    if (target_kbps < 160.0f) return 4;
    if (target_kbps < 190.0f) return 8;
    if (target_kbps < 224.0f) return 14; 
    if (target_kbps < 510.0f) return 16;
    return 999;
}

inline int get_r_bits(int band_idx) {
    if (band_idx < 4) return 6;
    if (band_idx < 8) return 5;
    return 4;
}

inline void compute_is_parameters_ex(const std::vector<float>& left,
                                     const std::vector<float>& right,
                                     std::vector<float>& Y,
                                     float& r,
                                     float& sqrtE,
                                     bool& inv_flag)
{
    const size_t n = left.size();

    double eL = 0.0, eR = 0.0, dot = 0.0;

    for (size_t i = 0; i < n; ++i) {
        eL  += left[i]  * left[i];
        eR  += right[i] * right[i];
        dot += left[i]  * right[i];
    }

    const double eps = 1e-12;

    double total = eL + eR;

    if (total < eps || eL < eps || eR < eps) {
        r = 0.5f;
        sqrtE = 0.0f;
        Y.assign(n, 0.0f);
        inv_flag = false;
        return;
    }

    r = static_cast<float>(eR / total);
    sqrtE = static_cast<float>(std::sqrt(total));

    double corr = dot / (std::sqrt(eL * eR) + eps);

    inv_flag = (corr < -0.2);

    double sqrt_eL = std::sqrt(eL);
    double sqrt_eR = std::sqrt(eR);
    double inv_sqrt_total = 1.0 / std::sqrt(total);

    std::vector<float> X(n);
    double eX = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double Rterm = inv_flag ? -sqrt_eR * right[i]
                                :  sqrt_eR * right[i];

        double x = (sqrt_eL * left[i] + Rterm) * inv_sqrt_total;
        X[i] = static_cast<float>(x);
        eX += x * x;
    }

    if (eX < eps) {
        Y.assign(n, 0.0f);
        return;
    }

    double inv_sqrt_eX = 1.0 / std::sqrt(eX);

    Y.resize(n);
    for (size_t i = 0; i < n; ++i) {
        Y[i] = static_cast<float>(X[i] * inv_sqrt_eX * sqrtE);
    }
}

inline void apply_is(const std::vector<float>& Y, float r, bool inv_flag,
                     std::vector<float>& left, std::vector<float>& right) {
    size_t n = Y.size();
    left.resize(n);
    right.resize(n);
    float gl = std::sqrt(1.0f - r);
    float gr = std::sqrt(r);
    for (size_t i = 0; i < n; ++i) {
        left[i] = Y[i] * gl;
        right[i] = Y[i] * (inv_flag ? -gr : gr);
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


inline bool use_mid_side(float El, float Er,
                         float Em, float Es,
                         bool enable_ms) 
{
    if (!enable_ms) return false;
    
    const float eps = 1e-12f;
    
    if (El + Er < 1e-8f) return false;
    
    float ratio = Es / (Em + eps);
    bool math_says_yes = (ratio < 0.35f);
    
    float min_e = std::min(El, Er);
    float max_e = std::max(El, Er);
    float energy_ratio = max_e / (min_e + eps);
    
    if (energy_ratio > 4.0f) return false;  
    
    return math_says_yes;
}


inline bool use_mid_side1(float El, float Er, float Em, float Es, bool enable_ms) {
    if (!enable_ms) return false;
    const float eps = 1e-12f;
    float prod_lr = El * Er;
    float prod_ms = Em * Es;
    return prod_ms < prod_lr - eps;
}

#endif // JOINT_STEREO_H