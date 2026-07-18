#ifndef JOINT_STEREO_H
#define JOINT_STEREO_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>

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

inline int get_is_start_band(float target_kbps, int total_bands) {
    int base;
    if (target_kbps < 96.0f)      base = 2;
    else if (target_kbps < 128.0f) base = 3;
    else if (target_kbps < 160.0f) base = 4;
    else if (target_kbps < 190.0f) base = 8;
    else if (target_kbps < 224.0f) base = 14;
    else if (target_kbps < 510.0f) base = 16;
    else return 999;
    return base * total_bands / 16;
}

inline int get_r_bits(int /*band_idx*/) {
    return 4;
}

inline float compute_r_segment(const std::vector<float>& left, const std::vector<float>& right,
                                int start, int end) {
    double eL = 0.0, eR = 0.0;
    for (int i = start; i < end; ++i) {
        eL += left[i] * left[i];
        eR += right[i] * right[i];
    }
    const double eps = 1e-12;
    double total = eL + eR;
    if (total < eps) return 0.5f;
    return static_cast<float>(eR / total);
}

inline bool should_use_adaptive_panorama(const std::array<float, 4>& r_segments) {
    int changes = 0;
    for (int i = 1; i < 4; ++i) {
        if (std::abs(r_segments[i] - r_segments[i-1]) > 0.3f) {
            changes++;
        }
    }
    return (changes >= 2);
}

inline void compute_is_parameters_ex(const std::vector<float>& left,
                                     const std::vector<float>& right,
                                     std::vector<float>& Y,
                                     std::array<float, 4>& r_out,
                                     bool& inv_flag,
                                     bool& use_segmented,
                                     int band_idx,
                                     int total_bands)
{
    const size_t n = left.size();

    double eL_total = 0.0, eR_total = 0.0, dot_total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        eL_total  += left[i]  * left[i];
        eR_total  += right[i] * right[i];
        dot_total += left[i]  * right[i];
    }

    const double eps = 1e-12;
    double total = eL_total + eR_total;

    if (total < eps || eL_total < eps || eR_total < eps) {
        for (int s = 0; s < 4; ++s) r_out[s] = 0.5f;
        Y.assign(n, 0.0f);
        inv_flag = false;
        use_segmented = false;
        return;
    }

    double corr = dot_total / (std::sqrt(eL_total * eR_total) + eps);
    inv_flag = (corr < -0.2);

    int segmented_threshold = 6 * total_bands / 16;
    use_segmented = (band_idx < segmented_threshold);

    if (use_segmented) {
        const int seg_size = n / 4;
        Y.resize(n);

        for (int s = 0; s < 4; ++s) {
            int start = s * seg_size;
            int end = (s == 3) ? n : start + seg_size;
            int len = end - start;

            double seg_eL = 0.0, seg_eR = 0.0;
            for (int i = start; i < end; ++i) {
                seg_eL += left[i] * left[i];
                seg_eR += right[i] * right[i];
            }

            double seg_total = seg_eL + seg_eR;

            if (seg_total < eps) {
                r_out[s] = 0.5f;
                for (int i = start; i < end; ++i) Y[i] = 0.0f;
                continue;
            }
            r_out[s] = static_cast<float>(seg_eR / seg_total);

            double sqrt_eL = std::sqrt(seg_eL);
            double sqrt_eR = std::sqrt(seg_eR);
            double inv_sqrt_total = 1.0 / std::sqrt(seg_total);

            double eX = 0.0;
            std::vector<float> X_seg(len);

            for (int i = 0; i < len; ++i) {
                int idx = start + i;
                double Rterm = inv_flag ? -sqrt_eR * right[idx] : sqrt_eR * right[idx];
                double x = (sqrt_eL * left[idx] + Rterm) * inv_sqrt_total;
                X_seg[i] = static_cast<float>(x);
                eX += x * x;
            }

            if (eX < eps) {
                for (int i = start; i < end; ++i) Y[i] = 0.0f;
            } else {
                double inv_sqrt_eX = 1.0 / std::sqrt(eX);
                double target_energy = std::sqrt(seg_total);
                for (int i = 0; i < len; ++i) {
                    Y[start + i] = static_cast<float>(X_seg[i] * inv_sqrt_eX * target_energy);
                }
            }
        }
    } else {
        use_segmented = false;
        float r = static_cast<float>(eR_total / total);
        for (int s = 0; s < 4; ++s) r_out[s] = r;

        double sqrtE = std::sqrt(total);
        double sqrt_eL = std::sqrt(eL_total);
        double sqrt_eR = std::sqrt(eR_total);
        double inv_sqrt_total = 1.0 / std::sqrt(total);

        std::vector<float> X(n);
        double eX = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double Rterm = inv_flag ? -sqrt_eR * right[i] : sqrt_eR * right[i];
            double x = (sqrt_eL * left[i] + Rterm) * inv_sqrt_total;
            X[i] = static_cast<float>(x);
            eX += x * x;
        }

        if (eX < eps) {
            Y.assign(n, 0.0f);
        } else {
            double inv_sqrt_eX = 1.0 / std::sqrt(eX);
            Y.resize(n);
            for (size_t i = 0; i < n; ++i) {
                Y[i] = static_cast<float>(X[i] * inv_sqrt_eX * sqrtE);
            }
        }
    }
}

inline void apply_is(const std::vector<float>& Y,
                     const std::array<float, 4>& r_vals,
                     bool inv_flag,
                     bool use_segmented,
                     std::vector<float>& left,
                     std::vector<float>& right) {
    size_t n = Y.size();
    left.resize(n);
    right.resize(n);

    if (use_segmented && n >= 16) {
        int seg_size = n / 4;
        for (int s = 0; s < 4; ++s) {
            float r = r_vals[s];
            float gl = std::sqrt(1.0f - r);
            float gr = std::sqrt(r);

            int start = s * seg_size;
            int end = (s == 3) ? n : start + seg_size;

            for (int i = start; i < end; ++i) {
                left[i] = Y[i] * gl;
                right[i] = Y[i] * (inv_flag ? -gr : gr);
            }
        }
    } else {
        float r = r_vals[0];
        float gl = std::sqrt(1.0f - r);
        float gr = std::sqrt(r);
        for (size_t i = 0; i < n; ++i) {
            left[i] = Y[i] * gl;
            right[i] = Y[i] * (inv_flag ? -gr : gr);
        }
    }
}

inline uint32_t quantize_r(float r, int bits) {
    float r_clamped = std::clamp(r, 0.0f, 1.0f);
    float theta = std::asin(std::sqrt(r_clamped));
    int max_val = (1 << bits) - 1;
    float theta_norm = theta / (float)(M_PI * 0.5f);
    return static_cast<uint32_t>(std::clamp(theta_norm, 0.0f, 1.0f) * max_val + 0.5f);
}

inline float dequantize_r(uint32_t q, int bits) {
    int max_val = (1 << bits) - 1;
    float theta_norm = (float)q / max_val;
    float theta = theta_norm * (float)(M_PI * 0.5f);
    float s = std::sin(theta);
    return s * s;
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

inline bool use_mid_side(float El, float Er, float Em, float Es, bool enable_ms, float /*target_kbps*/) {
    if (!enable_ms) return false;
    const float eps = 1e-12f;
    float min_e = std::min(El, Er);
    float max_e = std::max(El, Er);
    float energy_ratio = max_e / (min_e + eps);
    if (energy_ratio > 4.0f) return false;
    if (Es > Em * 0.5f) return false;
    float prod_lr = El * Er;
    float prod_ms = Em * Es;
    return prod_ms < prod_lr * 0.95f;
}

#endif // JOINT_STEREO_H