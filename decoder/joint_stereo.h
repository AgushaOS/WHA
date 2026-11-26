#ifndef JOINT_STEREO_H
#define JOINT_STEREO_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline void ms_to_lr(const std::vector<float>& mid, const std::vector<float>& side,
                     std::vector<float>& left, std::vector<float>& right)
{
    size_t n = mid.size();
    left.resize(n);
    right.resize(n);
    for (size_t i = 0; i < n; ++i) {
        left[i]  = mid[i] + side[i];
        right[i] = mid[i] - side[i];
    }
}

inline void ms_to_lr_inplace(std::vector<float>& mid, std::vector<float>& side)
{
    size_t n = mid.size();
    for (size_t i = 0; i < n; ++i) {
        float m = mid[i], s = side[i];
        mid[i]  = m + s;
        side[i] = m - s;
    }
}

inline int get_is_base_original(float target_kbps) {
    if (target_kbps < 64.0f)        return 1;
    else if (target_kbps < 96.0f)   return 1;
    else if (target_kbps < 128.0f)  return 2;
    else if (target_kbps < 160.0f)  return 4;
    else if (target_kbps < 190.0f)  return 8;
    else if (target_kbps < 224.0f)  return 10;
    else if (target_kbps < 660.0f)  return 12;
    else                            return 16;
}

inline int get_is_start_band(float target_kbps, int total_bands) {
    int base = get_is_base_original(target_kbps);
    return base * total_bands / 16;
}

inline int get_is_start_from_base(int base, int total_bands) {
    base = std::clamp(base, 0, 16);
    int start = base * total_bands / 16;
    return std::clamp(start, 0, total_bands);
}

inline int get_is_segmented_threshold(float target_kbps, int total_bands) {
    int base;
    if (target_kbps < 64.0f) base = 4;
    else if (target_kbps < 96.0f)        base = 8;
    else if (target_kbps < 160.0f)  base = 10;
    else if (target_kbps < 192.0f)  base = 12;
    else                            base = 16;
    return base * total_bands / 16;
}

inline int get_r_bits(int band_idx) {
    return 4;
}

inline float compute_r_segment(const std::vector<float>& left, const std::vector<float>& right,
                               int start, int end)
{
    double eL = 0.0, eR = 0.0;
    for (size_t i = start; i < end; ++i) {
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
        if (std::abs(r_segments[i] - r_segments[i - 1]) > 0.3f) changes++;
    }
    return (changes >= 2);
}

inline void compute_is_parameters_ex(const std::vector<float>& left,
                                     const std::vector<float>& right,
                                     std::vector<float>& Y,
                                     std::array<float, 4>& r_out,
                                     std::array<bool, 4>& inv_flags,
                                     bool& use_segmented,
                                     int band_idx,
                                     int total_bands,
                                     int seg_threshold)
{
    const size_t n = std::min(left.size(), right.size());
    const double eps = 1e-12;
    (void)total_bands;
    use_segmented = (band_idx < seg_threshold);
    Y.assign(n, 0.0f);
    r_out.fill(0.5f);
    inv_flags.fill(false);

    const auto process_segment = [&](size_t start, size_t end, int segment) {
        double eL = 0.0, eR = 0.0, cross = 0.0;
        for (size_t i = start; i < end; ++i) {
            const double l = left[i];
            const double r = right[i];
            eL += l * l;
            eR += r * r;
            cross += l * r;
        }

        const double total = eL + eR;
        if (total < eps)
            return;

        r_out[segment] = static_cast<float>(eR / total);
        if (eL >= eps && eR >= eps) {
            const double correlation = cross / std::sqrt(eL * eR);
            inv_flags[segment] = (correlation < -0.2);
        }

        const double left_gain = std::sqrt(eL / total);
        const double right_gain = std::sqrt(eR / total) *
                                  (inv_flags[segment] ? -1.0 : 1.0);
        double mixed_energy = 0.0;
        for (size_t i = start; i < end; ++i) {
            const double mixed = left_gain * left[i] + right_gain * right[i];
            Y[i] = static_cast<float>(mixed);
            mixed_energy += mixed * mixed;
        }

        if (mixed_energy < eps) {
            const bool use_left = eL >= eR;
            const double source_energy = use_left ? eL : eR;
            const double gain = std::sqrt(total / source_energy);
            for (size_t i = start; i < end; ++i)
                Y[i] = static_cast<float>((use_left ? left[i] : right[i]) * gain);
        } else {
            const double gain = std::sqrt(total / mixed_energy);
            for (size_t i = start; i < end; ++i)
                Y[i] = static_cast<float>(Y[i] * gain);
        }
    };

    if (use_segmented) {
        for (int s = 0; s < 4; ++s)
            process_segment((static_cast<size_t>(s) * n) / 4,
                            (static_cast<size_t>(s + 1) * n) / 4, s);
    } else {
        process_segment(0, n, 0);
        for (int s = 1; s < 4; ++s) {
            r_out[s] = r_out[0];
            inv_flags[s] = inv_flags[0];
        }
    }
}

inline void apply_is(const std::vector<float>& Y,
                     const std::array<float, 4>& r_vals,
                     const std::array<bool, 4>& inv_flags,
                     bool use_segmented,
                     std::vector<float>& left,
                     std::vector<float>& right)
{
    size_t n = Y.size();
    left.resize(n);
    right.resize(n);

    if (use_segmented) {
        for (int s = 0; s < 4; ++s) {
            float r = std::clamp(r_vals[s], 0.0f, 1.0f);
            float gl = std::sqrt(1.0f - r);
            float gr = std::sqrt(r);
            size_t start = (static_cast<size_t>(s) * n) / 4;
            size_t end = (static_cast<size_t>(s + 1) * n) / 4;

            bool local_inv = inv_flags[s];
            for (size_t i = start; i < end; ++i) {
                left[i]  = Y[i] * gl;
                right[i] = Y[i] * (local_inv ? -gr : gr);
            }
        }
    } else {
        float r = std::clamp(r_vals[0], 0.0f, 1.0f);
        float gl = std::sqrt(1.0f - r);
        float gr = std::sqrt(r);
        bool local_inv = inv_flags[0];
        for (size_t i = 0; i < n; ++i) {
            left[i]  = Y[i] * gl;
            right[i] = Y[i] * (local_inv ? -gr : gr);
        }
    }
}

inline void apply_is_inplace(std::vector<float>& Y_and_left,
                             std::vector<float>& right_out,
                             const std::array<float, 4>& r_vals,
                             const std::array<bool, 4>& inv_flags,
                             bool use_segmented)
{
    size_t n = Y_and_left.size();
    right_out.resize(n);

    if (use_segmented) {
        for (int s = 0; s < 4; ++s) {
            float r = std::clamp(r_vals[s], 0.0f, 1.0f);
            float gl = std::sqrt(1.0f - r);
            float gr = std::sqrt(r);
            size_t start = (static_cast<size_t>(s) * n) / 4;
            size_t end = (static_cast<size_t>(s + 1) * n) / 4;
            bool local_inv = inv_flags[s];
            float gr_signed = local_inv ? -gr : gr;
            for (size_t i = start; i < end; ++i) {
                float y = Y_and_left[i];
                Y_and_left[i] = y * gl;
                right_out[i]  = y * gr_signed;
            }
        }
    } else {
        float r = std::clamp(r_vals[0], 0.0f, 1.0f);
        float gl = std::sqrt(1.0f - r);
        float gr = std::sqrt(r);
        bool local_inv = inv_flags[0];
        float gr_signed = local_inv ? -gr : gr;
        for (size_t i = 0; i < n; ++i) {
            float y = Y_and_left[i];
            Y_and_left[i] = y * gl;
            right_out[i]  = y * gr_signed;
        }
    }
}

inline uint32_t quantize_r(float r, int bits) {
    float r_clamped = std::clamp(r, 0.0f, 1.0f);
    float theta = std::asin(std::sqrt(r_clamped));
    int max_val = (1 << bits) - 2;
    float theta_norm = theta / (float)(M_PI * 0.5f);
    return static_cast<uint32_t>(std::clamp(theta_norm, 0.0f, 1.0f) * max_val + 0.5f);
}

inline float dequantize_r(uint32_t q, int bits) {
    int max_val = (1 << bits) - 2;
    float theta_norm = (float)q / max_val;
    float theta = theta_norm * (float)(M_PI * 0.5f);
    float s = std::sin(theta);
    return s * s;
}

inline std::pair<std::vector<float>, std::vector<float>> mid_side(const std::vector<float>& left,
                                                                  const std::vector<float>& right)
{
    size_t n = left.size();
    std::vector<float> mid(n), side(n);
    for (size_t i = 0; i < n; ++i) {
        mid[i]  = (left[i] + right[i]) * 0.5f;
        side[i] = (left[i] - right[i]) * 0.5f;
    }
    return {mid, side};
}

inline bool use_mid_side(float El, float Er, float Em, float Es,
                         bool enable_ms, float target_kbps)
{
    if (!enable_ms) return false;
    const float eps = 1e-12f;
    if (target_kbps < 64.0f) {
        if (Es > Em * 0.9f) return false;
        float prod_lr = El * Er;
        float prod_ms = Em * Es;
        if (prod_ms > prod_lr * 1.05f) return false;
        return true;
    }
    float min_e = std::min(El, Er);
    float max_e = std::max(El, Er);
    float energy_ratio = max_e / (min_e + eps);
    if (energy_ratio > 4.0f) return false;
    if (Es > Em * 0.5f) return false;
    float prod_lr = El * Er;
    float prod_ms = Em * Es;
    return prod_ms < prod_lr * 0.95f;
}

inline bool should_use_is_band(const std::vector<float>& left,
                               const std::vector<float>& right,
                               bool enable_ms,
                               float target_kbps,
                               float strong_ratio_threshold,
                               float equal_ratio_threshold,
                               float equal_ratio_threshold_low_bitrate)
{
    const size_t n = std::min(left.size(), right.size());
    if (n == 0) return false;
    const double eps = 1e-12;
    double eL = 0.0, eR = 0.0, cross = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double L = (double)left[i];
        const double R = (double)right[i];
        eL += L * L; eR += R * R; cross += L * R;
    }
    if (eL + eR < eps) return false;
    const double correlation = cross / (std::sqrt(eL * eR) + eps);
    const double abs_correlation = std::abs(correlation);
    constexpr double IS_CORRELATION_THRESHOLD = 0.60;
    if (abs_correlation >= IS_CORRELATION_THRESHOLD) return true;

    const double meanL = eL / (double)n;
    const double meanR = eR / (double)n;
    const double minE = std::min(meanL, meanR);
    const double maxE = std::max(meanL, meanR);
    const double ratio = maxE / (minE + eps);
    if (ratio > (double)strong_ratio_threshold) return true;

    const double Em = (eL + eR + 2.0 * cross) * 0.25;
    const double Es = (eL + eR - 2.0 * cross) * 0.25;
    const bool ms_benefit = use_mid_side((float)eL, (float)eR, (float)Em, (float)Es, enable_ms, target_kbps);
    return ms_benefit || maxE >= minE * 4.0;
}

inline int32_t quantize_r_centered(float r, int bits) {
    float r_clamped = std::clamp(r, 0.0f, 1.0f);
    float theta = std::asin(std::sqrt(r_clamped));
    float theta_center = (float)(M_PI * 0.25);
    int steps = (1 << (bits - 1)) - 1;
    float step_size = theta_center / steps;
    int32_t q = (int32_t)std::round((theta - theta_center) / step_size);
    return std::clamp(q, -steps, steps);
}

inline float dequantize_r_centered(int32_t q, int bits) {
    int steps = (1 << (bits - 1)) - 1;
    float theta_center = (float)(M_PI * 0.25);
    float step_size = theta_center / steps;
    float theta = theta_center + q * step_size;
    float s = std::sin(theta);
    return s * s;
}

#endif // JOINT_STEREO_H