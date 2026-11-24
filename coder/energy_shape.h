#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct BandEnergy {
    std::vector<float> e0;
    std::vector<float> e1;
    float              total = 0.0f;
};

inline BandEnergy compute_band_energy(
    const std::vector<std::vector<float>>& ch0_bands,
    const std::vector<std::vector<float>>& ch1_bands,
    const std::vector<uint8_t>&            is_band,
    bool stereo, int band_count)
{
    BandEnergy result;
    result.e0.assign(band_count, 0.0f);
    result.e1.assign(band_count, 0.0f);
    result.total = 0.0f;

    for (int i = 0; i < band_count; ++i) {
        float e0 = 0.0f, e1 = 0.0f;
        for (float v : ch0_bands[i]) e0 += v * v;
        for (float v : ch1_bands[i]) e1 += v * v;
        result.e0[i] = e0;
        result.e1[i] = e1;
        result.total += e0;
        if (stereo && !is_band[i])
            result.total += e1;
        if (stereo && is_band[i])
            result.e1[i] = 0.0f;
    }
    return result;
}

inline void apply_base_pow(std::vector<float>& e0, std::vector<float>& e1, int band_count) {
    for (int i = 0; i < band_count; ++i) {
        const float s0 = sqrtf(e0[i]);
        e0[i] = s0 * sqrtf(s0);
        const float s1 = sqrtf(e1[i]);
        e1[i] = s1 * sqrtf(s1);
    }
}

inline void apply_adaptive_pow(std::vector<float>& e0, std::vector<float>& e1,
                               int band_count, bool stereo,
                               float target_kbps, bool has_transient)
{
    if (has_transient) return;

    float exp;
    if (target_kbps <= 128.0f)
        exp = 0.75f;
    else if (target_kbps >= 192.0f)
        exp = 0.95f;
    else
        exp = 0.75f + (0.95f - 0.75f) * (target_kbps - 128.0f) / (192.0f - 128.0f);
    exp = std::clamp(exp, 0.5f, 1.0f);

    float max_energy = 0.0f;
    for (int i = 0; i < band_count; ++i)
        max_energy = fmaxf(max_energy, fmaxf(e0[i], e1[i]));

    if (max_energy > 1e-12f) {
        const float thresh = max_energy * 1e-5f;
        for (int i = 0; i < band_count; ++i) {
            if (e0[i] > thresh)
                e0[i] = powf(e0[i], exp);
            if (stereo && e1[i] > thresh)
                e1[i] = powf(e1[i], exp);
        }
    }
}

inline void apply_narrowband_correction(std::vector<float>& e0, std::vector<float>& e1,
                                        int band_count, bool stereo)
{
    float total_e = 0.0f, low_freq_e = 0.0f;
    const int half_bands = band_count / 2;
    for (int i = 0; i < band_count; ++i) {
        total_e += e0[i];
        if (i < half_bands)
            low_freq_e += e0[i];
    }
    const bool is_narrowband = (total_e > 1e-6f) && (low_freq_e / total_e > 0.98f);
    if (is_narrowband) {
        for (int i = half_bands; i < band_count; ++i) {
            e0[i] *= 0.5f;
            if (stereo)
                e1[i] *= 0.5f;
        }
    }
}

inline void apply_highfreq_attenuation(std::vector<float>& e0, std::vector<float>& e1,
                                       int band_count, float target_kbps,
                                       int num_channels, bool has_transient)
{
    const float kbps_per_channel = target_kbps / float(num_channels);
    
    if (kbps_per_channel >= 48.0f)
        return;

    size_t start;
    if (kbps_per_channel < 48.0f) {
        start = !has_transient
            ? 3 * e0.size() / 8
            : 4 * e0.size() / 8;
    } else {
        if (has_transient)
            return;
        start = e0.size() / 2;
    }

    const size_t end = std::min(e0.size(), e1.size());
    if (start >= end)
        return;

    const float start_gain = 0.25f;
    const float end_gain   = 0.05f;
    const float denom      = float(end - start - 1);

    for (size_t i = start; i < end; ++i) {
        const float t = denom > 0.0f ? float(i - start) / denom : 1.0f;
        const float gain = start_gain + (end_gain - start_gain) * t;
        e0[i] *= gain;
        e1[i] *= gain;
    }
}