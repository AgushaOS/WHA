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
        for (float v : ch0_bands[i]) result.e0[i] += v * v;
        for (float v : ch1_bands[i]) result.e1[i] += v * v;
        result.total += result.e0[i];
        if (stereo && !is_band[i])
            result.total += result.e1[i];
        if (stereo && is_band[i])
            result.e1[i] = 0.0f;
    }
    return result;
}

inline void apply_base_pow(std::vector<float>& e0, std::vector<float>& e1, int band_count) {
    for (int i = 0; i < band_count; ++i) {
        e0[i] = std::pow(e0[i], 0.75f);
        e1[i] = std::pow(e1[i], 0.75f);
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
        max_energy = std::max({max_energy, e0[i], e1[i]});

    if (max_energy > 1e-12f) {
        for (int i = 0; i < band_count; ++i) {
            if (e0[i] / max_energy > 1e-5f)
                e0[i] = std::pow(e0[i], exp);
            if (stereo && e1[i] / max_energy > 1e-5f)
                e1[i] = std::pow(e1[i], exp);
        }
    }
}

inline void apply_narrowband_correction(std::vector<float>& e0, std::vector<float>& e1,
                                        int band_count, bool stereo)
{
    float total_e = 0.0f, low_freq_e = 0.0f;
    int   half_bands = band_count / 2;
    for (int i = 0; i < band_count; ++i) {
        total_e += e0[i];
        if (i < half_bands)
            low_freq_e += e0[i];
    }
    bool is_narrowband = (total_e > 1e-6f) && (low_freq_e / total_e > 0.98f);
    if (is_narrowband) {
        for (int i = band_count / 2; i < band_count; ++i) {
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
    if (target_kbps / float(num_channels) >= 48.0f) return;

    int64_t start;
    if (!has_transient)
        start = 3 * (int64_t)e0.size() / 8;
    else
        start = 4 * (int64_t)e0.size() / 8;

    for (int64_t i = start; i < (int64_t)e0.size(); ++i)
        e0[i] *= 0.25f;
}