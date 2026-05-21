#ifndef PERC_PRIOR_H
#define PERC_PRIOR_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

inline std::vector<float> compute_channel_priority(
    const std::vector<std::vector<float>>& coeffs,
    std::vector<std::vector<float>>& prev_coeffs,
    const std::vector<int>& coeff_counts,
    float target_kbps,
    int sr,
    int level,
    bool is_side_channel)
{
    const float eps = 1e-12f;
    int band_count = (int)coeffs.size();
    if (prev_coeffs.size() != coeffs.size()) prev_coeffs = coeffs;
    static thread_local std::vector<float> priority;
    priority.resize(band_count);
    float global_energy = 0.0f;
    for (const auto& band : coeffs)
        for (float v : band) global_energy += v * v;
    global_energy = std::sqrt(global_energy + eps);

    int total_bands = 1 << level;
    std::vector<float> weight_original(total_bands, 1.0f);
    for (int i = 0; i < total_bands; ++i) {
        int popcnt = __builtin_popcount(i);  
        float w = std::exp2((level - 2 * popcnt) * 0.5f);
        weight_original[i] = w;
    }
    std::vector<int> perm(total_bands);
    for (int i = 0; i < total_bands; ++i) perm[i] = i ^ (i >> 1);
    std::vector<float> freq_weight(band_count);
    for (int i = 0; i < total_bands; ++i)
        freq_weight[perm[i]] = weight_original[i] * 0.75f;

    for (int i = 0; i < band_count; ++i) {
        const auto& cur = coeffs[i];
        const auto& prev = prev_coeffs[i];
        float max_v = 0.0f, abs_sum = 0.0f;
        for (float v : cur) {
            float a = std::fabs(v);
            max_v = std::max(max_v, a);
            abs_sum += a;
        }
        float mean = abs_sum / std::max<size_t>(cur.size(), 1);
        float peakiness = std::clamp((max_v - mean) / (max_v + eps), 0.0f, 1.0f);
        float abs_energy = 0.0f;
        for (float v : cur) abs_energy += v * v;
        float energy_local = std::sqrt(abs_energy + eps) / (global_energy + eps);
        energy_local = std::clamp(energy_local, 0.0f, 1.0f);
        float structure = 0.7f * peakiness + 0.3f * energy_local;

        float diff_energy = 0.0f, prev_energy = 0.0f;
        size_t N = std::min(cur.size(), prev.size());
        for (size_t k = 0; k < N; ++k) {
            float d = cur[k] - prev[k];
            diff_energy += d * d;
            prev_energy += prev[k] * prev[k];
        }
        float flux = diff_energy / (prev_energy + eps);
        flux = std::sqrt(flux);
        flux = std::clamp(flux, 0.0f, 1.0f);
        float transient = 1.0f - std::exp(-3.0f * flux);
        float energy_importance = std::log1p(abs_energy + eps);
        energy_importance = std::sqrt(energy_importance);
        energy_importance = std::clamp(energy_importance / 8.0f, 0.0f, 1.0f);

        float p_val = 0.55f * structure + 0.30f * transient + 0.15f * energy_importance;
        p_val *= freq_weight[i];
        p_val = std::tanh(1.5f * p_val);
        priority[i] = p_val;
    }

    static thread_local std::vector<float> prev_priority;
    if (prev_priority.size() != priority.size()) prev_priority = priority;
    for (int i = 0; i < band_count; ++i)
        priority[i] = 0.65f * priority[i] + 0.35f * prev_priority[i];
    prev_priority = priority;
    prev_coeffs = coeffs;

    float sum = 0.0f;
    for (float v : priority) sum += v;
    if (sum < eps) {
        std::fill(priority.begin(), priority.end(), 1.0f / band_count);
        return priority;
    }
    return priority;
}

#endif // PERC_PRIOR_H