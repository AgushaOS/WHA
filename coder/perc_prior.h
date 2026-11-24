#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

inline std::vector<float> compute_channel_priority(
    const std::vector<const std::vector<float>*>& bands,
    std::vector<std::vector<float>>& prev_coeffs,
    const std::vector<int>& coeff_counts,
    float target_kbps,
    int sr,
    int level,
    bool is_side_channel,
    float num_channels)
{
    const float eps = 1e-12f;
    const int band_count = (int)bands.size();
    if (band_count == 0)
        return {};

    if (prev_coeffs.size() != bands.size()) {
        prev_coeffs.resize(bands.size());
        for (int i = 0; i < band_count; ++i)
            prev_coeffs[i] = *bands[i];
    }

    static thread_local std::vector<float> priority;
    priority.resize(band_count);

    static thread_local std::vector<float> band_sqrt_e;
    band_sqrt_e.resize(band_count);

    float global_energy = 0.0f;
    for (int i = 0; i < band_count; ++i) {
        const auto& band = *bands[i];
        float e = 0.0f;
        for (float v : band) e += v * v;
        global_energy += e;
        band_sqrt_e[i] = sqrtf(e + eps);
    }

    const float global_energy_sq = global_energy + eps;
    const float global_sqrt_e    = sqrtf(global_energy_sq) + eps;

    const int total_bands = 1 << level;

    float exp2_tab[64];
    {
        const int n = std::min(level, 63);
        const float half_level = 0.5f * (float)level;
        for (int p = 0; p <= n; ++p)
            exp2_tab[p] = exp2f(half_level - (float)p);
    }

    static thread_local std::vector<float> weight_original;
    weight_original.resize(total_bands);
    for (int i = 0; i < total_bands; ++i) {
        const int pc = __builtin_popcount((unsigned)i);
        weight_original[i] = (pc < 64)
            ? exp2_tab[pc]
            : exp2f(0.5f * (float)level - (float)pc);
    }

    static thread_local std::vector<float> freq_weight;
    freq_weight.assign(band_count, 1.0f);
    for (int i = 0; i < total_bands && i < band_count; ++i) {
        const int old_index = i ^ (i >> 1);
        if (old_index < total_bands)
            freq_weight[i] = weight_original[old_index] * 0.75f;
    }

    const bool use_tanh = (target_kbps / num_channels >= 40.0f);
    constexpr float PEAK_THRESHOLD = 0.01f;

    for (int i = 0; i < band_count; ++i) {
        const auto& cur  = *bands[i];
        const auto& prev = prev_coeffs[i];

        float max_v = 0.0f;
        float l1    = 0.0f;
        float abs_energy = 0.0f;
        for (float v : cur) {
            const float a = fabsf(v);
            max_v = fmaxf(max_v, a);
            l1 += a;
            abs_energy += v * v;
        }

        const float mean = l1 / (float)std::max<size_t>(cur.size(), 1);
        const float peakiness = std::clamp((max_v - mean) / (max_v + eps), 0.0f, 1.0f);

        const float band_sqrt = band_sqrt_e[i];
        const float energy_local = std::clamp(band_sqrt / global_sqrt_e, 0.0f, 1.0f);

        const float structure = 0.65f * peakiness + 0.35f * energy_local;

        float corr = 0.0f, norm_prev = 0.0f, norm_cur = 0.0f;
        const size_t N = std::min(cur.size(), prev.size());
        for (size_t k = 0; k < N; ++k) {
            corr      += cur[k] * prev[k];
            norm_prev += prev[k] * prev[k];
            norm_cur  += cur[k]  * cur[k];
        }
        float predict = std::clamp(corr / (sqrtf(norm_prev * norm_cur) + eps), 0.0f, 1.0f);

        const float l2 = band_sqrt;
        float sparsity = std::clamp((l1 * l1) / ((float)cur.size() * l2 * l2 + eps),
                                    0.0f, 1.0f);

        float inter_scale = 0.5f;
        if (i > 0 && i < band_count - 1) {
            const float e_prev = band_sqrt_e[i - 1];
            const float e_next = band_sqrt_e[i + 1];
            const float e_cur  = band_sqrt_e[i];
            const float avg_neighbors = 0.5f * (e_prev + e_next);
            inter_scale = 1.0f - fabsf(e_cur - avg_neighbors) / (avg_neighbors + eps);
            inter_scale = std::clamp(inter_scale, 0.0f, 1.0f);
        }

        float tonality = 0.0f;
        if (cur.size() >= 16 && abs_energy > eps) {
            double s00 = 0.0, s01 = 0.0, s11 = 0.0;
            double sy0 = 0.0, sy1 = 0.0;
            double ar_energy = 0.0;
            const int count = (int)cur.size();
            for (int k = 2; k < count; ++k) {
                const double a = cur[k - 1];
                const double b = cur[k - 2];
                const double y = cur[k];
                s00 += a * a;
                s01 += a * b;
                s11 += b * b;
                sy0 += a * y;
                sy1 += b * y;
                ar_energy += y * y;
            }
            const double reg = ar_energy * 1e-6 / count;
            s00 += reg;
            s11 += reg;
            const double det = s00 * s11 - s01 * s01;
            if (std::fabs(det) > 1e-20) {
                const double a1 = (sy0 * s11 - sy1 * s01) / det;
                const double a2 = (sy1 * s00 - sy0 * s01) / det;
                double prediction_error = 0.0;
                for (int k = 2; k < count; ++k) {
                    const double predicted = a1 * cur[k - 1] + a2 * cur[k - 2];
                    const double e = cur[k] - predicted;
                    prediction_error += e * e;
                }
                const double raw_tonality = 1.0 - prediction_error / (ar_energy + eps);
                tonality = std::clamp((float)raw_tonality, 0.0f, 1.0f);
            }
        }

        const float cur_peak = max_v;
        float prev_peak = 0.0f;
        for (float v : prev)
            prev_peak = fmaxf(prev_peak, fabsf(v));

        float transient = 0.0f;
        if (cur_peak > PEAK_THRESHOLD && prev_peak > PEAK_THRESHOLD) {
            const float peak_ratio = cur_peak / prev_peak;
            transient = std::clamp(log2f(peak_ratio + 1.0f), 0.0f, 4.0f);
        }

        float p_val =
            0.35f * predict +
            0.55f * sparsity +
            0.20f * structure +
            0.20f * inter_scale +
            0.65f * tonality +
            transient;

        p_val *= freq_weight[i];

        if (use_tanh)
            p_val = tanhf(1.3f * p_val);

        priority[i] = p_val;
    }

    static thread_local std::vector<float> prev_priority;
    if (prev_priority.size() != priority.size())
        prev_priority = priority;

    for (int i = 0; i < band_count; ++i)
        priority[i] = 0.70f * priority[i] + 0.30f * prev_priority[i];
    prev_priority = priority;

    for (int i = 0; i < band_count; ++i)
        prev_coeffs[i] = *bands[i];

    float sum = 0.0f;
    for (float v : priority) sum += v;

    if (sum < eps) {
        std::fill(priority.begin(), priority.end(), 1.0f / (float)band_count);
        return priority;
    }
    return priority;
}