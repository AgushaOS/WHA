#pragma once
#include <vector>
#include <cmath>

#include <vector>
#include <cmath>
#include <algorithm>

#include <vector>
#include <cmath>
#include <algorithm>

inline bool detect_transient_block_stereo(
    const std::vector<float>& block,
    int num_channels)
{
    if (num_channels <= 0)
        return false;

    const int N = (int)(block.size() / num_channels);
    if (N < 64)
        return false;

    const float* data = block.data();

    constexpr float INV_SQRT2 = 0.70710678118f;

    float detail_energy = 0.0f;
    float detail_max = 0.0f;
    float total_energy = 0.0f;

    for (int i = 0; i + 1 < N; i += 2)
    {
        const float x0 = data[i * num_channels];
        const float x1 = data[(i + 1) * num_channels];

        const float a = (x0 + x1) * INV_SQRT2;
        const float d = (x0 - x1) * INV_SQRT2;

        const float e = d * d;

        detail_energy += e;
        detail_max = std::max(detail_max, e);
        total_energy += a * a + e;
    }

    if (total_energy < 1e-10f)
        return false;

    const float detail_ratio =
        detail_energy / (total_energy + 1e-12f);

    const int pairs = N / 2;

    const float mean_detail =
        detail_energy / (pairs + 1e-12f);

    const float detail_peak =
        detail_max / (mean_detail + 1e-12f);

    constexpr float DETAIL_THRESHOLD = 0.010f;
    constexpr float PEAK_THRESHOLD = 12.0f;

    if (detail_ratio < DETAIL_THRESHOLD)
        return false;

    return detail_peak > PEAK_THRESHOLD;
}

inline bool detect_transient_block_mono(const std::vector<float>& block, int num_channels) {
    int N = (int)(block.size() / num_channels);
    const float* data = block.data();
    if (N < 64) return false;

    float diff_energy = 0.0f, total_energy = 0.0f, max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float sample = data[i * num_channels];
        float av = std::abs(sample);
        if (av > max_abs) max_abs = av;
        total_energy += sample * sample;
        if (i > 0) {
            float diff = sample - data[(i - 1) * num_channels];
            diff_energy += diff * diff;
        }
    }
    if (total_energy < 1e-6f) return false;

    float smoothness_ratio = diff_energy / total_energy;
    float rms              = std::sqrt(total_energy / N);
    float peak_to_rms      = max_abs / (rms + 1e-12f);
    return !((smoothness_ratio < 0.5f) && (peak_to_rms < 3.00f));
}

inline bool detect_strong_transient(const std::vector<float>& energy0,
                                    const std::vector<float>& energy1,
                                    int band_count, bool stereo, float threshold)
{
    if (band_count < 3) return false;
    float total = 0.0f, high = 0.0f;
    int   high_start = band_count / 4;
    for (int i = 0; i < band_count; ++i) {
        float e = energy0[i] + (stereo ? energy1[i] : 0.0f);
        total += e;
        if (i >= high_start)
            high += e;
    }
    if (total < 1e-12f) return false;
    return ((high / total) > 0.15f);
}