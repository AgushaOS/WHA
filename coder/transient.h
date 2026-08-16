#pragma once
#include <vector>
#include <cmath>

inline bool detect_transient_block_stereo(const std::vector<float>& block, int num_channels) {
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

    static float smoothed_peak = 0.0f;
    static bool  initialized   = false;
    if (!initialized) {
        smoothed_peak = peak_to_rms;
        initialized   = true;
    } else {
        float alpha   = (peak_to_rms > smoothed_peak) ? 0.75f : 0.4f;
        smoothed_peak = alpha * peak_to_rms + (1.0f - alpha) * smoothed_peak;
    }
    return !((smoothness_ratio < 0.5f) && (smoothed_peak < 2.96f));
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