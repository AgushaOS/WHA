#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

inline bool detect_transient_block_stereo(
    const std::vector<float>& block,
    int num_channels)
{
    static bool is_transient = false;

    if (num_channels <= 0)
        return is_transient;

    const int N = static_cast<int>(block.size() / num_channels);

    if (N < 64)
        return is_transient;

    constexpr int MAX_N = 2048;
    constexpr int MAX_LEVELS = 6;

    constexpr float INV_SQRT2 = 0.70710678118f;
    constexpr float EPSILON = 1e-12f;

    constexpr float PEAK_THRESHOLD_ON = 28.0f;
    constexpr float PEAK_THRESHOLD_OFF = 18.0f;

    constexpr float BASS_DOMINANCE_THRESHOLD = 3.0f;

    if (N > MAX_N)
        return is_transient;

    auto check_channel = [&](int channel, bool& state) -> bool
    {
        float approx[MAX_N];
        float next[MAX_N];

        float detail_energy[MAX_LEVELS] = {};
        float detail_max[MAX_LEVELS] = {};
        int detail_count[MAX_LEVELS] = {};

        float total_energy = 0.0f;

        for (int i = 0; i < N; ++i) {
            const float x = block[i * num_channels + channel];

            approx[i] = x;
            total_energy += x * x;
        }

        if (total_energy < 1e-10f) {
            state = false;
            return false;
        }

        int len = N;

        for (int level = 0; level < MAX_LEVELS && len >= 2; ++level) {
            const int pairs = len / 2;

            float energy = 0.0f;
            float maximum = 0.0f;

            for (int i = 0; i < pairs; ++i) {
                const float x0 = approx[2 * i];
                const float x1 = approx[2 * i + 1];

                const float a =
                    (x0 + x1) * INV_SQRT2;

                const float d =
                    (x0 - x1) * INV_SQRT2;

                next[i] = a;

                const float e = d * d;

                energy += e;

                if (e > maximum)
                    maximum = e;
            }

            detail_energy[level] = energy;
            detail_max[level] = maximum;
            detail_count[level] = pairs;

            for (int i = 0; i < pairs; ++i)
                approx[i] = next[i];

            len = pairs;
        }

        float bass_energy = 0.0f;

        for (int i = 0; i < len; ++i)
            bass_energy += approx[i] * approx[i];

        float rest_energy = 0.0f;

        for (int level = 0; level < MAX_LEVELS; ++level)
            rest_energy += detail_energy[level];

        const float bass_dominance =
            bass_energy / (rest_energy + EPSILON);

        const bool bass_dominates =
            bass_dominance >= BASS_DOMINANCE_THRESHOLD;

        float peak_score = 0.0f;

        for (int level = 0; level < MAX_LEVELS; ++level) {
            const int count = detail_count[level];

            if (count <= 0)
                continue;

            const float energy = detail_energy[level];

            if (energy < EPSILON)
                continue;

            const float mean_detail =
                energy / static_cast<float>(count);

            const float detail_peak =
                detail_max[level] / (mean_detail + EPSILON);

            if (detail_peak > peak_score)
                peak_score = detail_peak;
        }

        if (bass_dominates) {
            state = false;
            return false;
        }

        if (!state) {
            if (peak_score > PEAK_THRESHOLD_ON)
                state = true;
        } else {
            if (peak_score < PEAK_THRESHOLD_OFF)
                state = false;
        }

        return state;
    };

    static bool is_transient_L = false;
    static bool is_transient_R = false;

    const bool trans_L =
        check_channel(0, is_transient_L);

    bool trans_R = false;

    if (num_channels > 1) {
        trans_R = check_channel(1, is_transient_R);
    } else {
        is_transient_R = false;
    }

    is_transient = trans_L || trans_R;

    return is_transient;
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