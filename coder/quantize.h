#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>

inline float percentile85_abs(const std::vector<float>& arr) {
    if (arr.empty()) return 0.0f;

    std::vector<float> tmp(arr.size());
    for (size_t i = 0; i < arr.size(); ++i)
        tmp[i] = std::abs(arr[i]);

    size_t idx = tmp.size() * 83 / 100;
    std::nth_element(tmp.begin(), tmp.begin() + idx, tmp.end());
    return tmp[idx];
}

inline float compute_powerlaw_deadzone(float step, float rms, float target_kbps) {
    if (step < 1e-12f)
        return 0.0f;

    float energy_ratio = rms / step;
    float mult = 0.55f + 0.10f * std::log2f(1.0f + energy_ratio);

    if (target_kbps <= 128) {
        mult = std::clamp(mult, 0.50f, 0.75f);
    } else {
        mult = std::clamp(mult, 0.50f, 0.55f);
    }

    return mult * step;
}

struct TemporalMaskingState {
    std::vector<float> prev_effective_rms;
    bool initialized = false;

    float decay_factor = 0.85f;
};

struct QuantResult {
    std::vector<float> scales;
    std::vector<std::vector<int32_t>> quantized_per_level;
};

TemporalMaskingState* tm_state = nullptr;

inline QuantResult quantize_levels(
    const std::vector<std::vector<std::vector<float>>>& coeffs_channels,
    const std::vector<int>& bits_per_level,
    float target_kbps,
    float true_num_channels)
{
    const int num_channels = static_cast<int>(coeffs_channels.size());
    const int band_count   = static_cast<int>(coeffs_channels[0].size());

    if (tm_state && !tm_state->initialized) {
        tm_state->prev_effective_rms.assign(band_count, 0.0f);
        tm_state->initialized = true;
    }

    std::vector<float> scales(band_count);
    std::vector<std::vector<int32_t>> quantized_per_level(band_count);

    float deadzone_scale = 1.0f;

    if (target_kbps > 64.0f && target_kbps <= 128.0f) {
        float t = (target_kbps - 96.0f) / (320.0f - 96.0f);
        t = std::clamp(t, 0.0f, 1.0f);
        deadzone_scale = 1.0f - std::pow(t, 1.5f);
    } 

    deadzone_scale = std::clamp(deadzone_scale, 0.0f, 1.0f);

    const int block_size = std::max(4, 256 / band_count);
    static thread_local std::vector<float> block_data;

    for (int l = 0; l < band_count; ++l) {
        const int bbits = bits_per_level[l];

        const size_t len = coeffs_channels[0][l].size();
        const size_t total_samples = len * static_cast<size_t>(num_channels);

        if (len == 0 || bbits <= 0) {
            scales[l] = 0.0f;
            quantized_per_level[l].assign(total_samples, 0);

            if (tm_state)
                tm_state->prev_effective_rms[l] = 0.0f;

            continue;
        }

        float max_abs = 0.0f;

        const int num_blocks =
            static_cast<int>((len + block_size - 1) / block_size);

        block_data.assign(num_blocks, 0.0f);

        for (int ch = 0; ch < num_channels; ++ch) {
            const auto& arr = coeffs_channels[ch][l];
            for (int b = 0; b < num_blocks; ++b) {
                const size_t start = static_cast<size_t>(b) * block_size;
                const size_t end = std::min(start + block_size, len);
                float sum_sq = block_data[b];
                for (size_t i = start; i < end; ++i) {
                    const float v = arr[i];
                    const float av = std::abs(v);
                    if (av > max_abs)
                        max_abs = av;
                    sum_sq += v * v;
                }
                block_data[b] = sum_sq;
            }

        }

        if (bbits <= 2)
            max_abs *= 0.5f;

        if (max_abs < 1e-12f)
            max_abs = 1.0f;

        int sgn = 1;

        const int max_int =
            (bbits > 1) ? ((1 << (bbits - 1)) - 1) : 1;

        const float step =
            max_abs / static_cast<float>(max_int);

        float current_prev_rms =
            tm_state ? tm_state->prev_effective_rms[l] : 0.0f;

        for (int b = 0; b < num_blocks; ++b) {
            const size_t block_start = static_cast<size_t>(b) * block_size;
            const size_t block_length =
                std::min(static_cast<size_t>(block_size), len - block_start);
            const float block_rms =
                std::sqrt(
                    block_data[b] /
                    static_cast<float>(block_length * num_channels)
                    + 1e-12f
                );

            float effective_rms = block_rms;

            if (tm_state) {
                const float masked_rms =
                    current_prev_rms * tm_state->decay_factor;

                effective_rms = std::max(block_rms, masked_rms);
                current_prev_rms = effective_rms;
            }

            block_data[b] =
                compute_powerlaw_deadzone(
                    step,
                    effective_rms,
                    target_kbps
                ) * deadzone_scale;
        }

        if (tm_state)
            tm_state->prev_effective_rms[l] = current_prev_rms;

        std::vector<int32_t> quantized;
        quantized.resize(total_samples);

        size_t out_pos = 0;

        if (bbits == 1) {
            for (int ch = 0; ch < num_channels; ++ch) {
                const auto& arr = coeffs_channels[ch][l];
                for (size_t i = 0; i < len; ++i) {
                    const float v_clip = std::max(0.0f, arr[i]);
                    int32_t q = static_cast<int32_t>(
                        std::round(v_clip / step));
                    quantized[out_pos++] = std::clamp(q, 0, 1);
                }
            }
        } else if (len % block_size == 0) {
            for (int ch = 0; ch < num_channels; ++ch) {
                const auto& arr = coeffs_channels[ch][l];
                for (int b = 0; b < num_blocks; ++b) {
                    const float deadzone = block_data[b];
                    const size_t start = static_cast<size_t>(b) * block_size;
                    const size_t end = start + block_size;
                    for (size_t i = start; i < end; ++i) {
                        const float v = arr[i];
                        const int32_t q = std::abs(v) < deadzone
                            ? 0
                            : static_cast<int32_t>(std::round(v / step));
                        quantized[out_pos++] = q;
                    }
                }
            }
        } else {
            for (int ch = 0; ch < num_channels; ++ch) {
                const auto& arr = coeffs_channels[ch][l];
                int b = 0;
                size_t next_boundary = block_size;
                for (size_t i = 0; i < len; ++i) {
                    if (i >= next_boundary) {
                        ++b;
                        next_boundary += block_size;
                    }

                    const float v = arr[i];
                    const int32_t q = std::abs(v) < block_data[b]
                        ? 0
                        : static_cast<int32_t>(std::round(v / step));
                    quantized[out_pos++] = q;
                }
            }
        }

        scales[l] =
            (bbits == 1)
                ? ((sgn == 1) ? max_abs : -max_abs)
                : max_abs;
        
        if (target_kbps / true_num_channels < 96.0f) {
            scales[l] /= 16.0f;
        }

        quantized_per_level[l] = std::move(quantized);
    }

    return { std::move(scales), std::move(quantized_per_level) };
}

#endif // QUANTIZE_H
