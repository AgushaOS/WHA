#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>


inline float percentile85_abs(const std::vector<float>& arr) {
    if (arr.empty()) return 0.0f;
    std::vector<float> tmp(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) tmp[i] = std::abs(arr[i]);
    size_t idx = tmp.size() * 83 / 100;
    std::nth_element(tmp.begin(), tmp.begin() + idx, tmp.end());
    return tmp[idx];
}

inline float compute_energy_adaptive_deadzone(float step, float rms) {
    if (step < 1e-12f) return 0.0f;
    
    float energy_ratio = rms / step;
    
    float mult = 0.50f + 0.15f * std::log2f(1.0f + energy_ratio);
    
    mult = std::clamp(mult, 0.45f, 0.75f);
    
    return mult * step;
}

struct QuantResult {
    std::vector<float> scales;
    std::vector<std::vector<int32_t>> quantized_per_level;
};

inline QuantResult quantize_levels(
    const std::vector<std::vector<std::vector<float>>>& coeffs_channels,
    const std::vector<int>& bits_per_level,
    float target_kbps)
{
    const int num_channels = static_cast<int>(coeffs_channels.size());
    const int band_count   = static_cast<int>(coeffs_channels[0].size());

    std::vector<float> scales(band_count);
    std::vector<std::vector<int32_t>> quantized_per_level(band_count);

    float deadzone_scale = 1.0f;
    if (target_kbps > 256.0f) {
        float t = (target_kbps - 256.0f) / (320.0f - 256.0f);
        t = std::clamp(t, 0.0f, 1.0f);
        deadzone_scale = 1.0f - std::pow(t, 1.5f);
    }
    deadzone_scale = std::clamp(deadzone_scale, 0.0f, 1.0f);

    for (int l = 0; l < band_count; ++l) {
        const int bbits = bits_per_level[l];
        size_t total_samples = 0;
        for (int ch = 0; ch < num_channels; ++ch)
            total_samples += coeffs_channels[ch][l].size();

        if (total_samples == 0 || bbits <= 0) {
            scales[l] = 0.0f;
            quantized_per_level[l].assign(total_samples, 0);
            continue;
        }

        float max_abs = 0.0f;
        if (bbits <= 2) {
            std::vector<float> abs_vals;
            abs_vals.reserve(total_samples);
            for (int ch = 0; ch < num_channels; ++ch) {
                const auto& arr = coeffs_channels[ch][l];
                for (float v : arr) {
                    float av = std::abs(v);
                    abs_vals.push_back(av);
                    if (av > max_abs) max_abs = av;
                }
            }
            float p85 = percentile85_abs(abs_vals);
            const float THRESH = 3.5f;
            max_abs = (max_abs > THRESH * p85) ? max_abs / 2.0f : p85;
        } else {
            for (int ch = 0; ch < num_channels; ++ch) {
                const auto& arr = coeffs_channels[ch][l];
                for (float v : arr) {
                    float av = std::abs(v);
                    if (av > max_abs) max_abs = av;
                }
            }
        }
        if (max_abs < 1e-12f) max_abs = 1.0f;

        int sgn = 1;
        if (bbits == 2) {
            double sum = 0.0;
            for (int ch = 0; ch < num_channels; ++ch)
                for (float v : coeffs_channels[ch][l]) sum += v;
            sgn = (sum >= 0.0) ? 1 : -1;
        }

        int max_int = (bbits > 1) ? ((1 << (bbits - 1)) - 1) : 1;
        float step = max_abs / static_cast<float>(max_int);

        float sum_sq = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
            for (float v : coeffs_channels[ch][l]) {
                sum_sq += v * v;
            }
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(total_samples) + 1e-12f);
        float deadzone = compute_energy_adaptive_deadzone(step, rms);

        std::vector<int32_t> quantized;
        quantized.reserve(total_samples);

        for (int ch = 0; ch < num_channels; ++ch) {
            const auto& arr = coeffs_channels[ch][l];
            for (float v : arr) {
                int32_t q;
                float av = std::abs(v);

                if (bbits == 1) {
                    float v_clip = (sgn == 1) ? std::max(0.0f, v) : std::max(0.0f, -v);
                    q = static_cast<int32_t>(std::round(v_clip / step));
                    q = std::clamp(q, 0, 1);
                } else {
                    if (av < deadzone * deadzone_scale) {
                        q = 0;
                    } else {
                        float scaled = v / step;
                        q = static_cast<int32_t>(std::round(scaled));
                    }
                }
                quantized.push_back(q);
            }
        }

        scales[l] = (bbits == 1) ? ((sgn == 1) ? max_abs : -max_abs) : max_abs;
        quantized_per_level[l] = std::move(quantized);
    }

    return { scales, quantized_per_level };
}

#endif // QUANTIZE_H