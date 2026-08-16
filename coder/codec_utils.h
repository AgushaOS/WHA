#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <climits>

inline int get_scale_bits(int band_idx) {
    if (band_idx < 1) return 12;
    return 8;
}

inline int compute_optimal_rice_k(const std::vector<uint32_t>& vals, int max_k = 7) {
    int    best_k    = 0;
    size_t best_bits = SIZE_MAX;
    for (int k = 0; k <= max_k; ++k) {
        size_t total = 0;
        for (uint32_t v : vals)
            total += (v >> k) + 1 + k;
        if (total < best_bits) {
            best_bits = total;
            best_k    = k;
        }
    }
    return best_k;
}

inline size_t estimate_rice_bits(const std::vector<uint32_t>& vals, int k) {
    size_t total = 0;
    for (uint32_t v : vals)
        total += (v >> k) + 1 + k;
    return total;
}

inline std::vector<uint8_t> pack_bits(const std::vector<bool>& bits, int start, int end) {
    int count = end - start;
    int bytes = (count + 7) / 8;
    std::vector<uint8_t> packed(bytes, 0);
    for (int i = start; i < end; ++i) {
        if (bits[i]) {
            int idx = i - start;
            packed[idx / 8] |= (1 << (idx % 8));
        }
    }
    return packed;
}

inline float scale_and_bits_to_step(float scale, int bits) {
    if (bits <= 0) return 0.0f;
    int max_int = (bits > 1) ? ((1 << (bits - 1)) - 1) : 1;
    return scale / max_int;
}

static constexpr float SCALE_LOG_MIN = -6.0f;
static constexpr float SCALE_LOG_MAX =  0.0f;

inline uint32_t get_scale_idx(float step, int sb) {
    if (step <= 0.0f) return 0;
    float log_s   = log10f(std::max(step, 1e-12f));
    int   max_idx = (1 << sb) - 1;
    int   idx     = (int)((log_s - SCALE_LOG_MIN) * max_idx / (SCALE_LOG_MAX - SCALE_LOG_MIN));
    idx = std::clamp(idx, 0, max_idx);
    return (uint32_t)idx;
}