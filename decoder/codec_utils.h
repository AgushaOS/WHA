#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

static constexpr float SCALE_LOG_MIN = -6.0f;
static constexpr float SCALE_LOG_MAX =  0.0f;

inline int get_scale_bits(int band_idx) {
    if (band_idx < 1) return 12;
    return 8;
}

inline std::vector<bool> unpack_bits(const uint8_t* data, int bytes, int total_bits) {
    std::vector<bool> bits(total_bits, false);
    for (int i = 0; i < total_bits; ++i) {
        int byte_idx = i / 8;
        if (byte_idx >= bytes) break;
        bits[i] = (data[byte_idx] >> (i % 8)) & 1;
    }
    return bits;
}

inline std::vector<int> compute_band_shapes(int block_size, int level) {
    std::vector<int> shapes = {block_size};
    for (int l = 0; l < level; ++l) {
        std::vector<int> new_shapes;
        new_shapes.reserve(shapes.size() * 2);
        for (int s : shapes) {
            new_shapes.push_back((s + 1) >> 1);
            new_shapes.push_back(s >> 1);
        }
        shapes = std::move(new_shapes);
    }
    return shapes;
}

inline float idx_to_step(uint32_t idx, int sb) {
    int max_idx = (1 << sb) - 1;
    if (idx > (uint32_t)max_idx) idx = max_idx;
    float log_step = SCALE_LOG_MIN + idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
    return exp10f(log_step);
}