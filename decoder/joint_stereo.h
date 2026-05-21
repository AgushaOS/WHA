#ifndef JOINT_STEREO_H
#define JOINT_STEREO_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

inline void ms_to_lr(const std::vector<float>& mid, const std::vector<float>& side,
                     std::vector<float>& left, std::vector<float>& right) {
    size_t n = mid.size();
    left.resize(n);
    right.resize(n);
    for (size_t i = 0; i < n; ++i) {
        left[i] = mid[i] + side[i];
        right[i] = mid[i] - side[i];
    }
}

inline int get_is_start_band(float target_kbps) {
    if (target_kbps < 128.0f) return 3;
    if (target_kbps < 160.0f) return 4;
    if (target_kbps < 190.0f) return 8;
    return 999;
}

inline float is_scale_from_idx(uint8_t idx) {
    return (idx / 255.0f) * 4.0f;
}

#endif // JOINT_STEREO_H