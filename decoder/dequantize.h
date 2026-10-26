#ifndef DEQUANTIZE_H
#define DEQUANTIZE_H

#include <vector>
#include <cmath>
#include <algorithm>

inline void dequantize_band(
    std::vector<int32_t>& quantized,
    float step,
    std::vector<float>& out_band,
    float target_kbps,
    float true_num_channels)
{
    if (target_kbps / true_num_channels < 96) {
        step *= 16.0f;
    }
    out_band.resize(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i)
        out_band[i] = quantized[i] * step;
    // std::cout << step << '\n';
}

#endif // DEQUANTIZE_H

