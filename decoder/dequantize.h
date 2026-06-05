#ifndef DEQUANTIZE_H
#define DEQUANTIZE_H

#include <vector>
#include <cmath>
#include <algorithm>

inline void dequantize_band(
    const std::vector<int32_t>& quantized,
    float step,
    std::vector<float>& out_band)
{
    out_band.resize(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i)
        out_band[i] = quantized[i] * step;
}

#endif // DEQUANTIZE_H
