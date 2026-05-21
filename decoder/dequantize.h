#ifndef DEQUANTIZE_H
#define DEQUANTIZE_H

#include <vector>
#include <cmath>
#include <algorithm>

inline void dequantize_band(
    const std::vector<int32_t>& quantized,
    int bits_per_sample,
    float scale,
    std::vector<float>& out_band)
{
    out_band.resize(quantized.size());

    int max_int = (bits_per_sample > 1)
        ? ((1 << (bits_per_sample - 1)) - 1)
        : 1;

    float factor = scale / max_int;

    static uint32_t rng_state = 0xA341316Cu;

    auto rand01 = [&]() -> float {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        return (rng_state & 0x00FFFFFF) / 16777216.0f;
    };

    auto dither = [&]() -> float {
        float n1 = rand01();
        float n2 = rand01();
        return (n1 - n2);
    };

    float dither_amp = 0.0f;
    if (bits_per_sample <= 3)
        dither_amp = scale * 0.0025f;

    for (size_t i = 0; i < quantized.size(); ++i)
    {
        float v = quantized[i] * factor;

        if (dither_amp > 0.0f)
            v += dither() * dither_amp;

        out_band[i] = v;
    }
}

#endif // DEQUANTIZE_H
