#ifndef WAVELET_H
#define WAVELET_H

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>

class DD64out {
protected:
    mutable std::vector<float> scratch_ext;
    mutable std::vector<float> scratch_even;
    mutable std::vector<float> tmp;

    void reconstruct_block(const float* __restrict source,
                           float* __restrict destination,
                           size_t N) const {
        if (N < 2) return;

        const size_t even_len = (N + 1) >> 1;
        const size_t odd_len  = N >> 1;

        const size_t scratch_size = even_len + 16;
        if (scratch_ext.size() < scratch_size)
            scratch_ext.resize(scratch_size);
        if (scratch_even.size() < scratch_size)
            scratch_even.resize(scratch_size);

        float* __restrict pd = scratch_ext.data() + 3;
        float* __restrict even = scratch_even.data() + 8;
        const float* __restrict odd = source + even_len;

        std::memcpy(pd, odd, odd_len * sizeof(float));

        pd[-1] = odd[0];
        pd[-2] = odd[1];
        pd[-3] = odd[2];

        pd[odd_len]     = odd[odd_len - 1];
        pd[odd_len + 1] = odd[odd_len - 2];
        pd[odd_len + 2] = odd[odd_len - 3];

        constexpr float INV64 = 1.0f / 64.0f;

        for (size_t i = 0; i < even_len; ++i) {
            float* __restrict p = pd + i;

            float upd =
                ((p[-3] + p[2])
                 - 6.0f  * (p[-2] + p[1])
                 + 21.0f * (p[-1] + p[0])) * INV64;

            even[i] = source[i] - upd;
        }

        float* __restrict pe = even;
        pe[-1] = pe[0];
        pe[-2] = pe[1];
        pe[-3] = pe[2];
        pe[-4] = pe[3];
        pe[-5] = pe[4];
        pe[-6] = pe[5];
        pe[-7] = pe[6];
        pe[-8] = pe[7];

        pe[even_len]     = pe[even_len - 1];
        pe[even_len + 1] = pe[even_len - 2];
        pe[even_len + 2] = pe[even_len - 3];
        pe[even_len + 3] = pe[even_len - 4];
        pe[even_len + 4] = pe[even_len - 5];
        pe[even_len + 5] = pe[even_len - 6];
        pe[even_len + 6] = pe[even_len - 7];
        pe[even_len + 7] = pe[even_len - 8];

        constexpr float C0 = -0.0000063926f;
        constexpr float C1 =  0.0001106411f;
        constexpr float C2 = -0.0009153038f;
        constexpr float C3 =  0.0048477203f;
        constexpr float C4 = -0.0186983496f;
        constexpr float C5 =  0.0575909168f;
        constexpr float C6 = -0.1599747688f;
        constexpr float C7 =  0.6170455217f;

        for (size_t i = 0; i < odd_len; ++i) {
            float* __restrict p = pe + i;

            float pred =
                C0 * (p[-7] + p[8]) +
                C1 * (p[-6] + p[7]) +
                C2 * (p[-5] + p[6]) +
                C3 * (p[-4] + p[5]) +
                C4 * (p[-3] + p[4]) +
                C5 * (p[-2] + p[3]) +
                C6 * (p[-1] + p[2]) +
                C7 * (p[0]  + p[1]);

            destination[(i << 1) + 1] = odd[i] + pred;
        }

        for (size_t i = 0; i < even_len; ++i)
            destination[i << 1] = pe[i];
    }

    void reconstruct_tree(float* __restrict data,
                          size_t N,
                          int level) const {
        if (level <= 0 || N < 2) return;

        if (tmp.size() < N)
            tmp.resize(N);

        float* source = data;
        float* destination = tmp.data();

        for (int L = level - 1; L >= 0; --L) {
            const size_t step = N >> L;

            for (size_t offset = 0; offset < N; offset += step)
                reconstruct_block(source + offset, destination + offset, step);

            std::swap(source, destination);
        }

        if (source != data)
            std::memcpy(data, source, N * sizeof(float));
    }

public:

    std::vector<float> wpt_reconstruct_flat(
        std::vector<float>& data,
        int level)
    {
        reconstruct_tree(data.data(), data.size(), level);
        return std::move(data);
    }
};


class PWPT : public DD64out {
private:

    static inline int gray_code(int i) {
        return i ^ (i >> 1);
    }

public:

    std::vector<float> iwpt(
        std::vector<std::vector<float>>& subbands,
        float sr,
        int levels,
        float target_kbps,
        float channels)
    {
        std::vector<float> data;
        iwpt_into(subbands, sr, levels, target_kbps, channels, data);
        return data;
    }

    void iwpt_into(
        std::vector<std::vector<float>>& subbands,
        float sr,
        int levels,
        float target_kbps,
        float channels,
        std::vector<float>& data)
    {
        const int total_bands = 1 << levels;

        if (total_bands == 0 || subbands.empty()) {
            data.clear();
            return;
        }

        const size_t band_size = subbands[0].size();
        const size_t N = size_t(total_bands) * band_size;

        float attenuation = 1.0f;

        int atten_start = total_bands;

        if (sr >= 44100.0f) {
            atten_start = total_bands * 3 / 4;

            attenuation = 8.0f;

            const float bitrate_per_channel =
                target_kbps / channels;

            if (bitrate_per_channel >= 32.0f && bitrate_per_channel < 64.0f) {
                attenuation *= 8.0f;
            }
        }

        data.resize(N);

        float* __restrict dst = data.data();

        if (levels > 1) {

            for (int i = 0; i < total_bands; ++i) {
                const size_t dst_offset =
                    size_t(gray_code(i)) * band_size;

                const float* __restrict src =
                    subbands[i].data();

                float* __restrict out =
                    dst + dst_offset;

                if (i >= atten_start && attenuation != 1.0f) {
                    for (size_t j = 0; j < band_size; ++j)
                        out[j] = src[j] * attenuation;
                } else {
                    std::memcpy(out, src, band_size * sizeof(float));
                }
            }
        } else {
            std::memcpy(dst, subbands[0].data(), band_size * sizeof(float));

            if (attenuation != 1.0f) {
                for (size_t i = 0; i < band_size; ++i)
                    dst[i] *= attenuation;
            }
        }

        reconstruct_tree(dst, N, levels);
    }
};

#endif // WAVELET_H