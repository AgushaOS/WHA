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
    mutable std::vector<float> tmp;

    void reconstruct_block(float* __restrict data, size_t N) const {
        if (N < 2) return;

        const size_t even_len = (N + 1) >> 1;
        const size_t odd_len  = N >> 1;

        float* __restrict even = data;
        float* __restrict odd  = data + even_len;

        if (scratch_ext.size() < even_len + 16)
            scratch_ext.resize(even_len + 16);

        float* __restrict pd = scratch_ext.data() + 3;

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

            even[i] -= upd;
        }

        float* __restrict pe = scratch_ext.data() + 8;

        std::memcpy(pe, even, even_len * sizeof(float));

        pe[-1] = even[0];
        pe[-2] = even[1];
        pe[-3] = even[2];
        pe[-4] = even[3];
        pe[-5] = even[4];
        pe[-6] = even[5];
        pe[-7] = even[6];
        pe[-8] = even[7];

        pe[even_len]     = even[even_len - 1];
        pe[even_len + 1] = even[even_len - 2];
        pe[even_len + 2] = even[even_len - 3];
        pe[even_len + 3] = even[even_len - 4];
        pe[even_len + 4] = even[even_len - 5];
        pe[even_len + 5] = even[even_len - 6];
        pe[even_len + 6] = even[even_len - 7];
        pe[even_len + 7] = even[even_len - 8];

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

            odd[i] += pred;
        }

        if (tmp.size() < N)
            tmp.resize(N);

        float* __restrict dst = tmp.data();

        for (size_t i = 0; i < even_len; ++i)
            dst[i << 1] = even[i];

        for (size_t i = 0; i < odd_len; ++i)
            dst[(i << 1) + 1] = odd[i];

        std::memcpy(data, dst, N * sizeof(float));
    }

    void reconstruct_tree(float* __restrict data,
                          size_t N,
                          int level) const {
        for (int L = level - 1; L >= 0; --L) {
            const size_t step = N >> L;

            for (size_t offset = 0; offset < N; offset += step)
                reconstruct_block(data + offset, step);
        }
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
        const int total_bands = 1 << levels;

        if (total_bands == 0 || subbands.empty())
            return {};

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

        std::vector<float> data(N);

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

        return data;
    }
};

#endif // WAVELET_H