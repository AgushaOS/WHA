#ifndef WAVELET_H
#define WAVELET_H

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

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

        for (size_t i = 0; i < even_len; ++i)
            tmp[i << 1] = even[i];

        for (size_t i = 0; i < odd_len; ++i)
            tmp[(i << 1) + 1] = odd[i];

        std::memcpy(data, tmp.data(), N * sizeof(float));
    }

public:
    std::vector<float> wpt_reconstruct(std::vector<std::vector<float>> tree, int level) {
        size_t bands = tree.size();
        if (bands == 0) return {};
        size_t band_size = tree[0].size();
        std::vector<float> data(bands * band_size);
        for (size_t i = 0; i < bands; ++i) {
            std::memcpy(data.data() + i * band_size,
                        tree[i].data(),
                        band_size * sizeof(float));
        }
        const size_t N = data.size();
        for (int L = level - 1; L >= 0; --L) {
            size_t step = N >> L;
            for (size_t offset = 0; offset < N; offset += step) {
                reconstruct_block(data.data() + offset, step);
            }
        }
        return data;
    }
};

class PWPT : public DD64out {
private:
    static inline float fast_pow_1333(float x) {
        return x * std::cbrt(x);
    }
    std::vector<int> gray_permutation(int level) const {
        int n = 1 << level;
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i)
            perm[i] = i ^ (i >> 1);
        return perm;
    }
public:
    std::vector<float> iwpt(std::vector<std::vector<float>>& subbands,
                            float sr,
                            int levels) {
        const int total_bands = 1 << levels;

        if (sr >= 44100) {
            int atten_start = total_bands * 3 / 4;
            for (size_t i = atten_start; i < subbands.size(); ++i) {
                for (float& x : subbands[i]) x *= 4.0f;
            }
        }

        if (levels > 1) {
            auto perm = gray_permutation(levels);
            std::vector<std::vector<float>> reordered(total_bands);
            for (int i = 0; i < total_bands; ++i)
                reordered[i] = std::move(subbands[perm[i]]);
            subbands = std::move(reordered);
        }

        return wpt_reconstruct(subbands, levels);
    }
};

#endif // WAVELET_H