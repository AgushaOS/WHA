#ifndef WAVELET_H
#define WAVELET_H

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

class DD64in {
protected:
    mutable std::vector<float> scratch_ext;
    mutable std::vector<float> tmp;

    void decompose_block(float* __restrict data, size_t N) const {
        if (N < 2) return;

        const size_t even_len = (N + 1) >> 1;
        const size_t odd_len  = N >> 1;

        if (tmp.size() < N)
            tmp.resize(N);

        float* __restrict even = tmp.data();
        float* __restrict odd  = tmp.data() + even_len;

        for (size_t i = 0; i < even_len; ++i)
            even[i] = data[i << 1];

        for (size_t i = 0; i < odd_len; ++i)
            odd[i] = data[(i << 1) + 1];

        if (scratch_ext.size() < even_len + 16)
            scratch_ext.resize(even_len + 16);

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

            odd[i] -= pred;
        }

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

            even[i] += upd;
        }

        std::memcpy(data, even, even_len * sizeof(float));
        std::memcpy(data + even_len, odd, odd_len * sizeof(float));
    }

public:
    std::vector<float>& wpt_decompose(std::vector<float>& signal, int level) {
        const size_t N = signal.size();
        float* __restrict data = signal.data();

        for (int L = 0; L < level; ++L) {
            size_t step = N >> L;
            for (size_t offset = 0; offset < N; offset += step)
                decompose_block(data + offset, step);
        }
        return signal;
    }
};

class PWPT : public DD64in {
private:
    static inline float fast_pow_075(float x) {
        float s = std::sqrt(x);
        return s * std::sqrt(s);
    }

    std::vector<int> gray_permutation(int level) const {
        int n = 1 << level;
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i)
            perm[i] = i ^ (i >> 1);
        return perm;
    }

public:
    std::vector<std::vector<float>> wpt(const std::vector<float>& signal,
                                        int levels,
                                        float bitrate,
                                        int sr, 
                                        int channels) {
        std::vector<float> data = signal;
        wpt_decompose(data, levels);

        const size_t N = data.size();
        const int total_bands = 1 << levels;
        const size_t band_size = N >> levels;  

        // for (float& x : data) {
        //     float absv = std::fabs(x);
        //     x = std::copysign(fast_pow_075(absv), x);
        // }

        if (levels > 1) {
            auto perm = gray_permutation(levels);
            std::vector<float> ordered(N);

            for (int i = 0; i < total_bands; ++i) {
                const size_t src = i * band_size;
                const size_t dst = perm[i] * band_size;
                std::copy(data.begin() + src,
                          data.begin() + src + band_size,
                          ordered.begin() + dst);
            }
            data.swap(ordered);
        }

        if (sr >= 44100) {
            for (int i = 12; i < total_bands; ++i) {
                float* begin = data.data() + i * band_size;
                float* end   = begin + band_size;
                for (float* p = begin; p != end; ++p)
                    *p *= 0.25f;
            }
            if (bitrate / float(channels) < 48) {
                for (int i = 8; i < total_bands; ++i) {
                float* begin = data.data() + i * band_size;
                float* end   = begin + band_size;
                for (float* p = begin; p != end; ++p)
                    *p *= 0.0f;
                }
            }
            if (bitrate / float(channels) < 32) {
                for (int i = 4; i < total_bands; ++i) {
                float* begin = data.data() + i * band_size;
                float* end   = begin + band_size;
                for (float* p = begin; p != end; ++p)
                    *p *= 0.0f;
                }
            }
        }

        std::vector<std::vector<float>> tree(total_bands);
        for (int i = 0; i < total_bands; ++i) {
            tree[i].assign(data.begin() + i * band_size,
                           data.begin() + (i + 1) * band_size);
        }
        return tree;
    }
};

#endif // WAVELET_H