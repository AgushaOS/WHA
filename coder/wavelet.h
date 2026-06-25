#ifndef WAVELET_H
#define WAVELET_H

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

class DD64 {
protected:
    mutable std::vector<float> scratch_ext_e;
    mutable std::vector<float> scratch_ext_d;
    mutable std::vector<float> tmp;

    void decompose_block(float* data, size_t N) const {
        if (N < 2) return;

        const size_t even_len = (N + 1) >> 1;
        const size_t odd_len  = N >> 1;

        if (tmp.size() < N)
            tmp.resize(N);

        float* even = tmp.data();
        float* odd  = tmp.data() + even_len;

        for (size_t i = 0; i < even_len; ++i)
            even[i] = data[i << 1];

        for (size_t i = 0; i < odd_len; ++i)
            odd[i] = data[(i << 1) + 1];

        if (scratch_ext_e.size() < even_len + 12)
            scratch_ext_e.resize(even_len + 12);

        float* pe = scratch_ext_e.data() + 6;

        std::memcpy(pe, even, even_len * sizeof(float));

        pe[-1] = even[0];
        pe[-2] = even[1];
        pe[-3] = even[2];
        pe[-4] = even[3];
        pe[-5] = even[4];
        pe[-6] = even[5];

        pe[even_len]     = even[even_len - 1];
        pe[even_len + 1] = even[even_len - 2];
        pe[even_len + 2] = even[even_len - 3];
        pe[even_len + 3] = even[even_len - 4];
        pe[even_len + 4] = even[even_len - 5];
        pe[even_len + 5] = even[even_len - 6];

        constexpr float C0 = -0.0001201630f;  
        constexpr float C1 =  0.0016155243f;  
        constexpr float C2 = -0.0103855133f;  
        constexpr float C3 =  0.0436191559f;  
        constexpr float C4 = -0.1453971863f; 
        constexpr float C5 =  0.6106681824f;

        for (size_t i = 0; i < odd_len; ++i) {
            float* p = pe + i;

            float pred =
                C0 * (p[-5] + p[6]) +
                C1 * (p[-4] + p[5]) +
                C2 * (p[-3] + p[4]) +
                C3 * (p[-2] + p[3]) +
                C4 * (p[-1] + p[2]) +
                C5 * (p[0]  + p[1]);

            odd[i] -= pred;
        }

        if (scratch_ext_d.size() < odd_len + 6)
            scratch_ext_d.resize(odd_len + 6);

        float* pd = scratch_ext_d.data() + 3;

        std::memcpy(pd, odd, odd_len * sizeof(float));

        pd[-1] = odd[0];
        pd[-2] = odd[1];
        pd[-3] = odd[2];

        pd[odd_len]     = odd[odd_len - 1];
        pd[odd_len + 1] = odd[odd_len - 2];
        pd[odd_len + 2] = odd[odd_len - 3];

        constexpr float INV64 = 1.0f / 64.0f;

        for (size_t i = 0; i < even_len; ++i) {
            float* p = pd + i;

            float upd =
                (  1.0f * p[-3]
                 - 6.0f * p[-2]
                 +21.0f * p[-1]
                 +21.0f * p[0]
                 - 6.0f * p[1]
                 + 1.0f * p[2]) * INV64;

            even[i] += upd;
        }

        std::memcpy(data, even, even_len * sizeof(float));
        std::memcpy(data + even_len, odd, odd_len * sizeof(float));
    }

public:
    std::vector<std::vector<float>> wpt_decompose(const std::vector<float>& signal, int level) {
        std::vector<float> data = signal;
        const size_t N = data.size();
        for (int L = 0; L < level; ++L) {
            size_t step = N >> L;
            for (size_t offset = 0; offset < N; offset += step)
                decompose_block(data.data() + offset, step);
        }
        std::vector<std::vector<float>> tree(1 << level);
        size_t band_size = N >> level;
        for (size_t i = 0; i < tree.size(); ++i)
            tree[i].assign(data.begin() + i * band_size, data.begin() + (i + 1) * band_size);
        return tree;
    }
};

class PWPT : public DD64 {
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
    std::vector<std::vector<float>> wpt(const std::vector<float>& signal, int levels, float bitrate, int sr) {
        auto subbands = wpt_decompose(signal, levels);
        for (auto& band : subbands)
            for (float& coeff : band) {
                float absv = std::fabs(coeff);
                coeff = std::copysign(fast_pow_075(absv), coeff);

            }
        if (levels > 1) {
            int total = 1 << levels;
            auto perm = gray_permutation(levels);
            std::vector<std::vector<float>> ordered(total);
            for (int i = 0; i < total; ++i)
                ordered[perm[i]] = std::move(subbands[i]);
            subbands = std::move(ordered);
        }
        if (sr >= 44100) {
            for (size_t i = 12; i < subbands.size(); ++i)
                for (float& x : subbands[i])
                    x *= 0.25f;
        }
        return subbands;
    }
};

#endif // WAVELET_H