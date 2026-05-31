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

    void reconstruct_block(float* data, size_t N) const {
        if (N < 2) return;
        const size_t even_len = (N + 1) >> 1;
        const size_t odd_len  = N >> 1;
        float* even = data;
        float* odd  = data + even_len;

        if (scratch_ext_d.size() < odd_len + 4)
            scratch_ext_d.resize(odd_len + 4);
        float* pd = scratch_ext_d.data() + 2;
        std::memcpy(pd, odd, odd_len * sizeof(float));
        pd[-1] = odd[0];
        pd[-2] = odd[1];
        pd[odd_len] = odd[odd_len - 1];
        pd[odd_len + 1] = odd[odd_len - 2];

        const float INV16 = 0.0625f;
        for (size_t i = 0; i < even_len; ++i) {
            float* p = pd + i;
            even[i] -= (-p[-2] + 5 * p[-1] + 5 * p[0] - p[1]) * INV16;
        }

        if (scratch_ext_e.size() < even_len + 6)
            scratch_ext_e.resize(even_len + 6);
        float* pe = scratch_ext_e.data() + 3;
        std::memcpy(pe, even, even_len * sizeof(float));
        pe[-1] = even[0];
        pe[-2] = even[1];
        pe[-3] = even[2];
        pe[even_len] = even[even_len - 1];
        pe[even_len + 1] = even[even_len - 2];
        pe[even_len + 2] = even[even_len - 3];

        const float INV256 = 1.0f / 256.0f;
        for (size_t i = 0; i < odd_len; ++i) {
            float* p = pe + i;
            float pred = (3 * p[-2] - 25 * p[-1] + 150 * p[0] + 150 * p[1] - 25 * p[2] + 3 * p[3]) * INV256;
            odd[i] += pred;
        }

        if (tmp.size() < N) tmp.resize(N);
        for (size_t i = 0; i < even_len; ++i) tmp[i << 1] = even[i];
        for (size_t i = 0; i < odd_len; ++i) tmp[(i << 1) + 1] = odd[i];
        std::memcpy(data, tmp.data(), N * sizeof(float));
    }

public:
    std::vector<float> wpt_reconstruct(std::vector<std::vector<float>> tree, int level) {
        size_t bands = tree.size();
        size_t band_size = tree[0].size();
        std::vector<float> data(bands * band_size);
        for (size_t i = 0; i < bands; ++i) {
            std::memcpy(data.data() + i * band_size, tree[i].data(), band_size * sizeof(float));
        }
        size_t N = data.size();
        for (int L = level - 1; L >= 0; --L) {
            size_t step = N >> L;
            for (size_t offset = 0; offset < N; offset += step) {
                reconstruct_block(data.data() + offset, step);
            }
        }
        return data;
    }
};

class PWPT : public DD64 {
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
    std::vector<float> iwpt(std::vector<std::vector<float>>& subbands, float sr, int levels) {
        if (sr >= 44100) {
            for (size_t i = 12; i < subbands.size(); ++i) {
                for (float& x : subbands[i]) x *= 4.0f;
            }
        }
        if (levels > 1) {
            int total = 1 << levels;
            auto perm = gray_permutation(levels);
            std::vector<std::vector<float>> reordered(total);
            for (int i = 0; i < total; ++i)
                reordered[i] = std::move(subbands[perm[i]]);
            subbands = std::move(reordered);
        }
        for (auto& band : subbands) {
            for (float& coeff : band) {
                float absv = std::fabs(coeff);
                coeff = std::copysign(fast_pow_1333(absv), coeff);
            }
        }
        return wpt_reconstruct(subbands, levels);
    }
};

#endif // WAVELET_H