#ifndef BIT_ALLOC_H
#define BIT_ALLOC_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <climits>

struct DualAllocResult {
    std::vector<int> bits0;
    std::vector<int> bits1;
    int used_bits;
    int new_reservoir;
};

inline DualAllocResult allocate_bits_dual(
    const std::vector<float>& priority0,
    const std::vector<float>& priority1,
    const std::vector<int>& coeff_counts,
    int total_bits_budget,
    std::vector<int>& min_bits_in,
    std::vector<int>& max_bits_in,
    std::vector<float>& energy0,
    std::vector<float>& energy1,
    int reservoir,
    int reservoir_max,
    float target_kbps)
{
    const int n = static_cast<int>(priority0.size());
    const float eps = 1e-12f;

    static thread_local std::vector<int> bits0, bits1, max_b, active;
    static thread_local std::vector<float> s2_0, s2_1, factor0, factor1, global_score;
    bits0.assign(min_bits_in.begin(), min_bits_in.end());
    bits1.assign(min_bits_in.begin(), min_bits_in.end());
    max_b.resize(n);
    s2_0.resize(n); s2_1.resize(n);
    factor0.resize(n); factor1.resize(n);
    global_score.resize(n);
    active.assign(n, 1);

    if (target_kbps < 128) {
        bits0[0] = 4; bits1[0] = 4;
    } else {
        bits0[0] = 5; bits1[0] = 4;
    }

    for (int i = 0; i < n; i++) {
        max_b[i] = std::min(std::max(max_bits_in[i], 0), 31);
        float inv = 1.0f / (coeff_counts[i] + eps);
        s2_0[i] = energy0[i] * inv;
        s2_1[i] = energy1[i] * inv;
        float sp0 = 0.5f * priority0[i];
        float sp1 = 0.5f * priority1[i];
        // if (i > 0) { sp0 += 0.25f * priority0[i-1]; sp1 += 0.25f * priority1[i-1]; }
        // if (i < n-1) { sp0 += 0.25f * priority0[i+1]; sp1 += 0.25f * priority1[i+1]; }
        factor0[i] = s2_0[i] * std::pow(std::max(sp0, eps), 0.6f);
        factor1[i] = s2_1[i] * std::pow(std::max(sp1, eps), 0.6f);
        global_score[i] = (factor0[i] + factor1[i]) * std::sqrt((float)coeff_counts[i]);
    }

    int used_bits = 0;
    for (int i = 0; i < n; i++) used_bits += (bits0[i] + bits1[i]) * coeff_counts[i];
    float bits_per_coeff = total_bits_budget / (float)(n + eps);
    int target_active = n;
    if (bits_per_coeff < 6.0f) target_active = std::max(1, (int)(n * 0.65f));
    if (bits_per_coeff < 4.0f) target_active = std::max(1, (int)(n * 0.45f));
    if (bits_per_coeff < 3.0f) target_active = std::max(1, (int)(n * 0.30f));

    if (target_active < n) {
        std::vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return global_score[a] > global_score[b]; });
        for (int i = target_active; i < n; i++) {
            active[idx[i]] = 0;
            bits0[idx[i]] = 0;
            bits1[idx[i]] = 0;
        }
    }

    static thread_local std::vector<float> exp2_table, decay_table;
    if (exp2_table.empty()) {
        exp2_table.resize(33);
        decay_table.resize(33);
        for (int b = 0; b <= 32; b++) {
            exp2_table[b] = std::exp2f(-2.0f * b);
            decay_table[b] = 1.0f / (1.0f + 0.25f * b);
        }
    }

    struct GainEntry { float gain; int idx; bool is0; int next_bits; bool operator<(const GainEntry& other) const { return gain < other.gain; } };
    std::priority_queue<GainEntry> pq;
    auto push_gain = [&](int i, bool is0) {
        if (!active[i]) return;
        int bits = is0 ? bits0[i] : bits1[i];
        if (bits >= max_b[i]) return;
        int trial = (bits == 0 ? 2 : bits + 1);
        if (trial > max_b[i]) trial = max_b[i];
        int step = (trial - bits) * coeff_counts[i];
        if (used_bits + step > total_bits_budget) return;
        float next_D = (is0 ? s2_0[i] : s2_1[i]) * exp2_table[trial];
        float cur_D  = (is0 ? s2_0[i] : s2_1[i]) * exp2_table[bits];
        if (next_D / (cur_D + eps) > 0.25f) return;
        float decay_cur = decay_table[bits];
        float decay_next = decay_table[trial];
        float factor = is0 ? factor0[i] : factor1[i];
        float gain = (factor * decay_cur * exp2_table[bits] - factor * decay_next * exp2_table[trial]) *
                     std::max(is0 ? priority0[i] : priority1[i], eps) / std::sqrt(float(coeff_counts[i]));
        if (gain > 0.0f) pq.push({gain, i, is0, trial});
    };
    for (int i = 0; i < n; i++) { if (active[i]) { push_gain(i, true); push_gain(i, false); } }
    while (!pq.empty() && used_bits < total_bits_budget) {
        auto top = pq.top(); pq.pop();
        int i = top.idx; if (!active[i]) continue;
        bool is0 = top.is0;
        int cur_bits = is0 ? bits0[i] : bits1[i];
        int step_bits = (top.next_bits - cur_bits) * coeff_counts[i];
        if (used_bits + step_bits > total_bits_budget) continue;
        if (is0) bits0[i] = top.next_bits; else bits1[i] = top.next_bits;
        used_bits += step_bits;
        push_gain(i, is0);
    }
    if (used_bits > total_bits_budget) {
        for (int i = 0; i < n; i++) {
            while (used_bits > total_bits_budget) {
                if (bits0[i] > min_bits_in[i] && bits0[i] > 2) { bits0[i]--; used_bits -= coeff_counts[i]; }
                if (used_bits <= total_bits_budget) break;
                if (bits1[i] > min_bits_in[i] && bits1[i] > 2) { bits1[i]--; used_bits -= coeff_counts[i]; }
                if (used_bits <= total_bits_budget) break;
            }
        }
    }
    int new_reservoir = (used_bits <= total_bits_budget) ?
        std::min(reservoir_max, reservoir + (total_bits_budget - used_bits)) :
        std::max(0, reservoir - (used_bits - total_bits_budget));

    // std::cout << "CH0 : \n";
    // for (auto x : bits0) {
    //     std::cout << x << ' ';
    // }
    // std::cout << "\nCH1 : \n";
    // for (auto x : bits1) {
    //     std::cout << x << ' ';
    // }
    // std::cout << '\n';
    return { bits0, bits1, used_bits, new_reservoir };
}

#endif // BIT_ALLOC_H