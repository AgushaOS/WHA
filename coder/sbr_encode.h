#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "codec_utils.h"
#include "pred_state.h"
#include "entropy_encoder.h"

extern float SBR_W0;
extern float SBR_W1;

inline void get_sbr_shape_weights(uint8_t mask, float w0, float w1, float out[4]) {
    float sum = 0.0f;

    for (int i = 0; i < 4; ++i) {
        out[i] = (mask & (1 << i)) ? w1 : w0;
        sum += out[i];
    }

    const float inv_sum = 1.0f / sum;

    for (int i = 0; i < 4; ++i)
        out[i] *= inv_sum;
}

struct SBREncodeResult {
    std::vector<bool>     noise_flag;
    std::vector<uint32_t> rms_idx;
    std::vector<uint8_t>  shape_mask;
    int                   sbr_end = 0;
};

inline SBREncodeResult sbr_analyze(
    const std::vector<std::vector<float>>& ch0_original,
    const std::vector<bool>&               active0,
    int sbr_end,
    int band_count,
    float w0,
    float w1)
{
    SBREncodeResult result;

    result.noise_flag.assign(band_count, false);
    result.rms_idx.assign(band_count, 0);
    result.shape_mask.assign(band_count, 0);
    result.sbr_end = sbr_end;

    float sum_by_pop[5];

    for (int p = 0; p <= 4; ++p)
        sum_by_pop[p] = w0 * (float)(4 - p) + w1 * (float)p;

    float w_all[16][4];

    for (int m = 0; m < 16; ++m) {
        const int pc = __builtin_popcount((unsigned)m);
        const float inv_s = 1.0f / sum_by_pop[pc];

        for (int b = 0; b < 4; ++b) {
            w_all[m][b] =
                ((m >> b) & 1) ? (w1 * inv_s) : (w0 * inv_s);
        }
    }

    const float eps = 1e-12f;

    for (int i = 0; i < sbr_end; ++i) {
        if (active0[i])
            continue;

        const auto& orig = ch0_original[i];

        if (orig.empty())
            continue;

        const int N = (int)orig.size();
        const int seg_size = N / 4;

        float seg_max[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int s = 0; s < 4; ++s) {
            const int start = s * seg_size;
            const int end = (s == 3) ? N : (s + 1) * seg_size;

            float max_abs = 0.0f;

            for (int j = start; j < end; ++j) {
                const float av = fabsf(orig[j]);

                if (av > max_abs)
                    max_abs = av;
            }

            seg_max[s] = max_abs;
        }

        const float total_max =
            seg_max[0] + seg_max[1] +
            seg_max[2] + seg_max[3] + eps;

        float target_p[4];

        for (int s = 0; s < 4; ++s)
            target_p[s] = seg_max[s] / total_max;

        float best_err = 1e9f;
        uint8_t best_mask = 0;

        for (int m = 0; m < 16; ++m) {
            float err = 0.0f;

            for (int s = 0; s < 4; ++s) {
                const float diff = target_p[s] - w_all[m][s];
                err += diff * diff;
            }

            if (err < best_err) {
                best_err = err;
                best_mask = (uint8_t)m;
            }
        }

        // std::cout << int64_t(best_mask) << '\n';

        result.shape_mask[i] = best_mask;

        double sum_sq = 0.0;
        float max_abs = 0.0f;

        for (float v : orig) {
            sum_sq += static_cast<double>(v) * v;

            const float av = fabsf(v);

            if (av > max_abs)
                max_abs = av;
        }

        const float rms =
            sqrtf(static_cast<float>(sum_sq / N) + eps);

        const float crest = max_abs / (rms + eps);

        result.noise_flag[i] = (crest < 3.5f);

        const int sb = get_scale_bits(i);

        result.rms_idx[i] = get_scale_idx(rms, sb);
        // std::cout << result.rms_idx[i] << ' ';
    }
    // std::cout << '\n';

    return result;
}

inline void sbr_write_huffman(BitWriterMSB& writer, uint8_t mask)
{
    switch (mask) {
        case 0:  writer.write_bits(0b1,        1); break;
        case 1:  writer.write_bits(0b01011,    5); break;
        case 2:  writer.write_bits(0b00000,    5); break;
        case 3:  writer.write_bits(0b01001,    5); break;
        case 4:  writer.write_bits(0b00001,    5); break;
        case 5:  writer.write_bits(0b01110010, 8); break;
        case 6:  writer.write_bits(0b011101,   6); break;
        case 7:  writer.write_bits(0b001,      3); break;
        case 8:  writer.write_bits(0b0001,     4); break;
        case 9:  writer.write_bits(0b01110011, 8); break;
        case 10: writer.write_bits(0b0111000,  7); break;
        case 11: writer.write_bits(0b01010,    5); break;
        case 12: writer.write_bits(0b01111,    5); break;
        case 13: writer.write_bits(0b01000,    5); break;
        case 14: writer.write_bits(0b0110,     4); break;
        default: writer.write_bits(0b1,        1); break;
    }
}

inline void sbr_write_to_stream(
    BitWriterMSB& writer,
    const SBREncodeResult& sbr,
    const std::vector<int>& sbr_indices,
    bool can_state_pred,
    const ModeState& pred_state)
{
    if (sbr_indices.empty())
        return;

    for (int idx : sbr_indices) {
        const bool cur = sbr.noise_flag[idx];
        bool write_bit = cur;

        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            write_bit = cur ^ (pred_state.prev_sbr_noise[idx] != 0);
        }

        writer.write_bits(write_bit ? 1 : 0, 1);
    }

    for (int idx : sbr_indices)
        sbr_write_huffman(writer, sbr.shape_mask[idx]);

    std::vector<uint32_t> sbr_vals;
    sbr_vals.reserve(sbr_indices.size());

    for (int idx : sbr_indices) {
        const uint32_t cur = sbr.rms_idx[idx];
        uint32_t val = cur;

        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            const int32_t d =
                (int32_t)cur - (int32_t)pred_state.prev_sbr_rms[idx];

            val = zigzag_encode(d);
        }

        sbr_vals.push_back(val);
    }

    const int k_sbr = compute_optimal_rice_k(sbr_vals, 7);

    writer.write_bits(k_sbr & 0x07, 3);
    rice_encode(writer, sbr_vals, k_sbr);
}

inline void sbr_write_to_stream1(
    BitWriterMSB& writer,
    const SBREncodeResult& sbr,
    const std::vector<int>& sbr_indices,
    bool can_state_pred,
    const ModeState& pred_state)
{
    if (sbr_indices.empty())
        return;

    for (int idx : sbr_indices) {
        const bool cur = sbr.noise_flag[idx];
        bool write_bit = cur;

        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            write_bit = cur ^ (pred_state.prev_sbr_noise[idx] != 0);
        }

        writer.write_bits(write_bit ? 1 : 0, 1);
    }

    for (int idx : sbr_indices)
        writer.write_bits(sbr.shape_mask[idx], 4);

    std::vector<uint32_t> sbr_vals;
    sbr_vals.reserve(sbr_indices.size());

    for (int idx : sbr_indices) {
        const uint32_t cur = sbr.rms_idx[idx];
        uint32_t val = cur;

        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            const int32_t d =
                (int32_t)cur - (int32_t)pred_state.prev_sbr_rms[idx];

            val = zigzag_encode(d);
        }

        sbr_vals.push_back(val);
    }

    const int k_sbr = compute_optimal_rice_k(sbr_vals, 7);

    writer.write_bits(k_sbr & 0x07, 3);
    rice_encode(writer, sbr_vals, k_sbr);
}
