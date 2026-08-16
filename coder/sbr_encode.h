#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "codec_utils.h"
#include "pred_state.h"
#include "entropy_encoder.h"

struct SBREncodeResult {
    std::vector<bool>     noise_flag;
    std::vector<uint32_t> rms_idx;
    int                   sbr_end = 0;
};

inline SBREncodeResult sbr_analyze(
    const std::vector<std::vector<float>>& ch0_original,
    const std::vector<bool>&               active0,
    int sbr_end, int band_count)
{
    SBREncodeResult result;
    result.noise_flag.assign(band_count, false);
    result.rms_idx.assign(band_count, 0);
    result.sbr_end = sbr_end;

    const float eps = 1e-12f;

    for (int i = 0; i < sbr_end; ++i) {
        if (active0[i]) continue;

        const auto& orig = ch0_original[i];
        if (orig.empty()) continue;

        double sum_sq  = 0.0;
        float  max_abs = 0.0f;
        for (float v : orig) {
            sum_sq += static_cast<double>(v) * v;
            float av = std::abs(v);
            if (av > max_abs) max_abs = av;
        }
        float rms   = std::sqrt(static_cast<float>(sum_sq / orig.size()) + eps);
        float crest = max_abs / (rms + eps);

        result.noise_flag[i] = (crest < 2.5f);
        int sb = get_scale_bits(i);
        result.rms_idx[i] = get_scale_idx(rms, sb);
    }
    return result;
}

inline void sbr_write(std::vector<uint8_t>&       out,
                      const SBREncodeResult&       sbr,
                      const std::vector<int>&      sbr_indices,
                      bool                         can_state_pred,
                      const ModeState&             pred_state)
{
    if (sbr_indices.empty()) return;

    std::vector<bool> noise_bits;
    noise_bits.reserve(sbr_indices.size());
    for (int idx : sbr_indices) {
        bool cur       = sbr.noise_flag[idx];
        bool write_bit = cur;
        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            write_bit = cur ^ (pred_state.prev_sbr_noise[idx] != 0);
        }
        noise_bits.push_back(write_bit);
    }
    auto noise_packed = pack_bits(noise_bits, 0, (int)noise_bits.size());
    out.insert(out.end(), noise_packed.begin(), noise_packed.end());

    uint8_t              sbr_rms_mode = 0;
    std::vector<uint32_t> sbr_vals;
    int                   k_sbr = 0;

    if (can_state_pred) {
        sbr_vals.reserve(sbr_indices.size());
        for (int idx : sbr_indices) {
            uint32_t cur = sbr.rms_idx[idx];
            uint32_t val = cur;
            if (idx < (int)pred_state.prev_sbr_valid.size() &&
                pred_state.prev_sbr_valid[idx])
            {
                int32_t d = (int32_t)cur - (int32_t)pred_state.prev_sbr_rms[idx];
                val = zigzag_encode(d);
            }
            sbr_vals.push_back(val);
        }
        k_sbr = compute_optimal_rice_k(sbr_vals, 7);

        size_t fixed_bits = 0;
        for (int idx : sbr_indices)
            fixed_bits += get_scale_bits(idx);

        size_t rice_bits = estimate_rice_bits(sbr_vals, k_sbr) + 24;
        if (rice_bits < fixed_bits)
            sbr_rms_mode = 1;
    }

    out.push_back(sbr_rms_mode);

    if (sbr_rms_mode == 1) {
        out.push_back((uint8_t)k_sbr);
        BitWriterMSB rms_writer;
        for (uint32_t v : sbr_vals)
            rice_encode(rms_writer, std::vector<uint32_t>{v}, k_sbr);
        rms_writer.flush();
        auto rms_bytes = rms_writer.data();
        uint16_t rms_len = (uint16_t)rms_bytes.size();
        out.push_back(rms_len & 0xFF);
        out.push_back((rms_len >> 8) & 0xFF);
        out.insert(out.end(), rms_bytes.begin(), rms_bytes.end());
    } else {
        BitWriterMSB rms_writer;
        for (int idx : sbr_indices) {
            int sb = get_scale_bits(idx);
            rms_writer.write_bits(sbr.rms_idx[idx], sb);
        }
        rms_writer.flush();
        auto rms_bytes = rms_writer.data();
        out.insert(out.end(), rms_bytes.begin(), rms_bytes.end());
    }
}