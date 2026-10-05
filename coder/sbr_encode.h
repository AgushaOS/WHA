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
    float v[4];
    for (int i = 0; i < 4; ++i) {
        v[i] = (mask & (1 << i)) ? w1 : w0;
        sum += v[i];
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = v[i] / sum;
    }
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
    int sbr_end, int band_count,
    float w0, float w1)
{
    SBREncodeResult result;
    result.noise_flag.assign(band_count, false);
    result.rms_idx.assign(band_count, 0);
    result.shape_mask.assign(band_count, 0);
    result.sbr_end = sbr_end;

    const float eps = 1e-12f;
    for (int i = 0; i < sbr_end; ++i) {
        if (active0[i]) continue;
        const auto& orig = ch0_original[i];
        if (orig.empty()) continue;
        
        int N = orig.size();
        int seg_size = N / 4;
        
        float seg_energy[4] = {0, 0, 0, 0};
        for (int s = 0; s < 4; ++s) {
            int start = s * seg_size;
            int end = (s == 3) ? N : (s + 1) * seg_size;
            for (int j = start; j < end; ++j) {
                seg_energy[s] += orig[j] * orig[j];
            }
        }
        float total_e = seg_energy[0] + seg_energy[1] + seg_energy[2] + seg_energy[3] + eps;
        float target_p[4];
        for(int s = 0; s < 4; ++s) target_p[s] = seg_energy[s] / total_e;
        
        float best_err = 1e9f;
        uint8_t best_mask = 0;
        for (uint8_t m = 0; m < 16; ++m) {
            float w[4];
            get_sbr_shape_weights(m, w0, w1, w);
            float err = 0.0f;
            for(int s = 0; s < 4; ++s) {
                float diff = target_p[s] - w[s];
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_mask = m;
            }
        }
        result.shape_mask[i] = best_mask;
        
        double sum_sq  = 0.0;
        float  max_abs = 0.0f;
        for (float v : orig) {
            sum_sq += static_cast<double>(v) * v;
            float av = std::abs(v);
            if (av > max_abs) max_abs = av;
        }
        float rms   = std::sqrt(static_cast<float>(sum_sq / N) + eps);
        float crest = max_abs / (rms + eps);
        
        result.noise_flag[i] = (crest < 3.5f);
        int sb = get_scale_bits(i);
        result.rms_idx[i] = get_scale_idx(rms, sb);
    }
    return result;
}

inline void sbr_write_to_stream(BitWriterMSB& writer,
                                const SBREncodeResult& sbr,
                                const std::vector<int>& sbr_indices,
                                bool can_state_pred,
                                const ModeState& pred_state)
{
    if (sbr_indices.empty()) return;

    for (int idx : sbr_indices) {
        bool cur = sbr.noise_flag[idx];
        bool write_bit = cur;
        if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
            write_bit = cur ^ (pred_state.prev_sbr_noise[idx] != 0);
        }
        writer.write_bits(write_bit ? 1 : 0, 1);
    }

    for (int idx : sbr_indices) {
        writer.write_bits(sbr.shape_mask[idx], 4);
    }

    std::vector<uint32_t> sbr_vals;
    sbr_vals.reserve(sbr_indices.size());
    for (int idx : sbr_indices) {
        uint32_t cur = sbr.rms_idx[idx];
        uint32_t val = cur;
        if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
            int32_t d = (int32_t)cur - (int32_t)pred_state.prev_sbr_rms[idx];
            val = zigzag_encode(d);
        }
        sbr_vals.push_back(val);
    }
    
    int k_sbr = compute_optimal_rice_k(sbr_vals, 7);
    writer.write_bits(k_sbr & 0x07, 3); 
    rice_encode(writer, sbr_vals, k_sbr);
}