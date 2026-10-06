#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "codec_utils.h"
#include "pred_state.h"
#include "entropy_decoder.h"

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

struct SBRDecodeResult {
    std::vector<bool>     noise_flag;
    std::vector<float>    rms;
    std::vector<uint32_t> rms_idx;
    std::vector<uint8_t>  shape_mask;
    std::vector<int>      indices;
    bool                  used = false;
};

inline SBRDecodeResult sbr_decode_v23(BitReaderMSB& reader,
                                      const std::vector<uint8_t>& active0,
                                      int sbr_end, int expected_bands,
                                      bool can_state_pred,
                                      const ModeState& pred_state)
{
    SBRDecodeResult result;
    result.noise_flag.assign(expected_bands, false);
    result.rms.assign(expected_bands, 0.0f);
    result.rms_idx.assign(expected_bands, 0);
    result.shape_mask.assign(expected_bands, 0);
    
    if (sbr_end <= 0) return result;
    
    std::vector<int> indices;
    for (int i = 0; i < sbr_end; ++i) {
        if (!active0[i]) indices.push_back(i);
    }
    if (indices.empty()) return result;
    
    result.indices = indices;
    result.used = true;
    
    for (int idx : indices) {
        bool bit = (reader.read_bits(1) != 0);
        if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
            bit = bit ^ (pred_state.prev_sbr_noise[idx] != 0);
        }
        result.noise_flag[idx] = bit;
    }
    
    for (int idx : indices) {
        result.shape_mask[idx] = (uint8_t)reader.read_bits(4);
    }
    
    int k_sbr = (int)reader.read_bits(3);
    auto sbr_vals = rice_decode(reader, (int)indices.size(), k_sbr);
    
    if ((int)sbr_vals.size() != (int)indices.size()) {
        throw std::runtime_error("SBR RMS decode size mismatch");
    }
    
    for (size_t j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        uint32_t val = sbr_vals[j];
        int sb = get_scale_bits(idx);
        int max_idx = (1 << sb) - 1;
        uint32_t rms_idx;
        
        if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
            int32_t d = zigzag_decode(val);
            int64_t v = (int64_t)pred_state.prev_sbr_rms[idx] + d;
            if (v < 0) v = 0;
            if (v > max_idx) v = max_idx;
            rms_idx = (uint32_t)v;
        } else {
            rms_idx = val;
            if (rms_idx > (uint32_t)max_idx) rms_idx = max_idx;
        }
        
        result.rms_idx[idx] = rms_idx;
        float log_rms = SCALE_LOG_MIN + rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
        result.rms[idx] = exp10f(log_rms);
    }
    
    return result;
}

inline SBRDecodeResult sbr_decode(const uint8_t* blk, size_t& ptr, size_t blk_size,
                                  const std::vector<uint8_t>& active0,
                                  int sbr_end, int expected_bands,
                                  uint8_t block_format_version,
                                  bool can_state_pred,
                                  const ModeState& pred_state)
{
    SBRDecodeResult result;
    result.noise_flag.assign(expected_bands, false);
    result.rms.assign(expected_bands, 0.0f);
    result.rms_idx.assign(expected_bands, 0);
    result.shape_mask.assign(expected_bands, 0);
    if (sbr_end <= 0) return result;
    for (int i = 0; i < sbr_end; ++i) {
        if (!active0[i]) result.indices.push_back(i);
    }
    if (result.indices.empty()) return result;

    int noise_bytes = ((int)result.indices.size() + 7) / 8;
    if (ptr + noise_bytes > blk_size) return result;
    for (size_t j = 0; j < result.indices.size(); ++j) {
        int  idx = result.indices[j];
        bool bit = (blk[ptr + j / 8] >> (j % 8)) & 1;
        if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
            bit = bit ^ (pred_state.prev_sbr_noise[idx] != 0);
        }
        result.noise_flag[idx] = bit;
    }
    ptr += noise_bytes;

    int mask_count = result.indices.size();
    int mask_bytes = (mask_count + 1) / 2;
    if (ptr + mask_bytes > blk_size) return result;
    for (int j = 0; j < mask_count; ++j) {
        uint8_t byte = blk[ptr + j / 2];
        uint8_t m = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        result.shape_mask[result.indices[j]] = m;
    }
    ptr += mask_bytes;

    if (block_format_version >= 19) {
        if (ptr >= blk_size) return result;
        uint8_t sbr_rms_mode = blk[ptr++];
        if (sbr_rms_mode == 1) {
            if (ptr >= blk_size) return result;
            int k_sbr = blk[ptr++];
            if (ptr + 2 > blk_size) return result;
            uint16_t rms_len;
            memcpy(&rms_len, &blk[ptr], 2);
            ptr += 2;
            if (ptr + rms_len > blk_size) throw std::runtime_error("SBR RMS truncated");
            BitReaderMSB rms_reader(blk + ptr, rms_len);
            ptr += rms_len;
            result.used = true;
            for (int idx : result.indices) {
                auto val_vec = rice_decode(rms_reader, 1, k_sbr);
                if (val_vec.empty()) throw std::runtime_error("SBR RMS index missing");
                uint32_t val = val_vec[0];
                int sb = get_scale_bits(idx);
                int max_idx = (1 << sb) - 1;
                uint32_t rms_idx;
                if (can_state_pred && idx < (int)pred_state.prev_sbr_valid.size() && pred_state.prev_sbr_valid[idx]) {
                    int32_t d = zigzag_decode(val);
                    int64_t v = (int64_t)pred_state.prev_sbr_rms[idx] + d;
                    if (v < 0) v = 0;
                    if (v > max_idx) v = max_idx;
                    rms_idx = (uint32_t)v;
                } else {
                    rms_idx = val;
                    if (rms_idx > (uint32_t)max_idx) rms_idx = max_idx;
                }
                result.rms_idx[idx] = rms_idx;
                float log_rms = SCALE_LOG_MIN + rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
                result.rms[idx] = exp10f(log_rms);
            }
        } else {
            int total_rms_bits = 0;
            for (int idx : result.indices) total_rms_bits += get_scale_bits(idx);
            int rms_bytes = (total_rms_bits + 7) / 8;
            if (ptr + rms_bytes > blk_size) return result;
            BitReaderMSB rms_reader(blk + ptr, rms_bytes);
            result.used = true;
            for (int idx : result.indices) {
                int sb = get_scale_bits(idx);
                uint32_t rms_idx = rms_reader.read_bits(sb);
                int max_idx = (1 << sb) - 1;
                if (rms_idx > (uint32_t)max_idx) rms_idx = max_idx;
                result.rms_idx[idx] = rms_idx;
                float log_rms = SCALE_LOG_MIN + rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
                result.rms[idx] = exp10f(log_rms);
            }
            ptr += rms_bytes;
        }
    } else {
        int total_rms_bits = 0;
        for (int idx : result.indices) total_rms_bits += get_scale_bits(idx);
        int rms_bytes = (total_rms_bits + 7) / 8;
        if (ptr + rms_bytes > blk_size) return result;
        BitReaderMSB rms_reader(blk + ptr, rms_bytes);
        result.used = true;
        for (int idx : result.indices) {
            int sb = get_scale_bits(idx);
            uint32_t rms_idx = rms_reader.read_bits(sb);
            int max_idx = (1 << sb) - 1;
            if (rms_idx > (uint32_t)max_idx) rms_idx = max_idx;
            result.rms_idx[idx] = rms_idx;
            float log_rms = SCALE_LOG_MIN + rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
            result.rms[idx] = exp10f(log_rms);
        }
        ptr += rms_bytes;
    }
    return result;
}

inline void sbr_synthesize(
    std::vector<std::vector<float>>& ch0_bands,
    const std::vector<uint8_t>&      active0,
    const std::vector<int>&          band_shapes,
    const SBRDecodeResult&           sbr,
    int                              sbr_end,
    uint32_t                         block_idx,
    float w0, float w1)
{
    if (!sbr.used) return;
    const int band_count = (int)ch0_bands.size();
    std::vector<int> old_index(band_count);
    for (int i = 0; i < band_count; ++i) old_index[i] = i ^ (i >> 1);
    
    std::vector<int> available_sources;
    for (int j = 0; j < band_count; ++j) {
        if (active0[j]) available_sources.push_back(j);
    }
    
    std::vector<bool> source_used(band_count, false);
    for (int i = 0; i < sbr_end; ++i) {
        if (active0[i]) continue;
        const int N = band_shapes[i];
        const float target_rms = sbr.rms[i];
        const float target_energy = target_rms * target_rms * N;
        ch0_bands[i].resize(N);
        
        if (sbr.noise_flag[i]) {
            uint32_t seed = block_idx * 1000000u + (uint32_t)i * 137u + 1u;
            for (int j = 0; j < N; ++j) {
                seed = seed * 1664525u + 1013904223u;
                ch0_bands[i][j] = (static_cast<float>(seed) / 2147483648.0f) - 1.0f;
            }
        } else {
            int source_band = -1;
            float best_weight = -1.0f;
            const int IDEAL_DISTANCE = 2;
            const float SIGMA = 3.0f;
            const int target_old = old_index[i];
            for (int src : available_sources) {
                if (source_used[src]) continue;
                const int distance = std::abs((old_index[src]) - target_old);
                const float d = (float)(distance - IDEAL_DISTANCE);
                const float weight = std::exp(-(d * d) / (2.0f * SIGMA * SIGMA));
                if (weight > best_weight) {
                    best_weight = weight;
                    source_band = src;
                }
            }
            if (source_band >= 0) {
                source_used[source_band] = true;
                const auto& src = ch0_bands[source_band];
                const int src_size = (int)src.size();
                for (int j = 0; j < N; ++j) {
                    ch0_bands[i][j] = src[j % src_size];
                }
            } else {
                std::fill(ch0_bands[i].begin(), ch0_bands[i].end(), 0.0f);
            }
        }
        
        int seg_size = N / 4;
        float seg_src_energy[4] = {0, 0, 0, 0};
        for (int s = 0; s < 4; ++s) {
            int start = s * seg_size;
            int end = (s == 3) ? N : (s + 1) * seg_size;
            for (int j = start; j < end; ++j) {
                seg_src_energy[s] += ch0_bands[i][j] * ch0_bands[i][j];
            }
        }
        float weights[4];
        get_sbr_shape_weights(sbr.shape_mask[i], w0, w1, weights);
        for (int s = 0; s < 4; ++s) {
            int start = s * seg_size;
            int end = (s == 3) ? N : (s + 1) * seg_size;
            float seg_target_e = target_energy * weights[s];
            float gain = std::sqrt(seg_target_e / (seg_src_energy[s] + 1e-12f));
            if (gain > 2) {
                gain = 0;
            }
            for (int j = start; j < end; ++j) {
                ch0_bands[i][j] *= gain;
            }
        }
    }
}
