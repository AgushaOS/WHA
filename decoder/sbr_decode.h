#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "codec_utils.h"
#include "pred_state.h"
#include "entropy_decoder.h"

struct SBRDecodeResult {
    std::vector<bool>     noise_flag;
    std::vector<float>    rms;
    std::vector<uint32_t> rms_idx;
    std::vector<int>      indices;
    bool                  used = false;
};

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

    if (sbr_end <= 0) return result;

    for (int i = 0; i < sbr_end; ++i) {
        if (!active0[i])
            result.indices.push_back(i);
    }
    if (result.indices.empty()) return result;

    int noise_bytes = ((int)result.indices.size() + 7) / 8;
    if (ptr + noise_bytes > blk_size) return result;

    for (size_t j = 0; j < result.indices.size(); ++j) {
        int  idx = result.indices[j];
        bool bit = (blk[ptr + j / 8] >> (j % 8)) & 1;
        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
        {
            bit = bit ^ (pred_state.prev_sbr_noise[idx] != 0);
        }
        result.noise_flag[idx] = bit;
    }
    ptr += noise_bytes;

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
            if (ptr + rms_len > blk_size)
                throw std::runtime_error("SBR RMS truncated");

            BitReaderMSB rms_reader(blk + ptr, rms_len);
            ptr += rms_len;
            result.used = true;

            for (int idx : result.indices) {
                auto val_vec = rice_decode(rms_reader, 1, k_sbr);
                if (val_vec.empty())
                    throw std::runtime_error("SBR RMS index missing");
                uint32_t val = val_vec[0];

                int sb      = get_scale_bits(idx);
                int max_idx = (1 << sb) - 1;
                uint32_t rms_idx;

                if (can_state_pred &&
                    idx < (int)pred_state.prev_sbr_valid.size() &&
                    pred_state.prev_sbr_valid[idx])
                {
                    int32_t d = zigzag_decode(val);
                    int64_t v = (int64_t)pred_state.prev_sbr_rms[idx] + d;
                    if (v < 0) v = 0;
                    if (v > max_idx) v = max_idx;
                    rms_idx = (uint32_t)v;
                } else {
                    rms_idx = val;
                    if (rms_idx > (uint32_t)max_idx)
                        rms_idx = max_idx;
                }

                result.rms_idx[idx] = rms_idx;
                float log_rms = SCALE_LOG_MIN +
                    rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
                result.rms[idx] = exp10f(log_rms);
            }
        } else {
            int total_rms_bits = 0;
            for (int idx : result.indices)
                total_rms_bits += get_scale_bits(idx);
            int rms_bytes = (total_rms_bits + 7) / 8;
            if (ptr + rms_bytes > blk_size) return result;

            BitReaderMSB rms_reader(blk + ptr, rms_bytes);
            result.used = true;

            for (int idx : result.indices) {
                int      sb      = get_scale_bits(idx);
                uint32_t rms_idx = rms_reader.read_bits(sb);
                int      max_idx = (1 << sb) - 1;
                if (rms_idx > (uint32_t)max_idx)
                    rms_idx = max_idx;
                result.rms_idx[idx] = rms_idx;
                float log_rms = SCALE_LOG_MIN +
                    rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
                result.rms[idx] = exp10f(log_rms);
            }
            ptr += rms_bytes;
        }
    } else {
        int total_rms_bits = 0;
        for (int idx : result.indices)
            total_rms_bits += get_scale_bits(idx);
        int rms_bytes = (total_rms_bits + 7) / 8;
        if (ptr + rms_bytes > blk_size) return result;

        BitReaderMSB rms_reader(blk + ptr, rms_bytes);
        result.used = true;

        for (int idx : result.indices) {
            int      sb      = get_scale_bits(idx);
            uint32_t rms_idx = rms_reader.read_bits(sb);
            int      max_idx = (1 << sb) - 1;
            if (rms_idx > (uint32_t)max_idx)
                rms_idx = max_idx;
            result.rms_idx[idx] = rms_idx;
            float log_rms = SCALE_LOG_MIN +
                rms_idx * (SCALE_LOG_MAX - SCALE_LOG_MIN) / max_idx;
            result.rms[idx] = exp10f(log_rms);
        }
        ptr += rms_bytes;
    }

    return result;
}

inline void sbr_synthesize(std::vector<std::vector<float>>& ch0_bands,
                           const std::vector<uint8_t>&      active0,
                           const std::vector<int>&          band_shapes,
                           const SBRDecodeResult&           sbr,
                           int                              sbr_end,
                           uint32_t                         block_idx)
{
    if (!sbr.used) return;

    std::vector<int> available_sources;
    for (int j = 0; j < (int)ch0_bands.size(); ++j) {
        if (active0[j])
            available_sources.push_back(j);
    }

    std::vector<bool> source_used(ch0_bands.size(), false);

    for (int i = 0; i < sbr_end; ++i) {
        if (active0[i]) continue;

        int   N             = band_shapes[i];
        float target_rms    = sbr.rms[i];
        float target_energy = target_rms * target_rms * N;

        ch0_bands[i].resize(N);

        if (sbr.noise_flag[i]) {
            uint32_t seed = block_idx * 1000000u + (uint32_t)i * 137u + 1u;
            float noise_energy = 0.0f;
            for (int j = 0; j < N; ++j) {
                seed = seed * 1664525u + 1013904223u;
                float noise = (static_cast<float>(seed) / 2147483648.0f) - 1.0f;
                ch0_bands[i][j] = noise;
                noise_energy += noise * noise;
            }
            float gain = std::sqrt(target_energy / (noise_energy + 1e-12f));
            for (int j = 0; j < N; ++j)
                ch0_bands[i][j] *= gain;
        } else {
            int   source_band = -1;
            float best_weight = -1.0f;
            const int   IDEAL_DISTANCE = 2;
            const float SIGMA          = 3.0f;

            for (int src : available_sources) {
                if (source_used[src]) continue;
                int   distance = std::abs(src - i);
                float d        = (float)(distance - IDEAL_DISTANCE);
                float weight   = std::exp(-(d * d) / (2.0f * SIGMA * SIGMA));
                if (weight > best_weight) {
                    best_weight = weight;
                    source_band = src;
                }
            }

            if (source_band >= 0) {
                source_used[source_band] = true;
                const auto& src      = ch0_bands[source_band];
                int         src_size = (int)src.size();
                float       src_energy = 0.0f;
                for (float v : src) src_energy += v * v;
                float gain = std::sqrt(target_energy / (src_energy + 1e-12f));
                for (int j = 0; j < N; ++j)
                    ch0_bands[i][j] = src[j % src_size] * gain;
            } else {
                std::fill(ch0_bands[i].begin(), ch0_bands[i].end(), 0.0f);
            }
        }
    }
}