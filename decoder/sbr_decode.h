#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <utility>
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
    for (int i = 0; i < 4; ++i) out[i] = v[i] / sum;
}

struct SBRDecodeResult {
    std::vector<bool>     noise_flag;
    std::vector<float>    rms;
    std::vector<uint32_t> rms_idx;
    std::vector<uint8_t>  shape_mask;
    std::vector<int>      indices;
    bool                  used = false;

    void reset(int bands) {
        if (bands < 0) bands = 0;
        const size_t nb = (size_t)bands;

        noise_flag.resize(nb);
        for (size_t i = 0; i < nb; ++i) noise_flag[i] = false;

        rms.resize(nb);
        std::fill_n(rms.begin(), nb, 0.0f);

        rms_idx.resize(nb);
        std::fill_n(rms_idx.begin(), nb, (uint32_t)0);

        shape_mask.resize(nb);
        std::fill_n(shape_mask.begin(), nb, (uint8_t)0);

        indices.clear();
        if (indices.capacity() < nb) indices.reserve(nb);

        used = false;
    }
};

inline uint8_t sbr_read_huffman(BitReaderMSB& reader)
{
    uint32_t code = 0;

    for (int len = 1; len <= 8; ++len) {
        code = (code << 1) | reader.read_bits(1);

        switch (len) {
            case 1:
                if (code == 0b1) return 0;
                break;

            case 3:
                if (code == 0b001) return 7;
                break;

            case 4:
                if (code == 0b0001) return 8;
                if (code == 0b0110) return 14;
                break;

            case 5:
                if (code == 0b01011) return 1;
                if (code == 0b00000) return 2;
                if (code == 0b01001) return 3;
                if (code == 0b00001) return 4;
                if (code == 0b01010) return 11;
                if (code == 0b01111) return 12;
                if (code == 0b01000) return 13;
                break;

            case 6:
                if (code == 0b011101) return 6;
                break;

            case 7:
                if (code == 0b0111000) return 10;
                break;

            case 8:
                if (code == 0b01110010) return 5;
                if (code == 0b01110011) return 9;
                break;
        }
    }

    throw std::runtime_error("Invalid SBR Huffman code");
}


inline void sbr_decode_v23(SBRDecodeResult& result,
                           BitReaderMSB& reader,
                           const std::vector<uint8_t>& active0,
                           int sbr_end, int expected_bands,
                           bool can_state_pred,
                           const ModeState& pred_state)
{
    result.reset(expected_bands);
    if (sbr_end <= 0) return;

    for (int i = 0; i < sbr_end; ++i)
        if (!active0[i]) result.indices.push_back(i);
    if (result.indices.empty()) return;

    result.used = true;

    for (int idx : result.indices) {
        bool bit = (reader.read_bits(1) != 0);
        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx])
            bit ^= (pred_state.prev_sbr_noise[idx] != 0);
        result.noise_flag[idx] = bit;
    }

    for (int idx : result.indices)
        result.shape_mask[idx] = sbr_read_huffman(reader);

    int k_sbr = (int)reader.read_bits(3);
    auto sbr_vals = rice_decode(reader, (int)result.indices.size(), k_sbr);
    if ((int)sbr_vals.size() != (int)result.indices.size())
        throw std::runtime_error("SBR RMS decode size mismatch");

    for (size_t j = 0; j < result.indices.size(); ++j) {
        int idx = result.indices[j];
        uint32_t val = sbr_vals[j];
        int sb = get_scale_bits(idx);
        int max_idx = (1 << sb) - 1;
        uint32_t rms_idx;

        if (can_state_pred &&
            idx < (int)pred_state.prev_sbr_valid.size() &&
            pred_state.prev_sbr_valid[idx]) {
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
}

inline uint64_t sbr_hash64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline float sbr_random_sign(
    uint32_t block_idx,
    uint32_t band_idx,
    uint32_t sample_idx)
{
    uint64_t key =
        (uint64_t(block_idx) << 32) ^
        (uint64_t(band_idx) * 0xD6E8FEB86659FD93ULL) ^
        (uint64_t(sample_idx) * 0xA0761D6478BD642FULL);

    return (sbr_hash64(key) & 1ULL) ? 1.0f : -1.0f;
}

inline void sbr_synthesize(
    std::vector<std::vector<float>>& ch0_bands,
    const std::vector<uint8_t>&      active0,
    const std::vector<int>&          band_shapes,
    const SBRDecodeResult&           sbr,
    int                              sbr_end,
    uint32_t                         block_idx,
    float                            w0,
    float                            w1,
    const std::vector<float>&        window,
    int                              block_size,
    int                              left_overlap,
    int                              right_overlap)
{
    if (!sbr.used) return;

    const int band_count = (int)ch0_bands.size();

    sbr_end = std::min({
        sbr_end,
        band_count,
        (int)active0.size(),
        (int)band_shapes.size(),
        (int)sbr.noise_flag.size(),
        (int)sbr.rms.size(),
        (int)sbr.shape_mask.size()
    });

    if (sbr_end <= 0) return;

    bool has_noise = false;

    for (int i = 0; i < sbr_end; ++i) {
        if (!active0[i] && sbr.noise_flag[i]) {
            has_noise = true;
            break;
        }
    }

    struct WMeanEntry {
        int N;
        std::vector<float> data;
    };

    struct WMeanCache {
        int bs = -1;
        int lo = -1;
        int ro = -1;
        std::vector<WMeanEntry> entries;
    };

    static thread_local WMeanCache wcache;

    if (has_noise &&
        (wcache.bs != block_size ||
         wcache.lo != left_overlap ||
         wcache.ro != right_overlap))
    {
        wcache.bs = block_size;
        wcache.lo = left_overlap;
        wcache.ro = right_overlap;

        wcache.entries.clear();
        wcache.entries.reserve(8);

        for (int i = 0; i < sbr_end; ++i) {
            const int N = band_shapes[i];

            if (N <= 0) continue;

            bool exists = false;

            for (auto& e : wcache.entries) {
                if (e.N == N) {
                    exists = true;
                    break;
                }
            }

            if (exists) continue;

            const int span = std::max(1, block_size / N);

            std::vector<float> v(N);

            for (int j = 0; j < N; ++j) {
                const int ts = j * span;
                const int te = std::min(ts + span, block_size);

                float sum = 0.0f;

                for (int t = ts; t < te; ++t)
                    sum += window[t];

                v[j] = (te > ts)
                    ? sum / float(te - ts)
                    : 1.0f;
            }

            wcache.entries.push_back(
                WMeanEntry{N, std::move(v)}
            );
        }
    }

    static thread_local std::vector<int> available_sources;
    static thread_local std::vector<uint8_t> source_used;
    static thread_local std::vector<int> old_index_cache;

    if ((int)old_index_cache.size() != band_count) {
        old_index_cache.resize(band_count);

        for (int i = 0; i < band_count; ++i)
            old_index_cache[i] = i ^ (i >> 1);
    }

    const int* old_index = old_index_cache.data();

    available_sources.clear();
    available_sources.reserve(band_count);

    for (int j = 0; j < band_count; ++j) {
        if (j < (int)active0.size() && active0[j])
            available_sources.push_back(j);
    }

    source_used.assign(band_count, 0);

    for (int i = 0; i < sbr_end; ++i) {

        if (active0[i])
            continue;

        const int N = band_shapes[i];

        if (N <= 0)
            continue;

        const float target_rms = sbr.rms[i];

        ch0_bands[i].resize(N);

        float* dst = ch0_bands[i].data();

        if (sbr.noise_flag[i]) {

            const float amplitude = target_rms;

            const float* wm = nullptr;

            if (has_noise) {
                for (auto& e : wcache.entries) {
                    if (e.N == N) {
                        wm = e.data.data();
                        break;
                    }
                }
            }

            for (int j = 0; j < N; ++j) {

                const float sign = sbr_random_sign(
                    block_idx,
                    (uint32_t)i,
                    (uint32_t)j
                );

                dst[j] = amplitude * sign;

                if (wm)
                    dst[j] *= wm[j];
            }
        }
        else {
            int source_band = -1;

            float best_weight = -1.0f;

            const int IDEAL_DISTANCE = 2;
            const float SIGMA = 3.0f;

            const int target_old = old_index[i];

            for (int src : available_sources) {

                if (source_used[src])
                    continue;

                const int distance =
                    std::abs(old_index[src] - target_old);

                const float d =
                    (float)(distance - IDEAL_DISTANCE);

                const float weight =
                    expf(
                        -(d * d) /
                        (2.0f * SIGMA * SIGMA)
                    );

                if (weight > best_weight) {
                    best_weight = weight;
                    source_band = src;
                }
            }

            if (source_band >= 0) {

                source_used[source_band] = 1;

                const float* src =
                    ch0_bands[source_band].data();

                const int src_size =
                    (int)ch0_bands[source_band].size();

                if (src_size > 0) {

                    int sj = 0;

                    for (int j = 0; j < N; ++j) {

                        dst[j] = src[sj];

                        if (++sj >= src_size)
                            sj = 0;
                    }
                }
                else {
                    std::fill(dst, dst + N, 0.0f);
                }
            }
            else {
                std::fill(dst, dst + N, 0.0f);
            }
        }

        const float target_energy =
            target_rms * target_rms * N;

        const int seg_size = N / 4;

        float seg_src_energy[4] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };

        for (int s = 0; s < 4; ++s) {

            const int start = s * seg_size;

            const int end =
                (s == 3) ? N : (s + 1) * seg_size;

            float energy = 0.0f;

            for (int j = start; j < end; ++j)
                energy += dst[j] * dst[j];

            seg_src_energy[s] = energy;
        }

        float weights[4];

        get_sbr_shape_weights(
            sbr.shape_mask[i],
            w0,
            w1,
            weights
        );

        for (int s = 0; s < 4; ++s) {

            const int start = s * seg_size;

            const int end =
                (s == 3) ? N : (s + 1) * seg_size;

            const float seg_target_e =
                target_energy * weights[s];

            float gain = sqrtf(
                seg_target_e /
                (seg_src_energy[s] + 1e-12f)
            );

            if (sbr.shape_mask[i] == 0)
                gain *= 1.1f;

            for (int j = start; j < end; ++j)
                dst[j] *= gain;
        }
    }
}
