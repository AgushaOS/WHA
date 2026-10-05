#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <unordered_map>
#include <cstring>
#include <array>
#include <tuple>
#include <memory> 
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "wavelet.h"
#include "perc_prior.h"
#include "bit_alloc.h"
#include "joint_stereo.h"
#include "quantize.h"
#include "entropy_encoder.h"
#include "settings.h"
#include "pred_state.h"
#include "codec_utils.h"
#include "transient.h"
#include "energy_shape.h"
#include "sbr_encode.h"
#include "pow_filter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float SBR_W0 = 1.0f;
float SBR_W1 = 9.0f;

static int get_adaptive_is_max_base(float target_kbps, const EncoderSettings& s)
{
    int base_default = get_is_base_original(target_kbps);
    if (s.adaptive_is_base_override > 0)
        return std::clamp(s.adaptive_is_base_override, base_default, 16);
    if (s.adaptive_is_base_override == 0)
        return base_default;
    if (target_kbps < s.adaptive_is_kbps_min || target_kbps > s.adaptive_is_kbps_max)
        return base_default;
    
    float k0, k1;
    int b0, b1;
    if (target_kbps <= 64.0f) {
        k0 = k1 = target_kbps;
        b0 = b1 = s.adaptive_is_base_at_64;
    } else if (target_kbps <= 96.0f) {
        k0 = 64.0f; k1 = 96.0f;
        b0 = s.adaptive_is_base_at_64; b1 = s.adaptive_is_base_at_96;
    } else if (target_kbps <= 128.0f) {
        k0 = 96.0f; k1 = 128.0f;
        b0 = s.adaptive_is_base_at_96; b1 = s.adaptive_is_base_at_128;
    } else {
        k0 = 128.0f; k1 = 160.0f;
        b0 = s.adaptive_is_base_at_128; b1 = s.adaptive_is_base_at_160;
    }
    
    float t = 0.0f;
    if (std::abs(k1 - k0) > 1e-6f)
        t = (target_kbps - k0) / (k1 - k0);
    t = std::clamp(t, 0.0f, 1.0f);
    int b = (int)std::lround((float)b0 + t * (float)(b1 - b0));
    b = std::max(b, base_default);
    return std::clamp(b, 0, 16);
}

std::vector<uint8_t> compress_block_adaptive_joint(
    const std::vector<float>& block,
    PWPT& wpt,
    int num_channels,
    int level,
    int target_bits_budget,
    float target_kbps,
    int sr,
    int block_size,
    bool enable_ms,
    bool block_is_full,
    uint32_t block_index,
    PredContext& pred_ctx)
{
    const float eps = 1e-12f;
    const bool stereo = (num_channels == 2);
    std::vector<std::vector<float>> left_coeffs, right_coeffs;
    std::vector<int> coeff_counts;

    if (stereo) {
        size_t spc = block.size() / 2;
        std::vector<float> left(spc), right(spc);
        for (size_t i = 0; i < spc; ++i) {
            left[i] = block[i * 2];
            right[i] = block[i * 2 + 1];
        }
        left_coeffs = wpt.wpt(left, level, target_kbps, sr, num_channels);
        right_coeffs = wpt.wpt(right, level, target_kbps, sr, num_channels);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i)
            coeff_counts[i] = (int)left_coeffs[i].size();
    } else {
        left_coeffs = wpt.wpt(block, level, target_kbps, sr, num_channels);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i)
            coeff_counts[i] = (int)left_coeffs[i].size();
    }

    int band_count = (int)coeff_counts.size();
    int total_bands = 1 << level;
    ModeState& pred_state = (level == 5) ? pred_ctx.long_state : pred_ctx.short_state;
    pred_state.ensure(band_count);

    bool can_state_pred = false;
    if (pred_state.ready && block_index > pred_state.last_bi) {
        uint32_t age = block_index - pred_state.last_bi;
        if (age <= SAME_MODE_PRED_MAX_AGE)
            can_state_pred = true;
    }

    bool use_is = (stereo && target_kbps < 510.0f);
    std::vector<uint8_t> mode_ms(band_count, 0);
    std::vector<uint8_t> is_band(band_count, 0);
    std::vector<std::vector<float>> ch0_bands(band_count);
    std::vector<std::vector<float>> ch1_bands(band_count);
    std::vector<std::vector<float>> ch0_original(band_count);
    for (int i = 0; i < band_count; ++i)
        ch0_original[i] = left_coeffs[i];

    int is_start_default = band_count;
    int is_start_max = band_count;
    std::vector<std::array<float, 4>> is_r_vals(band_count);
    for (auto& a : is_r_vals) a.fill(0.5f);
    std::vector<std::array<bool, 4>> is_inv_flags(band_count);
    for (auto& a : is_inv_flags) a.fill(false);
    std::vector<bool> is_use_segmented(band_count, false);

    if (use_is) {
        is_start_default = get_is_start_band(target_kbps, total_bands);
        if (is_start_default >= band_count)
            use_is = false;
    }

    bool adaptive_is = false;
    if (use_is) {
        bool in_range = (target_kbps >= SETTINGS.adaptive_is_kbps_min &&
                         target_kbps <= SETTINGS.adaptive_is_kbps_max);
        bool allow_adaptive = SETTINGS.enable_adaptive_is &&
                              (SETTINGS.adaptive_is_base_override != 0) &&
                              (in_range || SETTINGS.adaptive_is_base_override > 0);
        if (allow_adaptive) {
            int base_max = get_adaptive_is_max_base(target_kbps, SETTINGS);
            is_start_max = get_is_start_from_base(base_max, total_bands);
            if (is_start_max < is_start_default) is_start_max = is_start_default;
            if (is_start_max > band_count) is_start_max = band_count;
            adaptive_is = true;
        }
    }

    if (stereo) {
        int low_end = use_is ? is_start_default : band_count;
        for (int i = 0; i < low_end; ++i) {
            const auto& lb = left_coeffs[i];
            const auto& rb = right_coeffs[i];
            int n = (int)lb.size();
            if (n == 0) {
                ch0_bands[i] = lb;
                ch1_bands[i] = rb;
                continue;
            }
            float El = 0, Er = 0, Em = 0, Es = 0;
            for (int j = 0; j < n; ++j) {
                El += lb[j] * lb[j];
                Er += rb[j] * rb[j];
            }
            auto [mid, side] = mid_side(lb, rb);
            for (int j = 0; j < n; ++j) {
                Em += mid[j] * mid[j];
                Es += side[j] * side[j];
            }
            if (use_mid_side(El, Er, Em, Es, enable_ms, target_kbps)) {
                mode_ms[i] = 1;
                ch0_bands[i] = std::move(mid);
                ch1_bands[i] = std::move(side);
            } else {
                mode_ms[i] = 0;
                ch0_bands[i] = lb;
                ch1_bands[i] = rb;
            }
        }

        if (use_is) {
            BandEnergy be = compute_band_energy(ch0_bands, ch1_bands, is_band, stereo, band_count);
            int is_seg_threshold = 8 * total_bands / 16;
            for (int i = is_start_default; i < band_count; ++i) {
                bool force_is = !adaptive_is || (i >= is_start_max);
                bool choose_is = force_is;
                if (!force_is) {
                    choose_is = should_use_is_band(
                        left_coeffs[i], right_coeffs[i], enable_ms, target_kbps,
                        SETTINGS.is_strong_ratio_threshold,
                        SETTINGS.is_equal_ratio_threshold,
                        SETTINGS.is_equal_ratio_threshold_low_bitrate);
                }
                if (choose_is) {
                    is_band[i] = 1;
                    mode_ms[i] = 1;
                    std::array<float, 4> r_arr;
                    std::array<bool, 4> inv_arr;
                    bool use_segmented;
                    std::vector<float> Y;
                    compute_is_parameters_ex(
                        left_coeffs[i], right_coeffs[i], Y, r_arr,
                        inv_arr, use_segmented, i, total_bands);
                    is_r_vals[i] = r_arr;
                    is_inv_flags[i] = inv_arr;
                    is_use_segmented[i] = (i < is_seg_threshold);
                    ch0_bands[i] = std::move(Y);
                    ch1_bands[i] = ch0_bands[i];
                } else {
                    is_band[i] = 0;
                    mode_ms[i] = 0;
                    auto [mid, side] = mid_side(left_coeffs[i], right_coeffs[i]);
                    ch0_bands[i] = mid;
                    ch1_bands[i] = side;
                }
            }
        }
    } else {
        for (int i = 0; i < band_count; ++i) {
            ch0_bands[i] = left_coeffs[i];
            ch1_bands[i].clear();
        }
    }

    BandEnergy be = compute_band_energy(ch0_bands, ch1_bands, is_band, stereo, band_count);
    std::vector<float> energy0 = std::move(be.e0);
    std::vector<float> energy1 = std::move(be.e1);

    apply_base_pow(energy0, energy1, band_count);
    bool has_transient = detect_strong_transient(
        energy0, energy1, band_count, stereo, SETTINGS.transient_ratio_threshold);
    apply_adaptive_pow(energy0, energy1, band_count, stereo, target_kbps, has_transient);
    apply_narrowband_correction(energy0, energy1, band_count, stereo);
    apply_highfreq_attenuation(
        energy0, energy1, band_count, target_kbps, num_channels, has_transient);

    static std::vector<std::vector<float>> prev_ch0, prev_ch1;
    std::vector<float> priority0, priority1;
    if (stereo) {
        priority0 = compute_channel_priority(
            ch0_bands, prev_ch0, coeff_counts, target_kbps, sr, level, false);
        priority1 = compute_channel_priority(
            ch1_bands, prev_ch1, coeff_counts, target_kbps, sr, level, true);
    } else {
        priority0 = compute_channel_priority(
            ch0_bands, prev_ch0, coeff_counts, target_kbps, sr, level, false);
        priority1.assign(band_count, 0.0f);
    }

    std::vector<int> min_bits(band_count, 0);
    std::vector<int> max_bits(band_count, 10);
    if (use_is) {
        for (int i = 0; i < band_count; ++i) {
            if (is_band[i]) {
                min_bits[i] = 0;
                priority1[i] = 0.0f;
                energy1[i] = 0.0f;
            }
        }
    }

    int header_bytes_est = 3;
    int payload_budget = target_bits_budget - header_bytes_est * 8;
    if (payload_budget < 0) payload_budget = target_bits_budget / 2;
    int reservoir_max = (int)(payload_budget * SETTINGS.reservoir_max_factor);

    DualAllocResult alloc = allocate_bits_dual(
        priority0, priority1, coeff_counts, payload_budget,
        min_bits, max_bits, energy0, energy1, 0, reservoir_max,
        target_kbps, num_channels == 2);

    std::vector<int> bits0 = std::move(alloc.bits0);
    std::vector<int> bits1 = std::move(alloc.bits1);

    std::vector<bool> active0(band_count), active1(band_count);
    for (int i = 0; i < band_count; ++i) {
        if (stereo && is_band[i]) bits1[i] = 0;
        active0[i] = (bits0[i] > 0);
        if (stereo) active1[i] = (!is_band[i] && bits1[i] > 0);
        else active1[i] = false;
    }

    int sbr_end = block_is_full ? (3 * band_count / 4) : 0;
    SBREncodeResult sbr = sbr_analyze(
        ch0_original, active0, sbr_end, band_count, SBR_W0, SBR_W1);

    std::vector<uint8_t> header;
    std::vector<std::vector<std::vector<float>>> for_quant0(1);
    for_quant0[0] = std::move(ch0_bands);
    QuantResult qres0 = quantize_levels(for_quant0, bits0, target_kbps);

    QuantResult qres1;
    if (stereo) {
        std::vector<std::vector<std::vector<float>>> for_quant1(1);
        for_quant1[0] = std::move(ch1_bands);
        qres1 = quantize_levels(for_quant1, bits1, target_kbps);
    }

    std::vector<uint32_t> scale_raw0, scale_raw1, scale_pred0, scale_pred1;
    std::vector<uint32_t> full_scale0(band_count, 0), full_scale1(band_count, 0);
    for (int i = 0; i < band_count; ++i) {
        if (active0[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres0.scales[i], bits0[i]);
            uint32_t idx = get_scale_idx(step, sb);
            full_scale0[i] = idx;
            scale_raw0.push_back(idx);
            if (can_state_pred) {
                uint32_t val = idx;
                if (i < (int)pred_state.prev_active0.size() && pred_state.prev_active0[i]) {
                    int32_t d = (int32_t)idx - (int32_t)pred_state.prev_scale0[i];
                    val = zigzag_encode(d);
                }
                scale_pred0.push_back(val);
            }
        }
        if (stereo && active1[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres1.scales[i], bits1[i]);
            uint32_t idx = get_scale_idx(step, sb);
            full_scale1[i] = idx;
            scale_raw1.push_back(idx);
            if (can_state_pred) {
                uint32_t val = idx;
                if (i < (int)pred_state.prev_active1.size() && pred_state.prev_active1[i]) {
                    int32_t d = (int32_t)idx - (int32_t)pred_state.prev_scale1[i];
                    val = zigzag_encode(d);
                }
                scale_pred1.push_back(val);
            }
        }
    }

    int k_scale0_raw = compute_optimal_rice_k(scale_raw0);
    int k_scale1_raw = stereo ? compute_optimal_rice_k(scale_raw1) : 0;
    size_t raw_scale_bits = estimate_rice_bits(scale_raw0, k_scale0_raw) +
                            (stereo ? estimate_rice_bits(scale_raw1, k_scale1_raw) : 0);

    int k_scale0_pred = 0;
    int k_scale1_pred = 0;
    size_t pred_scale_bits = SIZE_MAX;
    if (can_state_pred) {
        k_scale0_pred = compute_optimal_rice_k(scale_pred0);
        k_scale1_pred = stereo ? compute_optimal_rice_k(scale_pred1) : 0;
        pred_scale_bits = estimate_rice_bits(scale_pred0, k_scale0_pred) +
                          (stereo ? estimate_rice_bits(scale_pred1, k_scale1_pred) : 0);
    }

    bool use_scale_pred = can_state_pred && (pred_scale_bits < raw_scale_bits);
    int k_scale0 = use_scale_pred ? k_scale0_pred : k_scale0_raw;
    int k_scale1 = use_scale_pred ? k_scale1_pred : k_scale1_raw;

    auto scale_vals0 = use_scale_pred ? std::move(scale_pred0) : std::move(scale_raw0);
    auto scale_vals1 = use_scale_pred ? std::move(scale_pred1) : std::move(scale_raw1);

    uint8_t k_scale_byte = (uint8_t)(k_scale0 & 0x07);
    if (stereo) k_scale_byte |= ((k_scale1 & 0x07) << 3);
    if (block_is_full) k_scale_byte |= 0x80;
    if (use_scale_pred) k_scale_byte |= 0x40;
    header.push_back(k_scale_byte);

    BitWriterMSB payload_writer;
    uint32_t packed_mode = 0;
    uint32_t packed_active0 = 0;
    for (int i = 0; i < band_count; ++i) {
        packed_mode = (packed_mode << 1) | mode_ms[i];
        packed_active0 = (packed_active0 << 1) | (active0[i] ? 1 : 0);
    }
    payload_writer.write_bits(packed_mode, band_count);
    payload_writer.write_bits(packed_active0, band_count);

    if (stereo) {
        int is_start = use_is ? get_is_start_band(target_kbps, total_bands) : band_count;
        uint32_t packed_active1 = 0;
        int non_is_count = 0;
        for (int i = 0; i < band_count; ++i) {
            bool is_is_band = use_is && (i >= is_start) && mode_ms[i];
            if (!is_is_band) {
                packed_active1 = (packed_active1 << 1) | (active1[i] ? 1 : 0);
                non_is_count++;
            }
        }
        if (non_is_count > 0) {
            payload_writer.write_bits(packed_active1, non_is_count);
        }
    }

    for (uint32_t v : scale_vals0)
        rice_encode(payload_writer, std::vector<uint32_t>{v}, k_scale0);
    if (stereo) {
        for (uint32_t v : scale_vals1)
            rice_encode(payload_writer, std::vector<uint32_t>{v}, k_scale1);
    }

    std::vector<uint32_t> k_sub_vals0, k_sub_vals1;
    std::vector<int> k_sub_per_band0(band_count, 0), k_sub_per_band1(band_count, 0);
    std::vector<std::vector<uint32_t>> uvals0(band_count), uvals1(band_count);
    k_sub_vals0.reserve(band_count);
    if (stereo) k_sub_vals1.reserve(band_count);

    for (int i = 0; i < band_count; ++i) {
        if (active0[i] && bits0[i] > 0) {
            const auto& qvals = qres0.quantized_per_level[i];
            auto& uv = uvals0[i];
            uv.resize(qvals.size());
            for (size_t j = 0; j < qvals.size(); ++j)
                uv[j] = zigzag_encode(qvals[j]);
            int k_sub = compute_optimal_rice_k(uv);
            k_sub_per_band0[i] = k_sub;
            k_sub_vals0.push_back(k_sub);
        }
        if (stereo && active1[i] && bits1[i] > 0) {
            const auto& qvals = qres1.quantized_per_level[i];
            auto& uv = uvals1[i];
            uv.resize(qvals.size());
            for (size_t j = 0; j < qvals.size(); ++j)
                uv[j] = zigzag_encode(qvals[j]);
            int k_sub = compute_optimal_rice_k(uv);
            k_sub_per_band1[i] = k_sub;
            k_sub_vals1.push_back(k_sub);
        }
    }

    int k_sub_k0 = k_sub_vals0.empty() ? 0 : std::min(compute_optimal_rice_k(k_sub_vals0), 3);
    int k_sub_k1 = 0;
    if (stereo && !k_sub_vals1.empty())
        k_sub_k1 = std::min(compute_optimal_rice_k(k_sub_vals1), 3);

    payload_writer.write_bits(k_sub_k0, 2);
    if (stereo) payload_writer.write_bits(k_sub_k1, 2);
    rice_encode(payload_writer, k_sub_vals0, k_sub_k0);
    if (stereo) rice_encode(payload_writer, k_sub_vals1, k_sub_k1);

    for (int i = 0; i < band_count; ++i) {
        if (active0[i] && bits0[i] > 0)
            rice_encode(payload_writer, uvals0[i], k_sub_per_band0[i]);
    }
    if (stereo) {
        for (int i = 0; i < band_count; ++i) {
            if (active1[i] && bits1[i] > 0)
                rice_encode(payload_writer, uvals1[i], k_sub_per_band1[i]);
        }
    }

    if (use_is) {
        std::vector<int> is_indices;
        for (int i = 0; i < band_count; ++i)
            if (is_band[i]) is_indices.push_back(i);

        if (!is_indices.empty()) {
            int segmented_threshold = 8 * total_bands / 16;
            int total_vals = 0;
            for (int idx : is_indices)
                total_vals += (idx < segmented_threshold) ? 4 : 1;

            uint32_t current_chunk = 0;
            int bits_in_chunk = 0;
            for (int i : is_indices) {
                bool is_seg = (i < segmented_threshold);
                int bits_to_add = is_seg ? 4 : 1;
                for (int s = 0; s < bits_to_add; ++s) {
                    int val = is_seg ? (is_inv_flags[i][s] ? 1 : 0) : (is_inv_flags[i][0] ? 1 : 0);
                    current_chunk = (current_chunk << 1) | val;
                    bits_in_chunk++;
                    if (bits_in_chunk == 32) {
                        payload_writer.write_bits(current_chunk, 32);
                        current_chunk = 0;
                        bits_in_chunk = 0;
                    }
                }
            }
            if (bits_in_chunk > 0) {
                payload_writer.write_bits(current_chunk, bits_in_chunk);
            }

            std::vector<uint32_t> is_zz_vals;
            is_zz_vals.reserve(total_vals);
            for (int i : is_indices) {
                int bits = get_r_bits(i);
                if (is_use_segmented[i]) {
                    for (int s = 0; s < 4; ++s) {
                        int32_t q = quantize_r_centered(is_r_vals[i][s], bits);
                        is_zz_vals.push_back(zigzag_encode(q));
                    }
                } else {
                    int32_t q = quantize_r_centered(is_r_vals[i][0], bits);
                    is_zz_vals.push_back(zigzag_encode(q));
                }
            }
            int k_is = compute_optimal_rice_k(is_zz_vals);
            payload_writer.write_bits(k_is & 0x07, 3);
            rice_encode(payload_writer, is_zz_vals, k_is);
        }
    }

    if (sbr_end > 0) {
        std::vector<int> sbr_indices;
        for (int i = 0; i < sbr_end; ++i)
            if (!active0[i]) sbr_indices.push_back(i);
        if (!sbr_indices.empty())
            sbr_write_to_stream(payload_writer, sbr, sbr_indices, can_state_pred, pred_state);
    }

    payload_writer.flush();
    auto payload = payload_writer.data();
    uint16_t payload_len = (uint16_t)payload.size();
    header.push_back(payload_len & 0xFF);
    header.push_back((payload_len >> 8) & 0xFF);

    std::vector<uint8_t> out = header;
    out.insert(out.end(), payload.begin(), payload.end());

    pred_state.prev_scale0 = full_scale0;
    pred_state.prev_scale1 = full_scale1;
    pred_state.prev_active0.assign(band_count, 0);
    for (int i = 0; i < band_count; ++i)
        if (active0[i]) pred_state.prev_active0[i] = 1;
    if (stereo) {
        pred_state.prev_active1.assign(band_count, 0);
        for (int i = 0; i < band_count; ++i)
            if (active1[i]) pred_state.prev_active1[i] = 1;
    } else {
        pred_state.prev_active1.assign(band_count, 0);
    }

    pred_state.prev_sbr_valid.assign(band_count, 0);
    pred_state.prev_sbr_noise.assign(band_count, 0);
    for (int i = 0; i < sbr_end && i < band_count; ++i) {
        if (!active0[i]) {
            pred_state.prev_sbr_valid[i] = 1;
            pred_state.prev_sbr_rms[i] = sbr.rms_idx[i];
            pred_state.prev_sbr_noise[i] = sbr.noise_flag[i] ? 1 : 0;
        }
    }

    pred_state.ready = true;
    pred_state.last_bi = block_index;

    return out;
}

void save_compressed_buffered(
    const std::vector<std::vector<uint8_t>>& blocks,
    const std::vector<bool>& block_modes,
    const std::string& path,
    uint32_t sr,
    int num_channels,
    float target_kbps,
    uint32_t total_frames,
    bool speed_mode, 
    size_t write_buffer_blocks = 8192)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot create output file");

    f.write("WHA1", 4);
    uint8_t version = 24;
    f.write((char*)&version, 1);
    f.write((char*)&sr, 4);
    uint8_t ch = (uint8_t)num_channels;
    f.write((char*)&ch, 1);
    uint32_t block_count = (uint32_t)blocks.size();
    f.write((char*)&block_count, 4);
    f.write((char*)&target_kbps, 4);
    uint8_t block_format_version = 24;
    f.write((char*)&block_format_version, 1);
    f.write((char*)&total_frames, 4);

    uint8_t flags = speed_mode ? 0x01 : 0x00;
    f.write((char*)&flags, 1);

    int mode_bytes = ((int)block_count + 7) / 8;
    std::vector<uint8_t> mode_packed(mode_bytes, 0);
    for (uint32_t i = 0; i < block_count; ++i)
        if (block_modes[i]) mode_packed[i / 8] |= (1 << (i % 8));
    f.write((char*)mode_packed.data(), mode_bytes);

    std::vector<uint8_t> write_buffer;
    write_buffer.reserve(write_buffer_blocks * 4096);

    auto flush_buffer = [&]() {
        if (!write_buffer.empty()) {
            f.write((char*)write_buffer.data(), write_buffer.size());
            write_buffer.clear();
        }
    };

    for (uint32_t i = 0; i < block_count; ++i) {
        const auto& blk = blocks[i];
        uint16_t len = (uint16_t)blk.size();
        write_buffer.insert(write_buffer.end(), (char*)&len, (char*)&len + 2);
        write_buffer.insert(write_buffer.end(), blk.begin(), blk.end());
        if (write_buffer.size() >= write_buffer_blocks * 4096)
            flush_buffer();
    }
    flush_buffer();
    f.close();

    int long_blocks = (int)std::count(block_modes.begin(), block_modes.end(), true);
    int short_blocks = (int)block_count - long_blocks;
    std::cout << "Saved container v24 to " << path << " ("
              << blocks.size() << " blocks, "
              << long_blocks << " long, "
              << short_blocks << " short, speed_mode=" << (speed_mode ? "yes" : "no") << ")" << std::endl;
}

std::tuple<std::vector<std::vector<uint8_t>>, uint32_t, int, float, uint32_t, std::vector<bool>, bool>
compress_audio_streaming(
    const std::string& input_path,
    float target_kbps_in,
    size_t read_buffer_size_samples = 8192,
    bool speed_mode = false) 
{
    uint32_t sr, channels;
    drwav wav;
    if (!drwav_init_file(&wav, input_path.c_str(), nullptr))
        throw std::runtime_error("Cannot open WAV file");

    sr = wav.sampleRate;
    channels = wav.channels;
    uint64_t total_frames = wav.totalPCMFrameCount;
    int num_channels = (int)channels;

    float target_kbps = (target_kbps_in > 0) ? target_kbps_in : SETTINGS.default_target_kbps;
    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor = (per_channel_kbps <= 0.0f) ? 0.0f : 1.0f;

    PWPT wpt;
    std::vector<std::vector<uint8_t>> blocks_raw;
    std::vector<bool> block_modes;
    PredContext pred_ctx;

    std::unique_ptr<PowCodec> pow_enc;
    if (!speed_mode) {
        pow_enc = std::make_unique<PowCodec>(num_channels, 0.85f, true, target_kbps / float(num_channels));
    }

    int current_pos = 0;
    int prev_block_size = 1024;
    float bits_per_sample = target_kbps * 1000.0f / sr;
    int reservoir = 0;
    int max_reservoir = (int)(bits_per_sample * 2048 * 0.3);

    std::vector<float> audio_buffer;
    size_t buffer_start = 0;
    bool eof = false;
    bool pow_flushed = false;

    std::vector<float> analyze_buf, block, read_chunk(read_buffer_size_samples * num_channels);
    std::vector<float> processed_buf, tail_buf;

    struct WindowCache {
        std::vector<float> buf;
        int bs = 0;
        int lo = -1;
        int ro = -1;
    };
    WindowCache win_long, win_short;

    auto get_window = [&](int bs, int lo, int ro) -> const std::vector<float>& {
        WindowCache& wc = (bs == 2048) ? win_long : win_short;
        if (wc.bs != bs || wc.lo != lo || wc.ro != ro) {
            wc.buf.assign(bs, 1.0f);
            if (lo > 1) {
                for (int i = 0; i < lo; ++i) {
                    float t = (float)i / lo;
                    wc.buf[i] = t * t * (3.0f - 2.0f * t);
                }
            }
            if (ro > 1) {
                for (int i = 0; i < ro; ++i) {
                    float t = (float)i / ro;
                    wc.buf[bs - 1 - i] = t * t * (3.0f - 2.0f * t);
                }
            }
            wc.bs = bs;
            wc.lo = lo;
            wc.ro = ro;
        }
        return wc.buf;
    };

    auto ensure_data_available = [&](int needed_pos) {
        while ((int)(buffer_start + audio_buffer.size() / num_channels) < needed_pos) {
            if (eof) {
                if (!pow_flushed) {
                    pow_flushed = true;
                    if (!speed_mode && pow_enc) {
                        tail_buf.clear();
                        pow_enc->flush(tail_buf);
                        audio_buffer.insert(audio_buffer.end(), tail_buf.begin(), tail_buf.end());
                        if (!tail_buf.empty()) continue;
                    }
                }
                break;
            }

            drwav_uint64 frames_read = drwav_read_pcm_frames_f32(
                &wav, read_buffer_size_samples, read_chunk.data());
            if (frames_read == 0) {
                eof = true;
                continue;
            }

            if (!speed_mode) {
                processed_buf.clear();
                pow_enc->process(read_chunk.data(), (size_t)frames_read, processed_buf);
                audio_buffer.insert(audio_buffer.end(), processed_buf.begin(), processed_buf.end());
            } else {
                audio_buffer.insert(audio_buffer.end(), read_chunk.begin(), read_chunk.begin() + frames_read * num_channels);
            }
        }
    };

    ensure_data_available(2048);

    while (current_pos < (int)total_frames) {
        int analyze_len = std::min(2048, (int)total_frames - current_pos);
        ensure_data_available(current_pos + analyze_len);
        if ((int)(buffer_start + audio_buffer.size() / num_channels) <= current_pos)
            break;

        analyze_buf.assign(analyze_len * num_channels, 0.0f);
        size_t offset = (current_pos - (int)buffer_start) * num_channels;
        if (offset + analyze_buf.size() <= audio_buffer.size()) {
            std::copy(
                audio_buffer.begin() + offset,
                audio_buffer.begin() + offset + analyze_buf.size(),
                analyze_buf.begin());
        } else {
            size_t avail = audio_buffer.size() - offset;
            std::copy(audio_buffer.begin() + offset, audio_buffer.end(), analyze_buf.begin());
        }

        bool is_transient = (num_channels == 2)
            ? detect_transient_block_stereo(analyze_buf, num_channels)
            : detect_transient_block_mono(analyze_buf, num_channels);

        int block_size, right_overlap, level;
        if (is_transient) {
            block_size = 1024;
            right_overlap = (int)(48 * overlap_factor);
            level = 4;
        } else {
            block_size = 2048;
            right_overlap = (int)(96 * overlap_factor);
            level = 5;
        }

        int left_overlap = (prev_block_size == 2048)
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);
        int hop = block_size - right_overlap;

        ensure_data_available(current_pos + block_size);

        block.assign(block_size * num_channels, 0.0f);
        size_t offset_block = (current_pos - (int)buffer_start) * num_channels;
        size_t avail_block = audio_buffer.size() - offset_block;
        if (avail_block > 0) {
            size_t copy_samples = std::min(avail_block, block.size());
            std::copy(
                audio_buffer.begin() + offset_block,
                audio_buffer.begin() + offset_block + copy_samples,
                block.begin());
        }

        const std::vector<float>& window = get_window(block_size, left_overlap, right_overlap);
        for (int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < block_size; ++i)
                block[i * num_channels + c] *= window[i];
        }

        bool block_is_full = (current_pos + block_size <= (int)total_frames);
        int target_bits = (int)(bits_per_sample * hop);
        if (target_bits < 256) target_bits = 256;
        int effective_budget = target_bits + reservoir;
        if (effective_budget > target_bits + max_reservoir)
            effective_budget = target_bits + max_reservoir;

        uint32_t block_index = (uint32_t)blocks_raw.size();
        auto comp = compress_block_adaptive_joint(
            block, wpt, num_channels, level, effective_budget, target_kbps, sr,
            block_size, SETTINGS.enable_ms, block_is_full, block_index, pred_ctx);

        int real_bits = (int)comp.size() * 8;
        reservoir += target_bits - real_bits;
        if (reservoir < 0) reservoir = 0;
        if (reservoir > max_reservoir) reservoir = max_reservoir;

        blocks_raw.push_back(std::move(comp));
        block_modes.push_back(!is_transient);

        if (SETTINGS.verbose && (blocks_raw.size() % 10 == 0)) {
            std::cout << "[Block " << blocks_raw.size() << "] size=" << block_size
                      << ", mode=" << (is_transient ? "short" : "long")
                      << ", bytes=" << blocks_raw.back().size()
                      << ", reservoir=" << reservoir << std::endl;
        }

        current_pos += hop;
        prev_block_size = block_size;

        int keep_start = current_pos - left_overlap;
        if (keep_start < 0) keep_start = 0;
        size_t remove_samples = (keep_start - (int)buffer_start) * num_channels;
        if (remove_samples > 0 && remove_samples <= audio_buffer.size()) {
            audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + remove_samples);
            buffer_start = keep_start;
        }

        ensure_data_available(current_pos + 2048);
    }

    drwav_uninit(&wav);

    return {
        std::move(blocks_raw),
        sr,
        num_channels,
        target_kbps,
        (uint32_t)total_frames,
        std::move(block_modes),
        speed_mode
    };
}

#include <sys/resource.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./encoder <input.wav> [bitrate_kbps] [read_buffer_samples] [is_max_base] [sbr_w0] [sbr_w1] [--speed]\n";
        return 1;
    }

    std::vector<char*> args;
    bool speed_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--speed") {
            speed_mode = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    
    int new_argc = args.size() + 1;
    char** new_argv = args.data();

    std::string input_file = new_argv[0];
    std::string out_container = input_file;
    for (int64_t i = 0; i < 3; i++) out_container.pop_back();
    out_container += "wha";

    float target_kbps = SETTINGS.default_target_kbps;
    if (new_argc >= 3) target_kbps = std::stof(new_argv[1]);

    size_t read_buffer = 8192;
    if (new_argc >= 4) {
        try {
            read_buffer = std::stoul(new_argv[2]);
            if (read_buffer < 256) read_buffer = 256;
        } catch (...) {
            std::cerr << "Invalid read buffer size\n";
        }
    }

    if (new_argc >= 5) {
        try {
            int mb = std::stoi(new_argv[3]);
            if (mb == 0) {
                SETTINGS.enable_adaptive_is = false;
                SETTINGS.adaptive_is_base_override = -1;
            } else if (mb > 0) {
                SETTINGS.enable_adaptive_is = true;
                SETTINGS.adaptive_is_base_override = mb;
            } else {
                SETTINGS.enable_adaptive_is = true;
                SETTINGS.adaptive_is_base_override = -1;
            }
        } catch (...) {
            SETTINGS.enable_adaptive_is = true;
            SETTINGS.adaptive_is_base_override = -1;
        }
    }

    if (new_argc >= 6) { try { SBR_W0 = std::stof(new_argv[4]); } catch (...) {} }
    if (new_argc >= 7) { try { SBR_W1 = std::stof(new_argv[5]); } catch (...) {} }

    std::cout << "SBR Shape Weights: W0=" << SBR_W0
              << ", W1=" << SBR_W1
              << " (Contrast: " << (20.0f * std::log10(SBR_W1 / SBR_W0)) << " dB)\n";
              
    if (speed_mode) {
        std::cout << ">>> SPEED MODE ENABLED: Pre-processing DISABLED (Flag will be saved to container) <<<\n";
    }

    struct rusage usage_before, usage_after;
    getrusage(RUSAGE_SELF, &usage_before);

    auto comp = compress_audio_streaming(input_file, target_kbps, read_buffer, speed_mode);
    auto blocks = std::get<0>(comp);
    uint32_t sr = std::get<1>(comp);
    int num_channels = std::get<2>(comp);
    float tk = std::get<3>(comp);
    uint32_t total_samples = std::get<4>(comp);
    auto block_modes = std::get<5>(comp);
    bool final_speed_mode = std::get<6>(comp);

    save_compressed_buffered(
        blocks, block_modes, out_container, sr, num_channels, tk,
        total_samples, final_speed_mode, read_buffer);

    getrusage(RUSAGE_SELF, &usage_after);
    double user_time_sec =
        (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
        (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;

    double audio_duration_sec = (total_samples > 0)
        ? (double)total_samples / (double)sr
        : 0.0;

    size_t total_file_bytes = 23; 
    for (const auto& blk : blocks)
        total_file_bytes += 2 + blk.size();

    double actual_bitrate_kbps = (audio_duration_sec > 0.0)
        ? (double)total_file_bytes * 8.0 / 1000.0 / audio_duration_sec
        : 0.0;

    double realtime_speed = (user_time_sec > 0.0)
        ? (audio_duration_sec / user_time_sec)
        : 0.0;

    std::cout << "Compressed: " << blocks.size() << " blocks, user time=" << user_time_sec << "s\n";
    std::cout << "Audio duration: " << audio_duration_sec << " s\n";
    std::cout << "File size: " << total_file_bytes << " bytes\n";
    std::cout << "Target bitrate: " << target_kbps << " kbps\n";
    std::cout << "Actual bitrate: " << actual_bitrate_kbps << " kbps\n";
    std::cout << "Encoding speed: " << realtime_speed << "x realtime\n";

    return 0;
}