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
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "wavelet.h"
#include "perc_prior.h"
#include "bit_alloc.h"
#include "joint_stereo.h"
#include "quantize.h"
#include "entropy_encoder.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int16_t exp = ((x >> 23) & 0xFF) - 112;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) return sign;
    if (exp > 30) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

struct EncoderSettings {
    bool enable_ms = true;
    int reservoir_max_factor = 1024;
    float default_target_kbps = 128.0f;
    bool verbose = false;
    float transient_ratio_threshold = 0.15f;
} SETTINGS;

static int get_scale_bits(int band_idx) {
    if (band_idx < 1) return 12;
    return 8;
}

static int compute_optimal_rice_k(const std::vector<uint32_t>& vals, int max_k = 7) {
    int best_k = 0;
    size_t best_bits = SIZE_MAX;
    for (int k = 0; k <= max_k; ++k) {
        size_t total = 0;
        for (uint32_t v : vals) {
            total += (v >> k) + 1 + k;
        }
        if (total < best_bits) {
            best_bits = total;
            best_k = k;
        }
    }
    return best_k;
}

static bool detect_transient_block_stereo(const std::vector<float>& block, int num_channels) {
    int N = block.size() / num_channels;
    const float* data = block.data();
    if (N < 64) return false;
    float diff_energy = 0.0f, total_energy = 0.0f, max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float sample = data[i * num_channels];
        float av = std::abs(sample);
        if (av > max_abs) max_abs = av;
        total_energy += sample * sample;
        if (i > 0) {
            float diff = sample - data[(i - 1) * num_channels];
            diff_energy += diff * diff;
        }
    }
    if (total_energy < 1e-6f) return false;
    float smoothness_ratio = diff_energy / total_energy;
    float rms = std::sqrt(total_energy / N);
    float peak_to_rms = max_abs / (rms + 1e-12f);
    
    static float smoothed_peak = 0.0f;
    static bool initialized = false;
    
    if (!initialized) {
        smoothed_peak = peak_to_rms;
        initialized = true;
    } else {
        float alpha = (peak_to_rms > smoothed_peak) ? 0.75f : 0.4f;
        smoothed_peak = alpha * peak_to_rms + (1.0f - alpha) * smoothed_peak;
    }
        
    return !((smoothness_ratio < 0.5f) && (smoothed_peak < 2.96f));
}

static bool detect_transient_block_mono(const std::vector<float>& block, int num_channels) {
    int N = block.size() / num_channels;
    const float* data = block.data();
    if (N < 64) return false;
    float diff_energy = 0.0f, total_energy = 0.0f, max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float sample = data[i * num_channels];
        float av = std::abs(sample);
        if (av > max_abs) max_abs = av;
        total_energy += sample * sample;
        if (i > 0) {
            float diff = sample - data[(i - 1) * num_channels];
            diff_energy += diff * diff;
        }
    }
    if (total_energy < 1e-6f) return false;
    float smoothness_ratio = diff_energy / total_energy;
    float rms = std::sqrt(total_energy / N);
    float peak_to_rms = max_abs / (rms + 1e-12f);
    // std::cout << peak_to_rms << '\n';
    return !((smoothness_ratio < 0.5f) && (peak_to_rms < 3.00f));
}

static std::vector<uint8_t> pack_bits(const std::vector<bool>& bits, int start, int end) {
    int count = end - start;
    int bytes = (count + 7) / 8;
    std::vector<uint8_t> packed(bytes, 0);
    for (int i = start; i < end; ++i) {
        if (bits[i]) {
            int idx = i - start;
            packed[idx / 8] |= (1 << (idx % 8));
        }
    }
    return packed;
}

static float scale_and_bits_to_step(float scale, int bits) {
    if (bits <= 0) return 0.0f;
    int max_int = (bits > 1) ? ((1 << (bits - 1)) - 1) : 1;
    return scale / max_int;
}

static bool detect_strong_transient(const std::vector<float>& energy0,
                                    const std::vector<float>& energy1,
                                    int band_count, bool stereo, float threshold) {
    if (band_count < 3) return false;
    float total = 0.0f, high = 0.0f;
    int high_start = band_count / 4;
    for (int i = 0; i < band_count; ++i) {
        float e = energy0[i] + (stereo ? energy1[i] : 0.0f);
        total += e;
        if (i >= high_start) high += e;
    }
    if (total < 1e-12f) return false;
    return ((high / total) > 0.15);
}

std::vector<uint8_t> compress_block_adaptive_joint(
    const std::vector<float>& block, PWPT& wpt, int num_channels, int level,
    int target_bits_budget, float target_kbps, int sr, int block_size,
    bool enable_ms, bool block_is_full)
{
    const float eps = 1e-12f;
    const bool stereo = (num_channels == 2);
    std::vector<std::vector<float>> left_coeffs, right_coeffs;
    std::vector<int> coeff_counts;

    if (stereo) {
        size_t samples_per_channel = block.size() / 2;
        std::vector<float> left(samples_per_channel), right(samples_per_channel);
        for (size_t i = 0; i < samples_per_channel; ++i) {
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
    bool use_is = (stereo && target_kbps < 510.0f);
    std::vector<uint8_t> mode_ms(band_count, 0);
    std::vector<std::vector<float>> ch0_bands(band_count);
    std::vector<std::vector<float>> ch1_bands(band_count);

    std::vector<std::vector<float>> ch0_original(band_count);
    for (int i = 0; i < band_count; ++i) {
        ch0_original[i] = left_coeffs[i];
    }

    int is_start_band = band_count;
    std::vector<std::array<float, 4>> is_r_vals(band_count);
    for (auto& a : is_r_vals) a.fill(0.5f);
    std::vector<bool> is_inv_flags(band_count, false);
    std::vector<bool> is_use_segmented(band_count, false);

    if (use_is) {
        is_start_band = get_is_start_band(target_kbps, total_bands);
        if (is_start_band >= band_count) use_is = false;
    }

    if (stereo) {
        for (int i = 0; i < is_start_band; ++i) {
            const auto& left_band = left_coeffs[i];
            const auto& right_band = right_coeffs[i];
            int n = (int)left_band.size();
            if (n == 0) { ch0_bands[i] = left_band; ch1_bands[i] = right_band; continue; }
            float El = 0, Er = 0;
            for (int j = 0; j < n; ++j) { El += left_band[j]*left_band[j]; Er += right_band[j]*right_band[j]; }
            auto [mid, side] = mid_side(left_band, right_band);
            float Em = 0, Es = 0;
            for (int j = 0; j < n; ++j) { Em += mid[j]*mid[j]; Es += side[j]*side[j]; }
            bool use_ms = use_mid_side(El, Er, Em, Es, enable_ms, target_kbps);
            if (use_ms || target_kbps <= 0) {
                mode_ms[i] = 1;
                ch0_bands[i] = std::move(mid);
                ch1_bands[i] = std::move(side);
            } else {
                mode_ms[i] = 0;
                ch0_bands[i] = std::move(left_band);
                ch1_bands[i] = std::move(right_band);
            }
        }
    } else {
        for (int i = 0; i < band_count; ++i) {
            ch0_bands[i] = left_coeffs[i];
            ch1_bands[i].clear();
        }
    }

    if (use_is && is_start_band < band_count) {
        for (int i = is_start_band; i < band_count; ++i) {
            mode_ms[i] = 1;
            std::array<float, 4> r_arr;
            bool inv_flag, use_segmented;
            std::vector<float> Y;
            compute_is_parameters_ex(left_coeffs[i], right_coeffs[i], Y, r_arr, inv_flag, use_segmented, i, total_bands);
            is_r_vals[i] = r_arr;
            is_inv_flags[i] = inv_flag;
            is_use_segmented[i] = use_segmented;
            ch0_bands[i] = Y;
            ch1_bands[i] = Y;
        }
    }

    std::vector<float> energy0(band_count, 0.0f), energy1(band_count, 0.0f);
    float total_energy = 0;
    for (int i = 0; i < band_count; ++i) {
        for (float v : ch0_bands[i]) energy0[i] += v * v;
        for (float v : ch1_bands[i]) energy1[i] += v * v;
        total_energy += energy0[i];
        if (i < is_start_band) total_energy += energy1[i];
    }
    for (int i = 0; i < band_count; i++) {
        energy0[i] = std::pow(energy0[i], 0.75);
        energy1[i] = std::pow(energy1[i], 0.75);
    }

    bool has_transient = detect_strong_transient(energy0, energy1, band_count, stereo,
                                                 SETTINGS.transient_ratio_threshold);
    if (!has_transient) {
        float exp;
        if (target_kbps <= 128.0f) exp = 0.75f;
        else if (target_kbps >= 192.0f) exp = 0.95f;
        else exp = 0.75f + (0.95f - 0.75f) * (target_kbps - 128.0f) / (192.0f - 128.0f);
        exp = std::clamp(exp, 0.5f, 1.0f);
        float max_energy = 0;
        for (int i = 0; i < band_count; i++)
            max_energy = std::max({max_energy, energy0[i], energy1[i]});
        for (int i = 0; i < band_count; ++i) {
            if (energy0[i] / max_energy > 1e-5) energy0[i] = std::pow(energy0[i], exp);
            if (stereo && energy1[i] / max_energy > 1e-5) energy1[i] = std::pow(energy1[i], exp);
        }
    }

    {
        float total_e = 0.0f, low_freq_e = 0.0f;
        int half_bands = band_count / 2;
        for (int i = 0; i < band_count; ++i) {
            total_e += energy0[i];
            if (i < half_bands) low_freq_e += energy0[i];
        }
        bool is_narrowband = (total_e > 1e-6f) && (low_freq_e / total_e > 0.98f);
        if (is_narrowband) {
            for (int i = band_count / 2; i < band_count; ++i) {
                energy0[i] *= 0.5f;
                if (stereo) energy1[i] *= 0.5f;
            }
        }
    }

    static std::vector<std::vector<float>> prev_ch0, prev_ch1;
    std::vector<float> priority0, priority1;
    if (stereo) {
        priority0 = compute_channel_priority(ch0_bands, prev_ch0, coeff_counts, target_kbps, sr, level, false);
        priority1 = compute_channel_priority(ch1_bands, prev_ch1, coeff_counts, target_kbps, sr, level, true);
    } else {
        priority0 = compute_channel_priority(ch0_bands, prev_ch0, coeff_counts, target_kbps, sr, level, false);
        priority1.assign(band_count, 0.0f);
    }

    std::vector<int> min_bits(band_count, 0);
    std::vector<int> max_bits(band_count, 10);
    if (use_is && is_start_band < band_count) {
        for (int i = is_start_band; i < band_count; ++i) {
            min_bits[i] = 0;
            priority1[i] = 0.0f;
            energy1[i] = 0.0f;
        }
    }

    int bits_header = 0;
    if (stereo) bits_header += ((band_count+7)/8)*8;
    bits_header += ((band_count+7)/8)*8;
    if (stereo) bits_header += ((band_count+7)/8)*8;
    bits_header += 8;

    int header_bytes_est = (bits_header + 7)/8 + 2;
    int payload_budget = target_bits_budget - header_bytes_est * 8;
    if (payload_budget < 0) payload_budget = target_bits_budget / 2;
    int reservoir_max = int(payload_budget * SETTINGS.reservoir_max_factor);

    DualAllocResult alloc = allocate_bits_dual(
        priority0, priority1, coeff_counts, payload_budget,
        min_bits, max_bits, energy0, energy1, 0, reservoir_max, target_kbps, num_channels == 2);

    std::vector<int> bits0 = alloc.bits0;
    std::vector<int> bits1 = alloc.bits1;
    if (use_is && is_start_band < band_count) {
        for (int i = is_start_band; i < band_count; ++i) bits1[i] = 0;
    }

    std::vector<bool> active0(band_count), active1(band_count);
    for (int i = 0; i < band_count; ++i) {
        active0[i] = (bits0[i] > 0);
        if (stereo) active1[i] = (bits1[i] > 0);
    }

    std::vector<bool> sbr_noise_flag(band_count, false);   
    std::vector<uint32_t> sbr_rms_idx(band_count, 0);
    int sbr_end = block_is_full ? (3 * band_count / 4) : 0;

    const float LOG_MIN = -6.0f;
    const float LOG_MAX =  0.0f;
    auto get_scale_idx = [&](float step, int sb) -> uint32_t {
        if (step <= 0) return 0;
        float log_s = log10f(std::max(step, 1e-12f));
        int max_idx = (1 << sb) - 1;
        int idx = (int)((log_s - LOG_MIN) * max_idx / (LOG_MAX - LOG_MIN));
        idx = std::clamp(idx, 0, max_idx);
        return (uint32_t)idx;
    };

    for (int i = 0; i < sbr_end; ++i) {
        if (!active0[i]) {
            const auto& orig = ch0_original[i];
            double sum_sq = 0.0;
            float max_abs = 0.0f;
            for (float v : orig) {
                sum_sq += static_cast<double>(v) * v;
                float av = std::abs(v);
                if (av > max_abs) max_abs = av;
            }
            float rms = std::sqrt(static_cast<float>(sum_sq / orig.size()) + eps);
            float crest = max_abs / (rms + eps);
            sbr_noise_flag[i] = (crest < 2.5f);
            int sb = get_scale_bits(i);
            sbr_rms_idx[i] = get_scale_idx(rms, sb);
        }
    }
    std::vector<uint8_t> header;
    int mask_bytes = (band_count + 7) / 8;

    std::vector<uint8_t> mode_bytes_vec((band_count + 7) / 8, 0);
    for (int i = 0; i < band_count; ++i)
        if (mode_ms[i]) mode_bytes_vec[i / 8] |= (1 << (i % 8));

    std::vector<uint8_t> mask0(mask_bytes, 0);
    for (int i = 0; i < band_count; ++i)
        if (active0[i]) mask0[i / 8] |= (1 << (i % 8));

    std::vector<uint8_t> mask1;
    if (stereo) {
        mask1.assign(mask_bytes, 0);
        for (int i = 0; i < band_count; ++i)
            if (active1[i]) mask1[i / 8] |= (1 << (i % 8));
    }

    header.insert(header.end(), mode_bytes_vec.begin(), mode_bytes_vec.end());
    header.insert(header.end(), mask0.begin(), mask0.end());
    if (stereo) header.insert(header.end(), mask1.begin(), mask1.end());

    std::vector<std::vector<std::vector<float>>> for_quant0 = { ch0_bands };
    QuantResult qres0 = quantize_levels(for_quant0, bits0, target_kbps);

    QuantResult qres1;
    if (stereo) {
        std::vector<std::vector<std::vector<float>>> for_quant1 = { ch1_bands };
        qres1 = quantize_levels(for_quant1, bits1, target_kbps);
    }

    std::vector<uint32_t> scale_indices0, scale_indices1;
    for (int i = 0; i < band_count; ++i) {
        if (active0[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres0.scales[i], bits0[i]);
            uint32_t idx = get_scale_idx(step, sb);
            scale_indices0.push_back(idx);
            // std::cout << idx << ' ';
        }
        if (stereo && active1[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres1.scales[i], bits1[i]);
            uint32_t idx = get_scale_idx(step, sb);
            scale_indices1.push_back(idx);
            // std::cout << idx << ' ';
        }
    }

    int k_scale0 = compute_optimal_rice_k(scale_indices0);
    int k_scale1 = stereo ? compute_optimal_rice_k(scale_indices1) : 0;

    uint8_t k_scale_byte = (uint8_t)(k_scale0 & 0x07);
    if (stereo) k_scale_byte |= ((k_scale1 & 0x07) << 3);
    if (block_is_full) k_scale_byte |= 0x80;   
    header.push_back(k_scale_byte);

    BitWriterMSB payload_writer;

    for (uint32_t idx : scale_indices0)
        rice_encode(payload_writer, std::vector<uint32_t>{idx}, k_scale0);
    if (stereo) {
        for (uint32_t idx : scale_indices1)
            rice_encode(payload_writer, std::vector<uint32_t>{idx}, k_scale1);
    }

    for (int i = 0; i < band_count; ++i) {
        if (!active0[i] || bits0[i] <= 0) continue;
        const auto& qvals = qres0.quantized_per_level[i];
        std::vector<uint32_t> uvals(qvals.size());
        for (size_t j = 0; j < qvals.size(); ++j) uvals[j] = zigzag_encode(qvals[j]);
        int k_sub = compute_optimal_rice_k(uvals);
        payload_writer.write_bits(k_sub, 3);
        rice_encode(payload_writer, uvals, k_sub);
    }

    if (stereo) {
        for (int i = 0; i < band_count; ++i) {
            if (!active1[i] || bits1[i] <= 0) continue;
            const auto& qvals = qres1.quantized_per_level[i];
            std::vector<uint32_t> uvals(qvals.size());
            for (size_t j = 0; j < qvals.size(); ++j) uvals[j] = zigzag_encode(qvals[j]);
            int k_sub = compute_optimal_rice_k(uvals);
            payload_writer.write_bits(k_sub, 3);
            rice_encode(payload_writer, uvals, k_sub);
        }
    }

    payload_writer.flush();
    std::vector<uint8_t> payload = payload_writer.data();

    uint16_t payload_len = (uint16_t)payload.size();
    header.push_back(payload_len & 0xFF);
    header.push_back((payload_len >> 8) & 0xFF);

    std::vector<uint8_t> out = header;
    out.insert(out.end(), payload.begin(), payload.end());

    if (use_is && is_start_band < band_count) {
        BitWriterMSB r_writer;
        for (int i = is_start_band; i < band_count; ++i) {
            int bits = get_r_bits(i);
            if (is_use_segmented[i]) {
                for (int s = 0; s < 4; ++s) {
                    uint32_t q = quantize_r(is_r_vals[i][s], bits);
                    r_writer.write_bits(q, bits);
                }
            } else {
                uint32_t q = quantize_r(is_r_vals[i][0], bits);
                r_writer.write_bits(q, bits);
            }
        }
        r_writer.flush();
        std::vector<uint8_t> r_bytes = r_writer.data();
        out.insert(out.end(), r_bytes.begin(), r_bytes.end());

        std::vector<bool> inv_bits(band_count, false);
        for (int i = is_start_band; i < band_count; ++i) inv_bits[i] = is_inv_flags[i];
        std::vector<uint8_t> inv_mask = pack_bits(inv_bits, is_start_band, band_count);
        out.insert(out.end(), inv_mask.begin(), inv_mask.end());
    }

    if (sbr_end > 0) {
        std::vector<int> sbr_indices;
        for (int i = 0; i < sbr_end; ++i) {
            if (!active0[i]) {
                sbr_indices.push_back(i);
            }
        }
        if (!sbr_indices.empty()) {
            std::vector<bool> noise_bits;
            noise_bits.reserve(sbr_indices.size());
            for (int idx : sbr_indices) {
                noise_bits.push_back(sbr_noise_flag[idx]);
            }
            std::vector<uint8_t> noise_packed = pack_bits(noise_bits, 0, (int)noise_bits.size());
            out.insert(out.end(), noise_packed.begin(), noise_packed.end());

            BitWriterMSB rms_writer;
            for (int idx : sbr_indices) {
                int sb = get_scale_bits(idx);
                rms_writer.write_bits(sbr_rms_idx[idx], sb);
            }
            rms_writer.flush();
            std::vector<uint8_t> rms_bytes = rms_writer.data();
            out.insert(out.end(), rms_bytes.begin(), rms_bytes.end());
        }
    }

    return out;
}

void save_compressed_buffered(const std::vector<std::vector<uint8_t>>& blocks,
                              const std::vector<bool>& block_modes,
                              const std::string& path,
                              uint32_t sr, int num_channels,
                              float target_kbps,
                              size_t write_buffer_blocks = 8192)  
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot create output file");

    f.write("WHA1", 4);
    uint8_t version = 18;
    f.write((char*)&version, 1);
    f.write((char*)&sr, 4);
    uint8_t ch = (uint8_t)num_channels;
    f.write((char*)&ch, 1);
    uint32_t block_count = (uint32_t)blocks.size();
    f.write((char*)&block_count, 4);
    f.write((char*)&target_kbps, 4);
    uint8_t block_format_version = 18;
    f.write((char*)&block_format_version, 1);

    int mode_bytes = (block_count + 7) / 8;
    std::vector<uint8_t> mode_packed(mode_bytes, 0);
    for (uint32_t i = 0; i < block_count; ++i) {
        if (block_modes[i]) mode_packed[i / 8] |= (1 << (i % 8));
    }
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
        uint32_t len = (uint32_t)blk.size();
        write_buffer.insert(write_buffer.end(), (char*)&len, (char*)&len + 4);
        write_buffer.insert(write_buffer.end(), blk.begin(), blk.end());

        if (write_buffer.size() >= write_buffer_blocks * 4096) {
            flush_buffer();
        }
    }
    flush_buffer();

    f.close();

    int long_blocks = std::count(block_modes.begin(), block_modes.end(), true);
    int short_blocks = block_count - long_blocks;
    std::cout << "Saved container v18 to " << path << " (" << blocks.size() << " blocks, "
              << long_blocks << " long, " << short_blocks << " short)" << std::endl;
}

std::tuple<std::vector<std::vector<uint8_t>>, uint32_t, int, float, uint32_t, std::vector<bool>>
compress_audio_streaming(const std::string& input_path,
                         float target_kbps_in,
                         size_t read_buffer_size_samples = 8192)  
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
    float overlap_factor = (per_channel_kbps <= 0.0f) ? 0.5f : 1.0f;

    PWPT wpt;
    std::vector<std::vector<uint8_t>> blocks_raw;
    std::vector<bool> block_modes;

    int current_pos = 0;          
    int prev_block_size = 1024;
    float bits_per_sample = target_kbps * 1000.0f / sr;
    int reservoir = 0;
    int max_reservoir = (int)(bits_per_sample * 2048 * 0.3);

    std::vector<float> audio_buffer;
    size_t buffer_start = 0;       

    auto ensure_data_available = [&](int needed_pos) {
        while ((int)(buffer_start + audio_buffer.size() / num_channels) < needed_pos) {
            size_t samples_to_read = read_buffer_size_samples * num_channels;
            std::vector<float> chunk(samples_to_read);
            drwav_uint64 frames_read = drwav_read_pcm_frames_f32(&wav, read_buffer_size_samples, chunk.data());
            if (frames_read == 0) break;
            chunk.resize(frames_read * num_channels);
            audio_buffer.insert(audio_buffer.end(), chunk.begin(), chunk.end());
        }
    };

    ensure_data_available(2048);  

    while (current_pos < (int)total_frames) {
        int analyze_len = std::min(2048, (int)total_frames - current_pos);
        ensure_data_available(current_pos + analyze_len);
        if ((int)(buffer_start + audio_buffer.size() / num_channels) <= current_pos) break; 

        std::vector<float> analyze_buf(analyze_len * num_channels);
        size_t offset = (current_pos - (int)buffer_start) * num_channels;
        if (offset + analyze_buf.size() <= audio_buffer.size()) {
            std::copy(audio_buffer.begin() + offset,
                      audio_buffer.begin() + offset + analyze_buf.size(),
                      analyze_buf.begin());
        } else {
            size_t avail = audio_buffer.size() - offset;
            std::copy(audio_buffer.begin() + offset, audio_buffer.end(), analyze_buf.begin());
            std::fill(analyze_buf.begin() + avail / num_channels, analyze_buf.end(), 0.0f);
        }

        bool is_transient = false;
        if (num_channels == 2) {
            is_transient = detect_transient_block_stereo(analyze_buf, num_channels);
        } else {
            is_transient = detect_transient_block_mono(analyze_buf, num_channels);
        }

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

        int left_overlap = (prev_block_size == 2048) ? (int)(96 * overlap_factor) : (int)(48 * overlap_factor);
        int hop = block_size - right_overlap;

        ensure_data_available(current_pos + block_size);

        std::vector<float> block(block_size * num_channels, 0.0f);
        size_t offset_block = (current_pos - (int)buffer_start) * num_channels;
        size_t avail_block = audio_buffer.size() - offset_block;
        if (avail_block > 0) {
            size_t copy_samples = std::min(avail_block, block.size());
            std::copy(audio_buffer.begin() + offset_block,
                      audio_buffer.begin() + offset_block + copy_samples,
                      block.begin());
        }

        std::vector<float> window(block_size, 1.0f);
        if (left_overlap > 1) {
            for (int i = 0; i < left_overlap; ++i) {
                float t = (float)i / left_overlap;
                window[i] = t * t * (3.0f - 2.0f * t);
            }
        }
        if (right_overlap > 1) {
            for (int i = 0; i < right_overlap; ++i) {
                float t = (float)i / right_overlap;
                window[block_size - 1 - i] = t * t * (3.0f - 2.0f * t);
            }
        }
        for (int c = 0; c < num_channels; ++c)
            for (int i = 0; i < block_size; ++i)
                block[i * num_channels + c] *= window[i];

        bool block_is_full = (current_pos + block_size <= (int)total_frames);

        int target_bits = (int)(bits_per_sample * hop);
        if (target_bits < 256) target_bits = 256;

        int effective_budget = target_bits + reservoir;
        if (effective_budget > target_bits + max_reservoir)
            effective_budget = target_bits + max_reservoir;

        auto comp = compress_block_adaptive_joint(block, wpt, num_channels, level,
                                                  effective_budget, target_kbps, sr, block_size,
                                                  SETTINGS.enable_ms, block_is_full);

        int real_bits = (int)comp.size() * 8;
        reservoir += target_bits - real_bits;
        if (reservoir < 0) reservoir = 0;
        if (reservoir > max_reservoir) reservoir = max_reservoir;

        int target_bytes = target_bits / 8;
        if (target_bytes > 0 && (int)comp.size() < (int)(0.4f * target_bytes)) {
            int min_needed = (int)(0.9f * target_bytes);
            if ((int)comp.size() < min_needed) comp.resize(min_needed, 0);
        }

        blocks_raw.push_back(comp);
        block_modes.push_back(!is_transient);

        if (SETTINGS.verbose && (blocks_raw.size() % 10 == 0)) {
            std::cout << "[Block " << blocks_raw.size() << "] size=" << block_size
                      << ", mode=" << (is_transient ? "short" : "long")
                      << ", bytes=" << comp.size() << ", reservoir=" << reservoir << std::endl;
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

    return {blocks_raw, sr, num_channels, target_kbps, (uint32_t)total_frames, block_modes};
}

#include <sys/resource.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./encoder <input.wav> [bitrate_kbps] [read_buffer_samples]\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string out_container = input_file;
    for (int64_t i = 0; i < 3; i++) out_container.pop_back();
    out_container += "wha";

    float target_kbps = SETTINGS.default_target_kbps;
    if (argc >= 3) target_kbps = std::stof(argv[2]);

    size_t read_buffer = 8192;  
    if (argc >= 4) {
        try {
            read_buffer = std::stoul(argv[3]);
            if (read_buffer < 256) read_buffer = 256;
        } catch (...) {
            std::cerr << "Invalid read buffer size, using default 8192\n";
        }
    }

    struct rusage usage_before, usage_after;
    getrusage(RUSAGE_SELF, &usage_before);

    auto comp = compress_audio_streaming(input_file, target_kbps, read_buffer);
    auto blocks = std::get<0>(comp);
    uint32_t sr = std::get<1>(comp);
    int num_channels = std::get<2>(comp);
    float tk = std::get<3>(comp);
    uint32_t total_samples = std::get<4>(comp);
    auto block_modes = std::get<5>(comp);
    int block_count = (int)blocks.size();

    save_compressed_buffered(blocks, block_modes, out_container, sr, num_channels, tk, read_buffer);

    getrusage(RUSAGE_SELF, &usage_after);
    double user_time_sec = (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
                           (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;
    double audio_duration_sec = (total_samples > 0) ? (double)total_samples / (double)sr : 0.0;

    size_t total_file_bytes = 22;
    for (const auto& blk : blocks) total_file_bytes += 4 + blk.size();

    double actual_bitrate_kbps = (audio_duration_sec > 0.0)
        ? (double)total_file_bytes * 8.0 / 1000.0 / audio_duration_sec : 0.0;
    double realtime_speed = (user_time_sec > 0.0) ? (audio_duration_sec / user_time_sec) : 0.0;

    std::cout << "Compressed: " << block_count << " blocks, user time=" << user_time_sec << "s\n";
    std::cout << "Audio duration: " << audio_duration_sec << " s\n";
    std::cout << "File size: " << total_file_bytes << " bytes\n";
    std::cout << "Target bitrate: " << target_kbps << " kbps\n";
    std::cout << "Actual bitrate: " << actual_bitrate_kbps << " kbps\n";
    std::cout << "Encoding speed: " << realtime_speed << "x realtime\n";

    return 0;
}