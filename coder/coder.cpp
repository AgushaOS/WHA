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

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "wavelet.h"
#include "perc_prior.h"
#include "bit_alloc.h"
#include "joint_stereo.h"
#include "quantize.h"
#include "entropy_encoder.h"

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
    int wpt_level = 4;
    int block_samples = 1024;
    int overlap = 48;
    bool enable_ms = true;
    int reservoir_max_factor = 1024;   
    float default_target_kbps = 160.0f;
    bool verbose = false;
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

std::vector<float> read_wav_f32(const std::string &path, uint32_t &sr, uint32_t &channels) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr))
        throw std::runtime_error("failed to open wav");
    sr = wav.sampleRate;
    channels = wav.channels;
    uint64_t frames = wav.totalPCMFrameCount;
    std::vector<float> out(frames * channels);
    drwav_read_pcm_frames_f32(&wav, frames, out.data());
    drwav_uninit(&wav);
    return out;
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

std::vector<uint8_t> compress_block_adaptive_joint(
    const std::vector<float>& block,
    PWPT& wpt,
    int num_channels,
    int level,
    int target_bits_budget,
    float target_kbps,
    int sr,
    int block_size,
    int version,
    bool enable_ms)
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
        left_coeffs = wpt.wpt(left, level, target_kbps, sr);
        right_coeffs = wpt.wpt(right, level, target_kbps, sr);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i)
            coeff_counts[i] = (int)left_coeffs[i].size();
    } else {
        left_coeffs = wpt.wpt(block, level, target_kbps, sr);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i)
            coeff_counts[i] = (int)left_coeffs[i].size();
    }
    int band_count = (int)coeff_counts.size();

    uint8_t block_ver = 13;
    bool use_is = (stereo && target_kbps < 510.0f);

    std::vector<uint8_t> mode_ms(band_count, 0);
    std::vector<std::vector<float>> ch0_bands(band_count);
    std::vector<std::vector<float>> ch1_bands(band_count);

    int is_start_band = band_count;
    std::vector<float> is_r_vals(band_count, 0.5f);
    std::vector<bool> is_inv_flags(band_count, false);   
    if (use_is) {
        is_start_band = get_is_start_band(target_kbps);
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
            if (use_ms) {
                mode_ms[i] = 1;
                ch0_bands[i] = std::move(mid);
                ch1_bands[i] = std::move(side);
            } else {
                mode_ms[i] = 0;
                ch0_bands[i] = left_band;
                ch1_bands[i] = right_band;
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
            float r, sqrtE;
            bool inv_flag;
            std::vector<float> Y;
            compute_is_parameters_ex(left_coeffs[i], right_coeffs[i], Y, r, sqrtE, inv_flag);
            is_r_vals[i] = r;
            is_inv_flags[i] = inv_flag;
            ch0_bands[i] = std::move(Y);
            ch1_bands[i].clear();
        }
        block_ver = 13; 
    }

    std::vector<float> energy0(band_count, 0.0f), energy1(band_count, 0.0f);
    for (int i = 0; i < band_count; ++i) {
        for (float v : ch0_bands[i]) energy0[i] += v * v;
        for (float v : ch1_bands[i]) energy1[i] += v * v;
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

    int bits_header = 4*8 + 8 + 8; 
    if (stereo) bits_header += ((band_count+7)/8)*8; 
    else bits_header += 0;
    bits_header += ((band_count+7)/8)*8; 
    if (stereo) bits_header += ((band_count+7)/8)*8; 
    bits_header += 8; 
    int header_bytes_est = (bits_header + 7)/8 + 4; 
    if (block_ver >= 9 && use_is && is_start_band < band_count) {
        int total_r_bits = 0;
        for (int i = is_start_band; i < band_count; ++i) total_r_bits += get_r_bits(i);
        header_bytes_est += (total_r_bits + 7) / 8;
        if (block_ver >= 12) {
            int inv_mask_bytes = (band_count - is_start_band + 7) / 8;
            header_bytes_est += inv_mask_bytes;
        }
    }
    int payload_budget = target_bits_budget - header_bytes_est * 8;
    if (payload_budget < 0) payload_budget = target_bits_budget / 2;

    int reservoir_max = int(payload_budget * SETTINGS.reservoir_max_factor);
    DualAllocResult alloc = allocate_bits_dual(
        priority0, priority1, coeff_counts, payload_budget,
        min_bits, max_bits, energy0, energy1, 0, reservoir_max, target_kbps);
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

    header.insert(header.end(), {'B','L','O','C'});
    header.push_back(block_ver);
    header.push_back((uint8_t)mode_bytes_vec.size()); 
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

    std::vector<uint32_t> scale_indices0, scale_indices1;
    for (int i = 0; i < band_count; ++i) {
        if (active0[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres0.scales[i], bits0[i]);
            uint32_t idx = get_scale_idx(step, sb);
            scale_indices0.push_back(idx);
        }
        if (stereo && active1[i]) {
            int sb = get_scale_bits(i);
            float step = scale_and_bits_to_step(qres1.scales[i], bits1[i]);
            uint32_t idx = get_scale_idx(step, sb);
            scale_indices1.push_back(idx);
        }
    }

    int k_scale0 = compute_optimal_rice_k(scale_indices0);
    int k_scale1 = stereo ? compute_optimal_rice_k(scale_indices1) : 0;

    uint8_t k_scale_byte = (uint8_t)(k_scale0 & 0x07);
    if (stereo) k_scale_byte |= ((k_scale1 & 0x07) << 3);
    header.push_back(k_scale_byte);

    BitWriterMSB payload_writer;

    for (uint32_t idx : scale_indices0) {
        rice_encode(payload_writer, std::vector<uint32_t>{idx}, k_scale0);
    }
    if (stereo) {
        for (uint32_t idx : scale_indices1) {
            rice_encode(payload_writer, std::vector<uint32_t>{idx}, k_scale1);
        }
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

    if (block_ver >= 9 && use_is && is_start_band < band_count) {
        BitWriterMSB r_writer;
        for (int i = is_start_band; i < band_count; ++i) {
            int bits = get_r_bits(i);
            uint32_t q = quantize_r(is_r_vals[i], bits);
            r_writer.write_bits(q, bits);
        }
        r_writer.flush();
        std::vector<uint8_t> r_bytes = r_writer.data();
        header.insert(header.end(), r_bytes.begin(), r_bytes.end());

        if (block_ver >= 12) {
            std::vector<bool> inv_bits(band_count, false);
            for (int i = is_start_band; i < band_count; ++i) inv_bits[i] = is_inv_flags[i];
            std::vector<uint8_t> inv_mask = pack_bits(inv_bits, is_start_band, band_count);
            header.insert(header.end(), inv_mask.begin(), inv_mask.end());
        }
    }

    payload_writer.flush();
    std::vector<uint8_t> payload = payload_writer.data();
    uint32_t payload_len = (uint32_t)payload.size();
    header.push_back(payload_len & 0xFF);
    header.push_back((payload_len>>8) & 0xFF);
    header.push_back((payload_len>>16) & 0xFF);
    header.push_back((payload_len>>24) & 0xFF);

    std::vector<uint8_t> out = header;
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void save_compressed(const std::vector<std::vector<uint8_t>>& blocks, const std::string& path,
                     uint32_t sr, int num_channels, int block_size, int block_count,
                     float target_kbps) {
    std::ofstream f(path, std::ios::binary);
    f.write("WHA1", 4);
    uint8_t version = 12;   
    f.write((char*)&version, 1);
    f.write((char*)&sr, 4);
    uint8_t ch = (uint8_t)num_channels;
    f.write((char*)&ch, 1);
    uint8_t lvl = (uint8_t)SETTINGS.wpt_level;
    f.write((char*)&lvl, 1);
    f.write((char*)&block_size, 4);
    uint32_t overlap = SETTINGS.overlap;
    f.write((char*)&overlap, 4);
    f.write((char*)&block_count, 4);
    float tk = target_kbps;
    f.write((char*)&tk, 4);
    for (auto &blk : blocks) {
        uint32_t len = blk.size();
        f.write((char*)&len, 4);
        f.write((char*)blk.data(), len);
    }
    f.close();
    std::cout << "Saved container v12 to " << path << " (" << blocks.size() << " blocks)\n";
}

std::tuple<std::vector<std::vector<uint8_t>>, uint32_t, int, int, int, float>
compress_audio(const std::string& input_path, const std::string& signal_type, float target_kbps_in) {
    (void)signal_type;
    uint32_t sr, channels;
    std::vector<float> raw = read_wav_f32(input_path, sr, channels);

    if (sr >= 44100) {
        SETTINGS.wpt_level = 4 + int(std::__lg(sr / 44100));
        SETTINGS.block_samples = 1024 * int(sr / 44100);
        SETTINGS.overlap = 3 * (1 << SETTINGS.wpt_level);
    }

    int num_channels = (int)channels;
    int N = SETTINGS.block_samples;
    int L = SETTINGS.overlap;
    int hop = N - L;
    int total_samples = (int)(raw.size() / num_channels);
    int total_blocks = (int)std::ceil((float)(total_samples - L) / hop);
    float target_kbps = (target_kbps_in > 0) ? target_kbps_in : SETTINGS.default_target_kbps;
    int target_bits_per_block = (int)(target_kbps * 1000.0f * hop / sr);
    if (target_bits_per_block < 256) target_bits_per_block = 256;

    int reservoir = 0;
    const int max_reservoir = target_bits_per_block * 0.3;

    PWPT wpt;
    std::vector<float> window(N, 1.0f);
    int edge = L;
    for (int i = 0; i < edge; ++i) {
        float t = (float)i / (edge - 1);
        float w = t * t * (3.0f - 2.0f * t);
        window[i] = w;
        window[N - 1 - i] = w;
    }

    std::vector<std::vector<uint8_t>> blocks_raw;
    for (int bidx = 0; bidx < total_blocks; ++bidx) {
        int effective_budget = target_bits_per_block + reservoir;
        if (effective_budget > target_bits_per_block + max_reservoir)
            effective_budget = target_bits_per_block + max_reservoir;

        int start = bidx * hop;
        int end = std::min(start + N, total_samples);
        std::vector<float> block(N * num_channels, 0.0f);
        for (int i = start; i < end; ++i) {
            for (int c = 0; c < num_channels; ++c) {
                block[(i - start) * num_channels + c] = raw[i * num_channels + c];
            }
        }
        for (int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < N; ++i) {
                block[i * num_channels + c] *= window[i];
            }
        }
        auto comp = compress_block_adaptive_joint(block, wpt, num_channels, SETTINGS.wpt_level,
                                                  effective_budget, target_kbps, sr, N, 13,
                                                  SETTINGS.enable_ms);
        int real_bits = (int)comp.size() * 8;
        reservoir += target_bits_per_block - real_bits;
        if (reservoir < 0) reservoir = 0;
        if (reservoir > max_reservoir) reservoir = max_reservoir;

        blocks_raw.push_back(comp);
        if (SETTINGS.verbose && bidx % 10 == 0) {
            std::cout << "[Block " << (bidx+1) << "/" << total_blocks << "] -> "
                      << comp.size() << " bytes, reservoir=" << reservoir << " bits\n";
        }
    }
    return {blocks_raw, sr, num_channels, N, hop, target_kbps};
}

int main(int argc, char** argv) {
    std::string input_file = "example.wav";
    std::string out_container = "example_restored.wha";
    if (argc < 2) {
        std::cerr << "Usage: ./encoder <input.wav> [bitrate_kbps]\n";
        return 1;
    }
    if (argc >= 2) {
        input_file = argv[1];
        out_container = argv[1];
        for (int64_t i = 0; i < 3; i++) out_container.pop_back();
        out_container += "wha";
    }
    if (argc >= 3) {
        SETTINGS.default_target_kbps = std::stof(argv[2]);
    }
    float target_kbps = SETTINGS.default_target_kbps;

    auto start = std::chrono::high_resolution_clock::now();
    auto comp = compress_audio(input_file, "music", target_kbps);
    auto blocks = std::get<0>(comp);
    uint32_t sr = std::get<1>(comp);
    int num_channels = std::get<2>(comp);
    int block_size = std::get<3>(comp);
    int hop = std::get<4>(comp);
    float tk = std::get<5>(comp);
    int block_count = (int)blocks.size();

    save_compressed(blocks, out_container, sr, num_channels, block_size, block_count, tk);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Compressed: " << block_count << " blocks, avg bytes/block="
              << (blocks.empty()?0:blocks[0].size()) << ", time=" << elapsed.count() << "s\n";
    return 0;
}