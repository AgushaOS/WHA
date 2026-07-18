#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "wavelet.h"
#include "entropy_decoder.h"
#include "dequantize.h"
#include "joint_stereo.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static std::array<float, 65536> half_to_float_table;
static bool half_table_initialized = false;
static void init_half_table() {
    if (half_table_initialized) return;
    for (uint32_t i = 0; i < 65536; ++i) {
        uint16_t h = static_cast<uint16_t>(i);
        uint32_t sign = (h >> 15) & 0x1;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) {
                half_to_float_table[i] = 0.0f;
                continue;
            }
            exp = 1;
        } else if (exp == 31) {
            exp = 255;
        } else {
            exp += 112;
        }
        uint32_t x = (sign << 31) | (exp << 23) | (mant << 13);
        float f;
        memcpy(&f, &x, 4);
        half_to_float_table[i] = f;
    }
    half_table_initialized = true;
}
static float half_to_float_fast(uint16_t h) {
    return half_to_float_table[h];
}
static int get_scale_bits(int band_idx) {
    if (band_idx < 1) return 12;
    return 8;
}
static uint8_t read_u8(std::ifstream &f) {
    uint8_t v = 0;
    f.read((char*)&v, 1);
    if (!f) throw std::runtime_error("Unexpected EOF (u8)");
    return v;
}
static uint32_t read_u32(std::ifstream &f) {
    uint32_t v = 0;
    f.read((char*)&v, 4);
    if (!f) throw std::runtime_error("Unexpected EOF (u32)");
    return v;
}
static float read_f32(std::ifstream &f) {
    float v = 0;
    f.read((char*)&v, 4);
    if (!f) throw std::runtime_error("Unexpected EOF (f32)");
    return v;
}
static std::vector<uint8_t> read_n(std::ifstream &f, size_t n) {
    std::vector<uint8_t> buf(n);
    f.read((char*)buf.data(), n);
    size_t got = (size_t)f.gcount();
    if (got < n) buf.resize(got);
    return buf;
}
static std::vector<int> compute_band_shapes(int block_size, int level) {
    std::vector<int> shapes = {block_size};
    for (int l = 0; l < level; ++l) {
        std::vector<int> new_shapes;
        new_shapes.reserve(shapes.size() * 2);
        for (int s : shapes) {
            new_shapes.push_back((s + 1) >> 1);
            new_shapes.push_back(s >> 1);
        }
        shapes = std::move(new_shapes);
    }
    return shapes;
}
static std::vector<bool> unpack_bits(const uint8_t* data, int bytes, int total_bits) {
    std::vector<bool> bits(total_bits, false);
    for (int i = 0; i < total_bits; ++i) {
        int byte_idx = i / 8;
        if (byte_idx >= bytes) break;
        bits[i] = (data[byte_idx] >> (i % 8)) & 1;
    }
    return bits;
}

void decompress_wha_to_wav(const std::string& in_wha, const std::string& out_wav) {
    init_half_table();
    std::ifstream f(in_wha, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + in_wha);
    auto magic = read_n(f, 4);
    if (magic.size() != 4 || std::string((char*)magic.data(), 4) != "WHA1")
        throw std::runtime_error("Not a WHA container");
    uint8_t version = read_u8(f);
    if (version != 17)
        throw std::runtime_error("Unsupported container version (only v17)");
    uint32_t sr = read_u32(f);
    uint8_t num_channels = read_u8(f);
    uint32_t block_count = read_u32(f);
    float target_kbps = read_f32(f);
    uint8_t block_format_version = read_u8(f);
    if (block_format_version != 17)
        throw std::runtime_error("Unsupported block format version (only v17)");
    
    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor = (per_channel_kbps <= 48.0f) ? 0.5f : 1.0f;
    
    int mode_bytes = (block_count + 7) / 8;
    std::vector<uint8_t> mode_packed = read_n(f, mode_bytes);
    std::vector<bool> block_modes = unpack_bits(mode_packed.data(), mode_bytes, block_count);
    
    int max_total_samples = block_count * 2048;
    std::vector<float> accum(max_total_samples * num_channels, 0.0f);
    std::vector<float> weight(max_total_samples * num_channels, 0.0f);
    
    PWPT wpt;
    int current_pos = 0;
    int prev_block_size = 1024;
    
    for (uint32_t bi = 0; bi < block_count; ++bi) {
        bool use_long_block = block_modes[bi];
        int level = use_long_block ? 5 : 4;
        int block_size = use_long_block ? 2048 : 1024;
        int right_overlap = use_long_block ? (int)(96 * overlap_factor) : (int)(48 * overlap_factor);
        int left_overlap = (prev_block_size == 2048) ? (int)(96 * overlap_factor) : (int)(48 * overlap_factor);
        int hop = block_size - right_overlap;
        
        std::vector<int> band_shapes = compute_band_shapes(block_size, level);
        int expected_band_count = static_cast<int>(band_shapes.size());
        int total_bands = 1 << level;
        
        std::vector<float> window(block_size, 1.0f);
        std::vector<float> window_sq(block_size, 1.0f);
        if (left_overlap > 1) {
            for (int i = 0; i < left_overlap; ++i) {
                float t = (float)i / left_overlap;
                float w = t * t * (3.0f - 2.0f * t);
                window[i] = w;
                window_sq[i] = w * w;
            }
        }
        if (right_overlap > 1) {
            for (int i = 0; i < right_overlap; ++i) {
                float t = (float)i / right_overlap;
                float w = t * t * (3.0f - 2.0f * t);
                window[block_size - 1 - i] = w;
                window_sq[block_size - 1 - i] = w * w;
            }
        }
        
        uint32_t block_len = read_u32(f);
        std::vector<uint8_t> blk(block_len);
        f.read((char*)blk.data(), block_len);
        if ((size_t)f.gcount() < block_len) throw std::runtime_error("Block truncated");
        size_t ptr = 0;
        auto need = [&](size_t n) { if (ptr + n > blk.size()) throw std::runtime_error("Block truncated"); };
        int mode_bytes = (expected_band_count + 7) / 8;
        std::vector<uint8_t> mode_ms(expected_band_count, 0);
        for (int i = 0; i < expected_band_count; ++i) {
            uint8_t byte = blk[ptr + (i >> 3)];
            mode_ms[i] = (byte >> (i & 7)) & 1;
        }
        ptr += mode_bytes;
        int mask_bytes = (expected_band_count + 7) >> 3;
        std::vector<uint8_t> active0(expected_band_count, 0);
        for (int i = 0; i < expected_band_count; ++i) {
            uint8_t byte = blk[ptr + (i >> 3)];
            active0[i] = (byte >> (i & 7)) & 1;
        }
        ptr += mask_bytes;
        std::vector<uint8_t> active1;
        bool stereo = (num_channels == 2);
        if (stereo) {
            active1.assign(expected_band_count, 0);
            for (int i = 0; i < expected_band_count; ++i) {
                uint8_t byte = blk[ptr + (i >> 3)];
                active1[i] = (byte >> (i & 7)) & 1;
            }
            ptr += mask_bytes;
        }
        need(1);
        uint8_t k_scale_byte = blk[ptr++];
        int k_scale0 = k_scale_byte & 0x07;
        int k_scale1 = stereo ? ((k_scale_byte >> 3) & 0x07) : 0;
        std::vector<std::array<float, 4>> is_r(expected_band_count);
        for (auto& a : is_r) a.fill(0.5f);
        std::vector<bool> is_inv(expected_band_count, false);
        std::vector<bool> is_use_segmented(expected_band_count, false);
        if (block_format_version >= 17 && stereo && target_kbps < 510.0f) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            if (is_start < expected_band_count) {
                int total_bits = 0;
                for (int i = is_start; i < expected_band_count; ++i) {
                    int bits = get_r_bits(i);
                    int segmented_threshold = 6 * total_bands / 16;
                    if (i < segmented_threshold) total_bits += bits * 4;
                    else total_bits += bits;
                }
                int total_bytes = (total_bits + 7) / 8;
                if (ptr + total_bytes > blk.size()) throw std::runtime_error("Not enough data for r");
                BitReaderMSB r_reader(blk.data() + ptr, total_bytes);
                for (int i = is_start; i < expected_band_count; ++i) {
                    int bits = get_r_bits(i);
                    int segmented_threshold = 6 * total_bands / 16;
                    if (i < segmented_threshold) {
                        is_use_segmented[i] = true;
                        for (int s = 0; s < 4; ++s) {
                            uint32_t q = r_reader.read_bits(bits);
                            is_r[i][s] = dequantize_r(q, bits);
                        }
                    } else {
                        is_use_segmented[i] = false;
                        uint32_t q = r_reader.read_bits(bits);
                        float r_val = dequantize_r(q, bits);
                        is_r[i].fill(r_val);
                    }
                }
                ptr += total_bytes;
                int inv_bytes = (expected_band_count - is_start + 7) / 8;
                if (ptr + inv_bytes > blk.size()) throw std::runtime_error("Not enough data for inv mask");
                std::vector<bool> inv_tmp = unpack_bits(blk.data() + ptr, inv_bytes, expected_band_count - is_start);
                for (int i = is_start; i < expected_band_count; ++i) {
                    is_inv[i] = inv_tmp[i - is_start];
                }
                ptr += inv_bytes;
            }
        }
        need(2);
        uint16_t payload_len;
        memcpy(&payload_len, &blk[ptr], 2);
        ptr += 2;
        BitReaderMSB payload_reader(blk.data() + ptr, payload_len);
        const float LOG_MIN = -6.0f;
        const float LOG_MAX =  0.0f;
        std::vector<float> steps0(expected_band_count, 0.0f);
        std::vector<float> steps1(expected_band_count, 0.0f);
        auto decode_scales = [&](std::vector<float>& steps, const std::vector<uint8_t>& active, int k_scale) {
            for (int i = 0; i < expected_band_count; ++i) {
                if (!active[i]) continue;
                auto idx_vec = rice_decode(payload_reader, 1, k_scale);
                if (idx_vec.empty()) throw std::runtime_error("Scale index missing");
                uint32_t idx = idx_vec[0];
                int sb = get_scale_bits(i);
                int max_idx = (1 << sb) - 1;
                float log_step = LOG_MIN + idx * (LOG_MAX - LOG_MIN) / max_idx;
                steps[i] = exp10f(log_step);
            }
        };
        decode_scales(steps0, active0, k_scale0);
        if (stereo) decode_scales(steps1, active1, k_scale1);
        std::vector<std::vector<int32_t>> quant0(expected_band_count);
        for (int i = 0; i < expected_band_count; ++i) {
            if (!active0[i]) continue;
            int k_sub = (int)payload_reader.read_bits(3);
            auto uvals = rice_decode(payload_reader, band_shapes[i], k_sub);
            quant0[i].resize(uvals.size());
            for (size_t j = 0; j < uvals.size(); ++j)
                quant0[i][j] = zigzag_decode(uvals[j]);
        }
        std::vector<std::vector<int32_t>> quant1;
        if (stereo) {
            quant1.resize(expected_band_count);
            for (int i = 0; i < expected_band_count; ++i) {
                if (!active1[i]) continue;
                int k_sub = (int)payload_reader.read_bits(3);
                auto uvals = rice_decode(payload_reader, band_shapes[i], k_sub);
                quant1[i].resize(uvals.size());
                for (size_t j = 0; j < uvals.size(); ++j)
                    quant1[i][j] = zigzag_decode(uvals[j]);
            }
        }
        
        std::vector<std::vector<float>> ch0_bands(expected_band_count);
        std::vector<std::vector<float>> ch1_bands(expected_band_count);
        
        for (int i = 0; i < expected_band_count; ++i) {
            if (active0[i]) {
                dequantize_band(quant0[i], steps0[i], ch0_bands[i]);
            } else {
                ch0_bands[i].assign(band_shapes[i], 0.0f);
            }
            if (stereo && active1[i]) {
                dequantize_band(quant1[i], steps1[i], ch1_bands[i]);
            } else if (stereo) {
                ch1_bands[i].assign(band_shapes[i], 0.0f);
            }
        }
        if (stereo && num_channels == 2) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            for (int i = 0; i < expected_band_count; ++i) {
                if (mode_ms[i]) {
                    if (block_format_version >= 17 && i >= is_start && is_start < expected_band_count) {
                        std::vector<float> left, right;
                        apply_is(ch0_bands[i], is_r[i], is_inv[i], is_use_segmented[i], left, right);
                        ch0_bands[i] = std::move(left);
                        ch1_bands[i] = std::move(right);
                    } else {
                        std::vector<float> left_tmp, right_tmp;
                        ms_to_lr(ch0_bands[i], ch1_bands[i], left_tmp, right_tmp);
                        ch0_bands[i] = std::move(left_tmp);
                        ch1_bands[i] = std::move(right_tmp);
                    }
                }
            }
        }
        
        std::vector<std::vector<float>> recon_chs(num_channels);
        for (int ch = 0; ch < num_channels; ++ch) {
            auto& bands = (ch == 0) ? ch0_bands : ch1_bands;
            auto rec = wpt.iwpt(bands, sr, level);
            if (rec.size() < block_size) rec.resize(block_size, 0.0f);
            else rec.resize(block_size);
            recon_chs[ch] = std::move(rec);
        }
        
        for (int ch = 0; ch < num_channels; ++ch) {
            int src_ch = (ch < (int)recon_chs.size()) ? ch : 0;
            const float* src = recon_chs[src_ch].data();
            int global_offset = current_pos * num_channels + ch;
            for (int i = 0; i < block_size; ++i) {
                int idx = global_offset + i * num_channels;
                if (idx < (int)accum.size()) {
                    float w = window[i];
                    accum[idx] += src[i] * w;
                    weight[idx] += window_sq[i];
                }
            }
        }
        
        current_pos += hop;
        prev_block_size = block_size;
    }
    
    size_t total_frames = current_pos;
    std::vector<float> out_interleaved(total_frames * num_channels);
    for (size_t i = 0; i < total_frames * num_channels; ++i) {
        if (weight[i] > 1e-9f) out_interleaved[i] = accum[i] / weight[i];
        else out_interleaved[i] = 0.0f;
    }
    drwav_data_format fmt;
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = num_channels;
    fmt.sampleRate = sr;
    fmt.bitsPerSample = 32;
    drwav wav;
    if (!drwav_init_file_write(&wav, out_wav.c_str(), &fmt, nullptr))
        throw std::runtime_error("dr_wav failed to init");
    drwav_uint64 written = drwav_write_pcm_frames(&wav, total_frames, out_interleaved.data());
    drwav_uninit(&wav);
    if (written != total_frames) throw std::runtime_error("dr_wav wrote fewer frames");
    std::cout << "Decompressed: " << out_wav << " (frames=" << written
              << ", sr=" << sr << ", ch=" << (int)num_channels << ")\n";
}

#include <sys/resource.h>
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: decoder <input.wha> <output.wav>\n";
        return 1;
    }
    try {
        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);
        decompress_wha_to_wav(argv[1], argv[2]);
        getrusage(RUSAGE_SELF, &usage_after);
        double user_time_sec = (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
                               (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;
        drwav wav_info;
        if (drwav_init_file(&wav_info, argv[2], nullptr)) {
            double duration = (double)wav_info.totalPCMFrameCount / (double)wav_info.sampleRate;
            double speed = (user_time_sec > 0.0) ? (duration / user_time_sec) : 0.0;
            std::cout << "Decoding speed: " << speed << "x realtime ("
                      << user_time_sec << "s user time)\n";
            drwav_uninit(&wav_info);
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}