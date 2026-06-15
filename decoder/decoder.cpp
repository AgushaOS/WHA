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

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "wavelet.h"
#include "entropy_decoder.h"
#include "dequantize.h"
#include "joint_stereo.h"

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
    if (band_idx < 2) return 16;
    if (band_idx < 4) return 16;
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
    if (version != 12)
        throw std::runtime_error("Unsupported container version (only v12 with IS inversion is supported)");

    uint32_t sr = read_u32(f);
    uint8_t num_channels = read_u8(f);
    uint8_t wpt_level = read_u8(f);
    uint32_t block_size = read_u32(f);
    uint32_t overlap = read_u32(f);
    uint32_t block_count = read_u32(f);
    float target_kbps = read_f32(f);

    int L = overlap;
    int hop = block_size - L;
    int total_samples_estimate = hop * (block_count - 1) + block_size;
    std::vector<float> accum(total_samples_estimate * num_channels, 0.0f);
    std::vector<float> weight(total_samples_estimate * num_channels, 0.0f);

    std::vector<float> window(block_size, 1.0f);
    std::vector<float> window_sq(block_size, 1.0f);
    int edge = L;
    if (edge > 1) {
        for (int i = 0; i < edge; ++i) {
            float t = (float)i / (edge - 1);
            float w = t * t * (3.0f - 2.0f * t);
            window[i] = w;
            window[block_size - 1 - i] = w;
            window_sq[i] = w * w;
            window_sq[block_size - 1 - i] = w * w;
        }
    }

    std::vector<int> band_shapes = compute_band_shapes(block_size, wpt_level);
    int expected_band_count = static_cast<int>(band_shapes.size());

    PWPT wpt;
    std::vector<int32_t> residuals, quantized;
    std::vector<std::vector<float>> ch0_bands(expected_band_count);
    std::vector<std::vector<float>> ch1_bands(expected_band_count);
    std::vector<std::vector<float>> recon_chs;

    for (uint32_t bi = 0; bi < block_count; ++bi) {
        uint32_t block_len = read_u32(f);
        std::vector<uint8_t> blk(block_len);
        f.read((char*)blk.data(), block_len);
        if ((size_t)f.gcount() < block_len) throw std::runtime_error("Block truncated");

        size_t ptr = 0;
        auto need = [&](size_t n) { if (ptr + n > blk.size()) throw std::runtime_error("Block truncated"); };
        need(4);
        if (std::memcmp(blk.data() + ptr, "BLOC", 4) != 0) throw std::runtime_error("Block magic missing");
        ptr += 4;
        need(1);
        uint8_t block_ver = blk[ptr++];
        need(3);
        uint8_t band_count = blk[ptr++];
        uint8_t block_ch = blk[ptr++];
        uint8_t block_lvl = blk[ptr++];
        if (band_count != expected_band_count) throw std::runtime_error("Band count mismatch");

        need(1);
        uint8_t mode_bytes = blk[ptr++];
        std::vector<uint8_t> mode_ms(band_count, 0);
        if (mode_bytes > 0) {
            for (int i = 0; i < band_count; ++i) {
                uint8_t byte = blk[ptr + (i >> 3)];
                mode_ms[i] = (byte >> (i & 7)) & 1;
            }
            ptr += mode_bytes;
        }

        int mask_bytes = (band_count + 7) >> 3;
        std::vector<uint8_t> active0(band_count, 0);
        for (int i = 0; i < band_count; ++i) {
            uint8_t byte = blk[ptr + (i >> 3)];
            active0[i] = (byte >> (i & 7)) & 1;
        }
        ptr += mask_bytes;

        std::vector<uint8_t> active1;
        if (block_ch == 2) {
            active1.assign(band_count, 0);
            for (int i = 0; i < band_count; ++i) {
                uint8_t byte = blk[ptr + (i >> 3)];
                active1[i] = (byte >> (i & 7)) & 1;
            }
            ptr += mask_bytes;
        }

        std::vector<int> bits0(band_count, 0), bits1(band_count, 0);
        for (int i = 0; i < band_count; ++i) {
            if (active0[i]) bits0[i] = 1;  
            if (active1[i]) bits1[i] = 1;
        }

        std::vector<float> steps0(band_count, 0.0f), steps1(band_count, 0.0f);
        auto read_steps = [&](std::vector<float>& steps, const std::vector<int>& bits,
                              const std::vector<uint8_t>& active) {
            const float LOG_MIN = -6.0f, LOG_MAX = 6.0f;
            for (int i = 0; i < band_count; ++i) {
                if (!active[i]) continue;
                int sb = get_scale_bits(i);
                if (sb == 17) {
                    uint16_t h = blk[ptr] | (blk[ptr + 1] << 8);
                    ptr += 2;
                    steps[i] = half_to_float_fast(h);
                } else {
                    int idx;
                    if (sb == 8) {
                        idx = blk[ptr++];
                    } else {
                        idx = (blk[ptr] << 8) | blk[ptr + 1];
                        ptr += 2;
                    }
                    float log_step = LOG_MIN + idx * (LOG_MAX - LOG_MIN) / ((1 << sb) - 1);
                    steps[i] = exp10f(log_step);
                }
            }
        };
        read_steps(steps0, bits0, active0);
        if (block_ch == 2) read_steps(steps1, bits1, active1);

        std::vector<float> is_r(band_count, 0.5f);
        std::vector<bool> is_inv(band_count, false);   
        if (block_ver >= 9 && block_ch == 2) {
            int is_start = get_is_start_band(target_kbps);
            if (is_start < band_count) {
                int total_bits = 0;
                for (int i = is_start; i < band_count; ++i) total_bits += get_r_bits(i);
                int total_bytes = (total_bits + 7) / 8;
                if (ptr + total_bytes > blk.size()) throw std::runtime_error("Not enough data for r");
                BitReaderMSB r_reader(blk.data() + ptr, total_bytes);
                for (int i = is_start; i < band_count; ++i) {
                    int bits = get_r_bits(i);
                    uint32_t q = r_reader.read_bits(bits);
                    is_r[i] = dequantize_r(q, bits);
                }
                ptr += total_bytes;

                if (block_ver >= 12) {
                    int inv_bytes = (band_count - is_start + 7) / 8;
                    if (ptr + inv_bytes > blk.size()) throw std::runtime_error("Not enough data for inv mask");
                    std::vector<bool> inv_tmp = unpack_bits(blk.data() + ptr, inv_bytes, band_count - is_start);
                    for (int i = is_start; i < band_count; ++i) {
                        is_inv[i] = inv_tmp[i - is_start];
                    }
                    ptr += inv_bytes;
                }
            }
        }

        need(4);
        uint32_t payload_len;
        memcpy(&payload_len, &blk[ptr], 4);
        ptr += 4;
        BitReaderMSB payload_reader(blk.data() + ptr, payload_len);

        for (int ib = 0; ib < band_count; ++ib) {
            ch0_bands[ib].clear();
            ch1_bands[ib].clear();
        }

        for (int ib = 0; ib < band_count; ++ib) {
            if (active0[ib]) {
                int order = 0;
                if (!(block_ver >= 9 && ib >= get_is_start_band(target_kbps) && block_ch == 2)) {
                    order = (int)payload_reader.read_bits(2);
                }
                int k = payload_reader.read_bits(4);
                auto uvals = rice_decode(payload_reader, band_shapes[ib], k);
                residuals.resize(uvals.size());
                for (size_t i = 0; i < uvals.size(); ++i)
                    residuals[i] = zigzag_decode(uvals[i]);
                apply_inverse_lpc(residuals, order, quantized);
                dequantize_band(quantized, steps0[ib], ch0_bands[ib]);
            } else {
                ch0_bands[ib].assign(band_shapes[ib], 0.0f);
            }
        }

        if (block_ch == 2) {
            for (int ib = 0; ib < band_count; ++ib) {
                if (active1[ib]) {
                    int order = (int)payload_reader.read_bits(2);
                    int k = payload_reader.read_bits(4);
                    auto uvals = rice_decode(payload_reader, band_shapes[ib], k);
                    residuals.resize(uvals.size());
                    for (size_t i = 0; i < uvals.size(); ++i)
                        residuals[i] = zigzag_decode(uvals[i]);
                    apply_inverse_lpc(residuals, order, quantized);
                    dequantize_band(quantized, steps1[ib], ch1_bands[ib]);
                } else {
                    ch1_bands[ib].assign(band_shapes[ib], 0.0f);
                }
            }
        }

        if (block_ch == 2 && num_channels == 2) {
            int is_start = get_is_start_band(target_kbps);
            for (int ib = 0; ib < band_count; ++ib) {
                if (mode_ms[ib]) {
                    if (block_ver >= 9 && ib >= is_start && is_start < band_count) {
                        std::vector<float> left, right;
                        apply_is(ch0_bands[ib], is_r[ib], is_inv[ib], left, right);
                        ch0_bands[ib] = std::move(left);
                        ch1_bands[ib] = std::move(right);
                    } else {
                        std::vector<float> left_tmp, right_tmp;
                        ms_to_lr(ch0_bands[ib], ch1_bands[ib], left_tmp, right_tmp);
                        ch0_bands[ib] = std::move(left_tmp);
                        ch1_bands[ib] = std::move(right_tmp);
                    }
                }
            }
        }

        recon_chs.resize(block_ch);
        for (int ch = 0; ch < block_ch; ++ch) {
            auto& bands = (ch == 0) ? ch0_bands : ch1_bands;
            auto rec = wpt.iwpt(bands, sr, wpt_level);
            if (rec.size() < block_size) rec.resize(block_size, 0.0f);
            else rec.resize(block_size);
            recon_chs[ch] = std::move(rec);
        }

        for (int ch = 0; ch < num_channels; ++ch) {
            int src_ch = (ch < (int)recon_chs.size()) ? ch : 0;
            const float* src = recon_chs[src_ch].data();
            int global_offset = bi * hop * num_channels + ch;
            for (int i = 0; i < (int)block_size; ++i) {
                int idx = global_offset + i * num_channels;
                float w = window[i];
                accum[idx] += src[i] * w;
                weight[idx] += window_sq[i];
            }
        }
    }

    size_t total_frames = accum.size() / num_channels;
    std::vector<float> out_interleaved(accum.size());
    for (size_t i = 0; i < accum.size(); ++i) {
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
              << ", sr=" << sr << ", ch=" << (int)num_channels
              << ", overlap=" << L << ")\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: decoder <input.wha> <output.wav>\n";
        return 1;
    }
    try {
        decompress_wha_to_wav(argv[1], argv[2]);
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}