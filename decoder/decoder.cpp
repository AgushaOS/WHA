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
#include <memory>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "wavelet.h"
#include "entropy_decoder.h"
#include "dequantize.h"
#include "joint_stereo.h"
#include "postfilter.h"

#include "pred_state.h"
#include "codec_utils.h"
#include "container_io.h"
#include "sbr_decode.h"
#include "overlap_add.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void decompress_wha_to_wav(const std::string& in_wha,
                           const std::string& out_wav,
                           size_t write_batch_frames = 8192)
{
    std::ifstream f(in_wha, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + in_wha);

    auto magic = read_n(f, 4);
    if (magic.size() != 4 || std::string((char*)magic.data(), 4) != "WHA1")
        throw std::runtime_error("Not a WHA container");

    uint8_t version = read_u8(f);
    if (version != 18 && version != 19 && version != 20)
        throw std::runtime_error("Unsupported container version (only v18/v19/v20)");

    uint32_t sr           = read_u32(f);
    uint8_t  num_channels = read_u8(f);
    uint32_t block_count  = read_u32(f);
    float    target_kbps  = read_f32(f);
    uint8_t  block_format_version = read_u8(f);

    if (block_format_version != 18 && block_format_version != 19 && block_format_version != 20)
        throw std::runtime_error("Unsupported block format version (only v18/v19/v20)");

    bool pred_enabled = (block_format_version >= 19);

    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor   = (per_channel_kbps <= 0.0f) ? 0.0f : 1.0f;

    int mode_bytes = (block_count + 7) / 8;
    std::vector<uint8_t> mode_packed = read_n(f, mode_bytes);
    std::vector<bool> block_modes = unpack_bits(mode_packed.data(), mode_bytes, block_count);

    drwav_data_format fmt;
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = num_channels;
    fmt.sampleRate    = sr;
    fmt.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, out_wav.c_str(), &fmt, nullptr))
        throw std::runtime_error("dr_wav failed to init");

    PWPT      wpt;
    ModeState long_state;
    ModeState short_state;
    bool      stereo = (num_channels == 2);

    OverlapAddState ola;
    ola.init(num_channels, write_batch_frames);

    std::vector<float> write_buf;
    write_buf.reserve(write_batch_frames * num_channels);

    std::unique_ptr<SpectralFloorPostProcessor> postproc;
    if (shouldApplySpectralFloor((int)num_channels, target_kbps)) {
        SpectralFloorPostFX::Config pp_cfg;
        postproc = std::make_unique<SpectralFloorPostProcessor>((int)num_channels, pp_cfg);
    }

    auto flush_write_buffer = [&]() {
        if (write_buf.empty()) return;
        size_t frames = write_buf.size() / num_channels;
        if (frames > 0) {
            if (postproc) {
                int n_out = postproc->process(write_buf.data(), (int)frames);
                if (n_out > 0)
                    drwav_write_pcm_frames(&wav, (uint64_t)n_out, postproc->out_data());
            } else {
                drwav_write_pcm_frames(&wav, frames, write_buf.data());
            }
        }
        write_buf.clear();
    };

    int prev_block_size = 1024;

    for (uint32_t bi = 0; bi < block_count; ++bi) {
        bool use_long_block = block_modes[bi];
        int  level          = use_long_block ? 5 : 4;
        int  block_size     = use_long_block ? 2048 : 1024;
        int  right_overlap  = use_long_block
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);
        int  left_overlap   = (prev_block_size == 2048)
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);

        if (bi == 0)
            left_overlap = 0;

        int  hop         = block_size - right_overlap;
        int  total_bands = 1 << level;

        std::vector<int> band_shapes = compute_band_shapes(block_size, level);
        int expected_band_count = total_bands;

        ModeState& pred_state = use_long_block ? long_state : short_state;
        pred_state.ensure(expected_band_count);

        bool can_state_pred = false;
        if (pred_enabled && pred_state.ready && bi > pred_state.last_bi) {
            uint32_t age = bi - pred_state.last_bi;
            if (age <= SAME_MODE_PRED_MAX_AGE)
                can_state_pred = true;
        }

        std::vector<float> window, window_sq;
        make_window(window, window_sq, block_size, left_overlap, right_overlap);

        uint32_t block_len = read_u32(f);
        std::vector<uint8_t> blk(block_len);
        f.read((char*)blk.data(), block_len);
        if ((size_t)f.gcount() < block_len)
            throw std::runtime_error("Block truncated");

        size_t ptr = 0;
        auto need = [&](size_t n) {
            if (ptr + n > blk.size())
                throw std::runtime_error("Block truncated");
        };

        int mode_bytes_loc = (expected_band_count + 7) / 8;
        std::vector<uint8_t> mode_ms(expected_band_count, 0);
        for (int i = 0; i < expected_band_count; ++i) {
            uint8_t byte = blk[ptr + (i >> 3)];
            mode_ms[i] = (byte >> (i & 7)) & 1;
        }
        ptr += mode_bytes_loc;

        int mask_bytes = (expected_band_count + 7) >> 3;
        std::vector<uint8_t> active0(expected_band_count, 0);
        for (int i = 0; i < expected_band_count; ++i) {
            uint8_t byte = blk[ptr + (i >> 3)];
            active0[i] = (byte >> (i & 7)) & 1;
        }
        ptr += mask_bytes;
        std::vector<uint8_t> active1;
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
        int  k_scale0       = k_scale_byte & 0x07;
        int  k_scale1       = stereo ? ((k_scale_byte >> 3) & 0x07) : 0;
        bool block_is_full  = (k_scale_byte & 0x80) != 0;
        bool use_scale_pred = pred_enabled && can_state_pred && ((k_scale_byte & 0x40) != 0);
        int  sbr_end        = block_is_full ? (3 * expected_band_count / 4) : 0;

        need(2);
        uint16_t payload_len;
        memcpy(&payload_len, &blk[ptr], 2);
        ptr += 2;
        if (ptr + payload_len > blk.size())
            throw std::runtime_error("Payload truncated");

        BitReaderMSB payload_reader(blk.data() + ptr, payload_len);
        ptr += payload_len;

        std::vector<std::array<float, 4>> is_r(expected_band_count);
        for (auto& a : is_r) a.fill(0.5f);
        std::vector<bool> is_inv(expected_band_count, false);
        std::vector<bool> is_use_segmented(expected_band_count, false);
        bool is_used = false;

        if (block_format_version >= 18 && stereo && target_kbps < 510.0f) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            if (is_start < expected_band_count) {
                int segmented_threshold = 6 * total_bands / 16;

                if (block_format_version >= 20) {
                    std::vector<int> is_indices;
                    for (int i = is_start; i < expected_band_count; ++i)
                        if (mode_ms[i]) is_indices.push_back(i);

                    int total_bits = 0;
                    for (int idx : is_indices) {
                        int bits = get_r_bits(idx);
                        total_bits += (idx < segmented_threshold) ? bits * 4 : bits;
                    }
                    int total_bytes = (total_bits + 7) / 8;
                    int inv_bytes   = ((int)is_indices.size() + 7) / 8;

                    if (ptr + total_bytes + inv_bytes <= blk.size()) {
                        if (!is_indices.empty()) {
                            is_used = true;
                            BitReaderMSB r_reader(blk.data() + ptr, total_bytes);
                            for (int idx : is_indices) {
                                int bits = get_r_bits(idx);
                                if (idx < segmented_threshold) {
                                    is_use_segmented[idx] = true;
                                    for (int s = 0; s < 4; ++s) {
                                        uint32_t q = r_reader.read_bits(bits);
                                        is_r[idx][s] = dequantize_r(q, bits);
                                    }
                                } else {
                                    is_use_segmented[idx] = false;
                                    uint32_t q = r_reader.read_bits(bits);
                                    is_r[idx].fill(dequantize_r(q, bits));
                                }
                            }
                            ptr += total_bytes;
                            auto inv_tmp = unpack_bits(blk.data() + ptr, inv_bytes,
                                                       (int)is_indices.size());
                            for (size_t j = 0; j < is_indices.size(); ++j)
                                is_inv[is_indices[j]] = inv_tmp[j];
                            ptr += inv_bytes;
                        }
                    }
                } else {
                    int total_bits = 0;
                    for (int i = is_start; i < expected_band_count; ++i) {
                        int bits = get_r_bits(i);
                        total_bits += (i < segmented_threshold) ? bits * 4 : bits;
                    }
                    int total_bytes = (total_bits + 7) / 8;
                    int inv_bytes   = (expected_band_count - is_start + 7) / 8;

                    if (ptr + total_bytes + inv_bytes <= blk.size()) {
                        is_used = true;
                        BitReaderMSB r_reader(blk.data() + ptr, total_bytes);
                        for (int i = is_start; i < expected_band_count; ++i) {
                            int bits = get_r_bits(i);
                            if (i < segmented_threshold) {
                                is_use_segmented[i] = true;
                                for (int s = 0; s < 4; ++s) {
                                    uint32_t q = r_reader.read_bits(bits);
                                    is_r[i][s] = dequantize_r(q, bits);
                                }
                            } else {
                                is_use_segmented[i] = false;
                                uint32_t q = r_reader.read_bits(bits);
                                is_r[i].fill(dequantize_r(q, bits));
                            }
                        }
                        ptr += total_bytes;
                        auto inv_tmp = unpack_bits(blk.data() + ptr, inv_bytes,
                                                   expected_band_count - is_start);
                        for (int i = is_start; i < expected_band_count; ++i)
                            is_inv[i] = inv_tmp[i - is_start];
                        ptr += inv_bytes;
                    }
                }
            }
        }

        SBRDecodeResult sbr = sbr_decode(
            blk.data(), ptr, blk.size(),
            active0, sbr_end, expected_band_count,
            block_format_version, can_state_pred, pred_state);

        std::vector<float> steps0(expected_band_count, 0.0f);
        std::vector<float> steps1(expected_band_count, 0.0f);

        auto decode_scales = [&](std::vector<float>& steps,
                                 const std::vector<uint8_t>& active,
                                 int k_scale,
                                 std::vector<uint32_t>& prev_scale,
                                 const std::vector<uint8_t>& prev_active)
        {
            for (int i = 0; i < expected_band_count; ++i) {
                if (!active[i]) continue;
                auto idx_vec = rice_decode(payload_reader, 1, k_scale);
                if (idx_vec.empty())
                    throw std::runtime_error("Scale index missing");
                uint32_t val = idx_vec[0];

                int sb      = get_scale_bits(i);
                int max_idx = (1 << sb) - 1;
                uint32_t idx;

                if (use_scale_pred &&
                    i < (int)prev_active.size() && prev_active[i])
                {
                    int32_t d = zigzag_decode(val);
                    int64_t v = (int64_t)prev_scale[i] + d;
                    if (v < 0) v = 0;
                    if (v > max_idx) v = max_idx;
                    idx = (uint32_t)v;
                } else {
                    idx = val;
                    if (idx > (uint32_t)max_idx) idx = max_idx;
                }

                steps[i] = idx_to_step(idx, sb);
                if (i < (int)prev_scale.size())
                    prev_scale[i] = idx;
            }
        };

        decode_scales(steps0, active0, k_scale0,
                      pred_state.prev_scale0, pred_state.prev_active0);
        if (stereo)
            decode_scales(steps1, active1, k_scale1,
                          pred_state.prev_scale1, pred_state.prev_active1);

        std::vector<std::vector<int32_t>> quant0(expected_band_count);
        for (int i = 0; i < expected_band_count; ++i) {
            if (!active0[i]) continue;
            int k_sub  = (int)payload_reader.read_bits(3);
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
                int k_sub  = (int)payload_reader.read_bits(3);
                auto uvals = rice_decode(payload_reader, band_shapes[i], k_sub);
                quant1[i].resize(uvals.size());
                for (size_t j = 0; j < uvals.size(); ++j)
                    quant1[i][j] = zigzag_decode(uvals[j]);
            }
        }

        std::vector<std::vector<float>> ch0_bands(expected_band_count);
        std::vector<std::vector<float>> ch1_bands(expected_band_count);

        for (int i = 0; i < expected_band_count; ++i) {
            if (active0[i])
                dequantize_band(quant0[i], steps0[i], ch0_bands[i]);
            else
                ch0_bands[i].assign(band_shapes[i], 0.0f);

            if (stereo) {
                if (active1[i])
                    dequantize_band(quant1[i], steps1[i], ch1_bands[i]);
                else
                    ch1_bands[i].assign(band_shapes[i], 0.0f);
            }
        }

        sbr_synthesize(ch0_bands, active0, band_shapes, sbr, sbr_end, bi);

        if (stereo) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            for (int i = 0; i < expected_band_count; ++i) {
                if (mode_ms[i]) {
                    if (block_format_version >= 18 && is_used &&
                        i >= is_start && is_start < expected_band_count)
                    {
                        std::vector<float> left, right;
                        apply_is(ch0_bands[i], is_r[i], is_inv[i],
                                 is_use_segmented[i], left, right);
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
            auto rec = wpt.iwpt(bands, sr, level, target_kbps, num_channels);
            if (rec.size() < (size_t)block_size)
                rec.resize(block_size, 0.0f);
            else
                rec.resize(block_size);
            recon_chs[ch] = std::move(rec);
        }

        int local_start = (ola.current_pos - ola.output_pos) + (int)ola.head_offset;
        overlap_add_block(ola, recon_chs, block_size, window, window_sq, local_start);
        ola.current_pos += hop;

        extract_ready_samples(ola, write_buf);

        if (write_buf.size() >= write_batch_frames * num_channels)
            flush_write_buffer();

        compact_ola_buffer(ola);

        if (pred_enabled) {
            pred_state.prev_active0.assign(expected_band_count, 0);
            for (int i = 0; i < expected_band_count; ++i)
                if (active0[i]) pred_state.prev_active0[i] = 1;

            if (stereo) {
                pred_state.prev_active1.assign(expected_band_count, 0);
                for (int i = 0; i < expected_band_count; ++i)
                    if (active1[i]) pred_state.prev_active1[i] = 1;
            } else {
                pred_state.prev_active1.assign(expected_band_count, 0);
            }

            pred_state.prev_sbr_valid.assign(expected_band_count, 0);
            pred_state.prev_sbr_noise.assign(expected_band_count, 0);
            for (int idx : sbr.indices) {
                if (idx >= 0 && idx < expected_band_count) {
                    pred_state.prev_sbr_valid[idx] = 1;
                    pred_state.prev_sbr_noise[idx] = sbr.noise_flag[idx] ? 1 : 0;
                    pred_state.prev_sbr_rms[idx]   = sbr.rms_idx[idx];
                }
            }
            pred_state.ready   = true;
            pred_state.last_bi = bi;
        }

        prev_block_size = block_size;
    }

    extract_remaining(ola, write_buf);
    flush_write_buffer();

    if (postproc) {
        int n_out = postproc->flush();
        if (n_out > 0)
            drwav_write_pcm_frames(&wav, (uint64_t)n_out, postproc->out_data());
    }

    drwav_uninit(&wav);
    std::cout << "Decompressed: " << out_wav
              << " (frames=" << ola.output_pos
              << ", sr=" << sr
              << ", ch=" << (int)num_channels << ")" << std::endl;
}

#include <sys/resource.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: decoder <input.wha> <output.wav> [buffer_size_in_samples]"
                  << std::endl;
        return 1;
    }

    size_t buffer_size = 8192;
    if (argc >= 4) {
        try {
            buffer_size = std::stoul(argv[3]);
            if (buffer_size == 0) {
                std::cerr << "Buffer size must be > 0, using default 8192" << std::endl;
                buffer_size = 8192;
            }
        } catch (...) {
            std::cerr << "Invalid buffer size, using default 8192" << std::endl;
            buffer_size = 8192;
        }
    }

    try {
        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);

        decompress_wha_to_wav(argv[1], argv[2], buffer_size);

        getrusage(RUSAGE_SELF, &usage_after);
        double user_time_sec =
            (usage_after.ru_utime.tv_sec  - usage_before.ru_utime.tv_sec) +
            (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;

        drwav wav_info;
        if (drwav_init_file(&wav_info, argv[2], nullptr)) {
            double duration =
                (double)wav_info.totalPCMFrameCount / (double)wav_info.sampleRate;
            double speed = (user_time_sec > 0.0) ? (duration / user_time_sec) : 0.0;
            std::cout << "Decoding speed: " << speed << "x realtime ("
                      << user_time_sec << "s user time)" << std::endl;
            drwav_uninit(&wav_info);
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}
