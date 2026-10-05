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
#include <sys/resource.h>
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
#include "pow_filter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float SBR_W0 = 1.0f;
float SBR_W1 = 9.0f;

void decompress_wha_to_wav(
    const std::string& in_wha,
    const std::string& out_wav,
    size_t write_batch_frames = 8192)
{
    std::ifstream f(in_wha, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + in_wha);

    auto magic = read_n(f, 4);
    if (magic.size() != 4 || std::string((char*)magic.data(), 4) != "WHA1")
        throw std::runtime_error("Not a WHA container");

    uint8_t version = read_u8(f);
    if (version < 18 || version > 24)
        throw std::runtime_error("Unsupported container version (only v18-v24)");

    uint32_t sr = read_u32(f);
    uint8_t num_channels = read_u8(f);
    uint32_t block_count = read_u32(f);
    float target_kbps = read_f32(f);
    uint8_t block_format_version = read_u8(f);

    if (block_format_version < 18 || block_format_version > 24)
        throw std::runtime_error("Unsupported block format version (only v18-v24)");

    uint32_t total_frames = 0;
    if (version >= 21) total_frames = read_u32(f);

    uint8_t flags = 0;
    if (version >= 24) {
        flags = read_u8(f);
    }
    
    bool speed_mode = (flags & 0x01) != 0;
    if (speed_mode) {
        std::cout << ">>> Container indicates SPEED MODE: Post-processing will be skipped automatically <<<\n";
    }

    bool pred_enabled = (block_format_version >= 19);

    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor = (per_channel_kbps <= 0.0f) ? 0.0f : 1.0f;

    int mode_bytes = (block_count + 7) / 8;
    std::vector<uint8_t> mode_packed = read_n(f, mode_bytes);
    std::vector<bool> block_modes = unpack_bits(
        mode_packed.data(), mode_bytes, block_count);

    drwav_data_format fmt;
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = num_channels;
    fmt.sampleRate = sr;
    fmt.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, out_wav.c_str(), &fmt, nullptr))
        throw std::runtime_error("dr_wav failed to init");

    PWPT wpt;
    ModeState long_state, short_state;
    bool stereo = (num_channels == 2);

    std::unique_ptr<PowCodec> pow_dec;
    if (!speed_mode) {
        pow_dec = std::make_unique<PowCodec>((int)num_channels, 0.85f, false, target_kbps / float(num_channels));
    }

    OverlapAddState ola;
    ola.init(num_channels, write_batch_frames);

    std::vector<float> write_buf, decoded;
    write_buf.reserve(write_batch_frames * num_channels);
    decoded.reserve(write_batch_frames * num_channels);

    std::unique_ptr<SpectralFloorPostProcessor> postproc;
    if (!speed_mode && shouldApplySpectralFloor((int)num_channels, target_kbps)) {
        SpectralFloorPostFX::Config pp_cfg;
        postproc = std::make_unique<SpectralFloorPostProcessor>(
            (int)num_channels, pp_cfg);
    }

    std::vector<int> band_shapes_long = compute_band_shapes(2048, 5);
    std::vector<int> band_shapes_short = compute_band_shapes(1024, 4);

    struct WindowCache {
        std::vector<float> window, window_sq;
        int bs = 0;
        int lo = -1;
        int ro = -1;
    };
    WindowCache win_long, win_short;

    std::vector<uint8_t> blk, mode_ms, active0, active1;
    std::vector<std::array<float, 4>> is_r;
    std::vector<std::array<bool, 4>> is_inv;
    std::vector<bool> is_use_segmented;
    std::vector<float> steps0, steps1;
    std::vector<std::vector<int32_t>> quant0, quant1;
    std::vector<std::vector<float>> ch0_bands, ch1_bands;

    quant0.reserve(32);
    quant1.reserve(32);
    ch0_bands.reserve(32);
    ch1_bands.reserve(32);

    std::vector<std::vector<float>> recon_chs(num_channels);
    uint64_t frames_written = 0;

    auto write_samples = [&](const std::vector<float>& samples) {
        if (samples.empty()) return;
        size_t frames = samples.size() / num_channels;
        if (frames == 0) return;

        if (total_frames > 0 && frames_written + frames > total_frames) {
            frames = total_frames - frames_written;
            if (frames == 0) return;
        }

        const float* data_to_write = samples.data();
        size_t frames_to_write = frames;

        if (postproc) {
            int n_out = postproc->process(samples.data(), (int)frames);
            if (n_out > 0) {
                data_to_write = postproc->out_data();
                frames_to_write = n_out;
            } else {
                return;
            }
        }

        if (total_frames > 0 && frames_written + frames_to_write > total_frames) {
            frames_to_write = total_frames - frames_written;
            if (frames_to_write == 0) return;
        }

        drwav_write_pcm_frames(&wav, (uint64_t)frames_to_write, data_to_write);
        frames_written += frames_to_write;
    };

    auto flush_write_buffer = [&]() {
        if (write_buf.empty()) return;
        
        if (!speed_mode && pow_dec) {
            decoded.clear();
            pow_dec->process(write_buf.data(), write_buf.size() / num_channels, decoded);
            write_samples(decoded);
        } else {
            write_samples(write_buf);
        }
        write_buf.clear();
    };

    int prev_block_size = 1024;

    for (uint32_t bi = 0; bi < block_count; ++bi) {
        bool use_long_block = block_modes[bi];
        int level = use_long_block ? 5 : 4;
        int block_size = use_long_block ? 2048 : 1024;
        int right_overlap = use_long_block
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);
        int left_overlap = (prev_block_size == 2048)
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);
        if (bi == 0) left_overlap = 0;
        int hop = block_size - right_overlap;

        int total_bands = 1 << level;
        const std::vector<int>& band_shapes = use_long_block
            ? band_shapes_long
            : band_shapes_short;

        int expected_band_count = total_bands;
        ModeState& pred_state = use_long_block ? long_state : short_state;
        pred_state.ensure(expected_band_count);

        bool can_state_pred = false;
        if (pred_enabled && pred_state.ready && bi > pred_state.last_bi) {
            uint32_t age = bi - pred_state.last_bi;
            if (age <= SAME_MODE_PRED_MAX_AGE)
                can_state_pred = true;
        }

        WindowCache& wc = use_long_block ? win_long : win_short;
        if (wc.bs != block_size || wc.lo != left_overlap || wc.ro != right_overlap) {
            make_window(wc.window, wc.window_sq, block_size, left_overlap, right_overlap);
            wc.bs = block_size;
            wc.lo = left_overlap;
            wc.ro = right_overlap;
        }

        const std::vector<float>& window = wc.window;
        const std::vector<float>& window_sq = wc.window_sq;

        size_t block_len;
        if (block_format_version >= 23) {
            uint16_t len16;
            f.read((char*)&len16, 2);
            block_len = len16;
        } else {
            block_len = read_u32(f);
        }

        blk.resize(block_len);
        f.read((char*)blk.data(), block_len);
        if ((size_t)f.gcount() < block_len)
            throw std::runtime_error("Block truncated");

        size_t ptr = 0;
        auto need = [&](size_t n) {
            if (ptr + n > blk.size())
                throw std::runtime_error("Block truncated");
        };

        if (block_format_version >= 24) {
            mode_ms.resize(expected_band_count);
            active0.resize(expected_band_count);
            if (stereo) active1.resize(expected_band_count);
        } else {
            mode_ms.resize(expected_band_count);
            int mode_bytes_loc = (expected_band_count + 7) / 8;
            for (int i = 0; i < expected_band_count; ++i) {
                uint8_t byte = blk[ptr + (i >> 3)];
                mode_ms[i] = (byte >> (i & 7)) & 1;
            }
            ptr += mode_bytes_loc;

            int mask_bytes = (expected_band_count + 7) >> 3;
            active0.resize(expected_band_count);
            for (int i = 0; i < expected_band_count; ++i) {
                uint8_t byte = blk[ptr + (i >> 3)];
                active0[i] = (byte >> (i & 7)) & 1;
            }
            ptr += mask_bytes;

            if (stereo) {
                active1.resize(expected_band_count);
                for (int i = 0; i < expected_band_count; ++i) {
                    uint8_t byte = blk[ptr + (i >> 3)];
                    active1[i] = (byte >> (i & 7)) & 1;
                }
                ptr += mask_bytes;
            }
        }

        need(1);
        uint8_t k_scale_byte = blk[ptr++];
        int k_scale0 = k_scale_byte & 0x07;
        int k_scale1 = stereo ? ((k_scale_byte >> 3) & 0x07) : 0;
        bool block_is_full = (k_scale_byte & 0x80) != 0;
        bool use_scale_pred = pred_enabled && can_state_pred && ((k_scale_byte & 0x40) != 0);

        int sbr_end = block_is_full ? (3 * expected_band_count / 4) : 0;

        need(2);
        uint16_t payload_len;
        memcpy(&payload_len, &blk[ptr], 2);
        ptr += 2;
        if (ptr + payload_len > blk.size())
            throw std::runtime_error("Payload truncated");

        BitReaderMSB payload_reader(blk.data() + ptr, payload_len);
        ptr += payload_len;

        if (block_format_version >= 24) {
            uint32_t packed_mode = payload_reader.read_bits(expected_band_count);
            uint32_t packed_active0 = payload_reader.read_bits(expected_band_count);
            for (int i = 0; i < expected_band_count; ++i) {
                mode_ms[i] = (packed_mode >> (expected_band_count - 1 - i)) & 1;
                active0[i] = (packed_active0 >> (expected_band_count - 1 - i)) & 1;
            }
            if (stereo) {
                int is_start = get_is_start_band(target_kbps, total_bands);
                int non_is_count = 0;
                for (int i = 0; i < expected_band_count; ++i) {
                    if (!((i >= is_start) && mode_ms[i])) non_is_count++;
                }
                uint32_t packed_active1 = 0;
                if (non_is_count > 0) {
                    packed_active1 = payload_reader.read_bits(non_is_count);
                }
                int bit_idx = non_is_count - 1;
                for (int i = 0; i < expected_band_count; ++i) {
                    bool is_is_band = (i >= is_start) && mode_ms[i];
                    if (!is_is_band) {
                        active1[i] = (packed_active1 >> bit_idx) & 1;
                        bit_idx--;
                    } else {
                        active1[i] = 0;
                    }
                }
            }
        }

        steps0.assign(expected_band_count, 0.0f);
        steps1.assign(expected_band_count, 0.0f);

        auto decode_scales = [&](
            std::vector<float>& steps,
            const std::vector<uint8_t>& active,
            int k_scale,
            std::vector<uint32_t>& prev_scale,
            const std::vector<uint8_t>& prev_active)
        {
            for (int i = 0; i < expected_band_count; ++i) {
                if (!active[i]) continue;
                auto idx_vec = rice_decode(payload_reader, 1, k_scale);
                if (idx_vec.empty()) throw std::runtime_error("Scale index missing");
                uint32_t val = idx_vec[0];
                int sb = get_scale_bits(i);
                int max_idx = (1 << sb) - 1;
                uint32_t idx;
                if (use_scale_pred && i < (int)prev_active.size() && prev_active[i]) {
                    if (i >= (int)prev_scale.size()) throw std::runtime_error("prev_scale out of bounds");
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
                if (i < (int)prev_scale.size()) prev_scale[i] = idx;
            }
        };

        decode_scales(steps0, active0, k_scale0, pred_state.prev_scale0, pred_state.prev_active0);
        if (stereo)
            decode_scales(steps1, active1, k_scale1, pred_state.prev_scale1, pred_state.prev_active1);

        int k_sub_k0 = (int)payload_reader.read_bits(2);
        int k_sub_k1 = stereo ? (int)payload_reader.read_bits(2) : 0;

        int active_count0 = 0, active_count1 = 0;
        for (int i = 0; i < expected_band_count; ++i) {
            if (active0[i]) active_count0++;
            if (stereo && active1[i]) active_count1++;
        }

        auto k_sub_vals0 = rice_decode(payload_reader, active_count0, k_sub_k0);
        std::vector<uint32_t> k_sub_vals1;
        if (stereo)
            k_sub_vals1 = rice_decode(payload_reader, active_count1, k_sub_k1);

        quant0.resize(expected_band_count);
        quant1.resize(expected_band_count);
        ch0_bands.resize(expected_band_count);
        ch1_bands.resize(expected_band_count);

        int k_idx0 = 0;
        for (int i = 0; i < expected_band_count; ++i) {
            quant0[i].clear();
            if (!active0[i]) continue;
            if (k_idx0 >= (int)k_sub_vals0.size()) throw std::runtime_error("k_sub_vals0 out of bounds");
            int k_sub = k_sub_vals0[k_idx0++];
            if (k_sub > 7) k_sub = 7;
            auto uvals = rice_decode(payload_reader, band_shapes[i], k_sub);
            if ((int)uvals.size() != band_shapes[i]) throw std::runtime_error("uvals size mismatch ch0");
            quant0[i].resize(uvals.size());
            for (size_t j = 0; j < uvals.size(); ++j)
                quant0[i][j] = zigzag_decode(uvals[j]);
        }

        if (stereo) {
            int k_idx1 = 0;
            for (int i = 0; i < expected_band_count; ++i) {
                quant1[i].clear();
                if (!active1[i]) continue;
                if (k_idx1 >= (int)k_sub_vals1.size()) throw std::runtime_error("k_sub_vals1 out of bounds");
                int k_sub = k_sub_vals1[k_idx1++];
                if (k_sub > 7) k_sub = 7;
                auto uvals = rice_decode(payload_reader, band_shapes[i], k_sub);
                if ((int)uvals.size() != band_shapes[i]) throw std::runtime_error("uvals size mismatch ch1");
                quant1[i].resize(uvals.size());
                for (size_t j = 0; j < uvals.size(); ++j)
                    quant1[i][j] = zigzag_decode(uvals[j]);
            }
        }

        is_r.resize(expected_band_count);
        for (auto& a : is_r) a.fill(0.5f);
        is_inv.assign(expected_band_count, std::array<bool, 4>{});
        for (auto& a : is_inv) a.fill(false);
        is_use_segmented.assign(expected_band_count, false);

        bool is_used = false;
        if (stereo && target_kbps < 510.0f) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            if (is_start < expected_band_count) {
                std::vector<int> is_indices;
                for (int i = is_start; i < expected_band_count; ++i)
                    if (mode_ms[i]) is_indices.push_back(i);

                if (!is_indices.empty()) {
                    is_used = true;
                    int segmented_threshold = 8 * total_bands / 16;
                    int total_vals = 0;
                    for (int idx : is_indices)
                        total_vals += (idx < segmented_threshold) ? 4 : 1;

                    if (block_format_version >= 24) {
                        int total_inv_bits = 0;
                        for (int idx : is_indices)
                            total_inv_bits += (idx < segmented_threshold) ? 4 : 1;

                        std::vector<uint8_t> inv_bits(total_inv_bits);
                        int abs_idx = 0;
                        int bits_left = total_inv_bits;
                        while (bits_left > 0) {
                            int to_read = std::min(32, bits_left);
                            uint32_t chunk = payload_reader.read_bits(to_read);
                            for (int i = 0; i < to_read; ++i) {
                                inv_bits[abs_idx++] = (chunk >> (to_read - 1 - i)) & 1;
                            }
                            bits_left -= to_read;
                        }

                        int read_idx = 0;
                        for (int idx : is_indices) {
                            bool is_seg = (idx < segmented_threshold);
                            if (is_seg) {
                                for (int s = 0; s < 4; ++s) {
                                    is_inv[idx][s] = (inv_bits[read_idx++] != 0);
                                }
                            } else {
                                is_inv[idx][0] = (inv_bits[read_idx++] != 0);
                            }
                        }
                    } else {
                        int total_inv_bits = (int)is_indices.size();
                        std::vector<uint8_t> inv_bits(total_inv_bits);
                        if (total_inv_bits > 0) {
                            uint32_t chunk = payload_reader.read_bits(total_inv_bits);
                            for (int i = 0; i < total_inv_bits; ++i) {
                                inv_bits[i] = (chunk >> (total_inv_bits - 1 - i)) & 1;
                            }
                        }
                        int read_idx = 0;
                        for (int idx : is_indices) {
                            bool inv_val = (inv_bits[read_idx++] != 0);
                            for (int s = 0; s < 4; ++s) is_inv[idx][s] = inv_val;
                        }
                    }

                    int k_is = (int)payload_reader.read_bits(3);
                    auto is_zz_vals = rice_decode(payload_reader, total_vals, k_is);
                    if ((int)is_zz_vals.size() != total_vals)
                        throw std::runtime_error("IS Rice decode size mismatch");

                    int val_idx = 0;
                    for (int idx : is_indices) {
                        int bits = get_r_bits(idx);
                        if (idx < segmented_threshold) {
                            is_use_segmented[idx] = true;
                            for (int s = 0; s < 4; ++s) {
                                int32_t q = zigzag_decode(is_zz_vals[val_idx++]);
                                is_r[idx][s] = dequantize_r_centered(q, bits);
                            }
                        } else {
                            is_use_segmented[idx] = false;
                            int32_t q = zigzag_decode(is_zz_vals[val_idx++]);
                            is_r[idx].fill(dequantize_r_centered(q, bits));
                        }
                    }
                }
            }
        } else if (block_format_version >= 18 && block_format_version < 24) {
            if (stereo && target_kbps < 510.0f) {
                int is_start = get_is_start_band(target_kbps, total_bands);
                if (is_start < expected_band_count) {
                    int segmented_threshold = 8 * total_bands / 16;
                    if (block_format_version >= 20) {
                        std::vector<int> is_indices;
                        for (int i = is_start; i < expected_band_count; ++i)
                            if (mode_ms[i]) is_indices.push_back(i);

                        if (block_format_version >= 22 && !is_indices.empty()) {
                            is_used = true;
                            BitReaderMSB is_reader(blk.data() + ptr, blk.size() - ptr);
                            size_t is_bits_read = 0;

                            int total_inv_bits = (int)is_indices.size();
                            std::vector<uint8_t> inv_bits(total_inv_bits);
                            if (total_inv_bits > 0) {
                                uint32_t chunk = is_reader.read_bits(total_inv_bits);
                                for (int i = 0; i < total_inv_bits; ++i) {
                                    inv_bits[i] = (chunk >> (total_inv_bits - 1 - i)) & 1;
                                }
                            }
                            int read_idx = 0;
                            for (int idx : is_indices) {
                                bool inv_val = (inv_bits[read_idx++] != 0);
                                for (int s = 0; s < 4; ++s) is_inv[idx][s] = inv_val;
                                is_bits_read += 1;
                            }

                            int k_is = (int)is_reader.read_bits(3);
                            is_bits_read += 3;

                            int total_vals = 0;
                            for (int idx : is_indices)
                                total_vals += (idx < segmented_threshold) ? 4 : 1;

                            auto is_zz_vals = rice_decode(is_reader, total_vals, k_is);
                            if ((int)is_zz_vals.size() != total_vals)
                                throw std::runtime_error("IS Rice decode size mismatch");

                            for (uint32_t v : is_zz_vals)
                                is_bits_read += (v >> k_is) + 1 + k_is;

                            int val_idx = 0;
                            for (int idx : is_indices) {
                                int bits = get_r_bits(idx);
                                if (idx < segmented_threshold) {
                                    is_use_segmented[idx] = true;
                                    for (int s = 0; s < 4; ++s) {
                                        int32_t q = zigzag_decode(is_zz_vals[val_idx++]);
                                        is_r[idx][s] = dequantize_r_centered(q, bits);
                                    }
                                } else {
                                    is_use_segmented[idx] = false;
                                    int32_t q = zigzag_decode(is_zz_vals[val_idx++]);
                                    is_r[idx].fill(dequantize_r_centered(q, bits));
                                }
                            }
                            ptr += (is_bits_read + 7) / 8;
                        }
                    }
                }
            }
        }

        SBRDecodeResult sbr;
        if (block_format_version >= 23) {
            sbr = sbr_decode_v23(
                payload_reader, active0, sbr_end, expected_band_count,
                can_state_pred, pred_state);
        } else {
            sbr = sbr_decode(
                blk.data(), ptr, blk.size(), active0, sbr_end,
                expected_band_count, block_format_version,
                can_state_pred, pred_state);
        }

        for (int i = 0; i < expected_band_count; ++i) {
            ch0_bands[i].clear();
            ch0_bands[i].resize(band_shapes[i], 0.0f);
            if (active0[i]) dequantize_band(quant0[i], steps0[i], ch0_bands[i]);

            if (stereo) {
                ch1_bands[i].clear();
                ch1_bands[i].resize(band_shapes[i], 0.0f);
                if (active1[i]) dequantize_band(quant1[i], steps1[i], ch1_bands[i]);
            }
        }

        sbr_synthesize(ch0_bands, active0, band_shapes, sbr, sbr_end, bi, SBR_W0, SBR_W1);

        if (stereo) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            for (int i = 0; i < expected_band_count; ++i) {
                if (mode_ms[i]) {
                    if (block_format_version >= 18 && is_used && i >= is_start && is_start < expected_band_count) {
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
                } else {
                    if (block_format_version >= 18 && is_used && i >= is_start && is_start < expected_band_count) {
                        std::vector<float> left_tmp, right_tmp;
                        ms_to_lr(ch0_bands[i], ch1_bands[i], left_tmp, right_tmp);
                        ch0_bands[i] = std::move(left_tmp);
                        ch1_bands[i] = std::move(right_tmp);
                    }
                }
            }
        }

        recon_chs.resize(num_channels);
        for (int ch = 0; ch < num_channels; ++ch) {
            auto& bands = (ch == 0) ? ch0_bands : ch1_bands;
            auto rec = wpt.iwpt(bands, sr, level, target_kbps, num_channels);
            if (rec.size() < (size_t)block_size) rec.resize(block_size, 0.0f);
            else rec.resize(block_size);
            recon_chs[ch] = std::move(rec);
        }

        int local_start = (ola.current_pos - ola.output_pos) + (int)ola.head_offset;
        overlap_add_block(ola, recon_chs, block_size, window, window_sq, local_start);
        ola.current_pos += hop;

        extract_ready_samples(ola, write_buf);
        compact_ola_buffer(ola);

        if (write_buf.size() >= write_batch_frames * num_channels)
            flush_write_buffer();

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
                    pred_state.prev_sbr_rms[idx] = sbr.rms_idx[idx];
                }
            }

            pred_state.ready = true;
            pred_state.last_bi = bi;
        }

        prev_block_size = block_size;
    }

    extract_remaining(ola, write_buf);
    flush_write_buffer();

    if (!speed_mode && pow_dec) {
        std::vector<float> pow_tail;
        pow_dec->flush(pow_tail);
        write_samples(pow_tail);
    }

    if (postproc) {
        int n_out = postproc->flush();
        if (n_out > 0) {
            size_t frames_to_write = n_out;
            if (total_frames > 0 && frames_written + frames_to_write > total_frames)
                frames_to_write = total_frames - frames_written;
            if (frames_to_write > 0) {
                drwav_write_pcm_frames(&wav, (uint64_t)frames_to_write, postproc->out_data());
                frames_written += frames_to_write;
            }
        }
    }

    drwav_uninit(&wav);

    std::cout << "Decompressed: " << out_wav
              << " (frames=" << ola.output_pos
              << ", sr=" << sr
              << ", ch=" << (int)num_channels << ")" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: decoder <input.wha> <output.wav> [buffer_size_in_samples] [sbr_w0] [sbr_w1]\n";
        std::cerr << "Note: --speed flag is NO LONGER NEEDED. The decoder reads the mode from the container header automatically.\n";
        return 1;
    }

    size_t buffer_size = 8192;
    if (argc >= 4) {
        try {
            buffer_size = std::stoul(argv[3]);
            if (buffer_size == 0) {
                std::cerr << "Buffer size must be > 0\n";
                buffer_size = 8192;
            }
        } catch (...) {
            std::cerr << "Invalid buffer size\n";
            buffer_size = 8192;
        }
    }

    if (argc >= 5) { try { SBR_W0 = std::stof(argv[4]); } catch (...) {} }
    if (argc >= 6) { try { SBR_W1 = std::stof(argv[5]); } catch (...) {} }

    std::cout << "SBR Shape Weights: W0=" << SBR_W0
              << ", W1=" << SBR_W1
              << " (Contrast: " << (20.0f * std::log10(SBR_W1 / SBR_W0)) << " dB)\n";

    try {
        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);

        decompress_wha_to_wav(argv[1], argv[2], buffer_size);

        getrusage(RUSAGE_SELF, &usage_after);
        double user_time_sec =
            (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
            (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;

        drwav wav_info;
        if (drwav_init_file(&wav_info, argv[2], nullptr)) {
            double duration = (double)wav_info.totalPCMFrameCount / (double)wav_info.sampleRate;
            double speed = (user_time_sec > 0.0) ? (duration / user_time_sec) : 0.0;
            std::cout << "Decoding speed: " << speed
                      << "x realtime (" << user_time_sec << "s user time)\n";
            drwav_uninit(&wav_info);
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}