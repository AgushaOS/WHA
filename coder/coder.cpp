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
#include <sys/resource.h>
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

static const uint16_t IS_INV_HUFF_CODE[16] = {
    0x1, 0x0, 0x5, 0x6, 0x4, 0x19, 0x1B, 0x9,
    0x7, 0x1A, 0x18, 0x1F, 0x5, 0x1E, 0x8, 0x0E
};
static const uint8_t IS_INV_HUFF_LEN[16] = {
    1, 3, 4, 5, 4, 6, 6, 6,
    4, 6, 6, 7, 5, 7, 6, 6
};

static const uint16_t ACTIVE_HUFF_CODE[16] = {
    0x001, 0x002, 0x000, 0x003,
    0x07D, 0x03F, 0x006, 0x001,
    0x01E, 0x0F8, 0x00E, 0x003,
    0x0F9, 0x002, 0x002, 0x001
};
static const uint8_t ACTIVE_HUFF_LEN[16] = {
    2,  5,  6,  5,
    12, 11, 8,  3,
    10, 13, 9,  6,
    13, 7,  6,  1
};

static const uint16_t MODE_MS_HUFF_CODE[16] = {
    0x01, 0x00, 0x0F, 0x0C, 0x0D, 0x04, 0x14, 0x15,
    0x01, 0x09, 0x08, 0x0B, 0x0E, 0x05, 0x03, 0x01
};
static const uint8_t MODE_MS_HUFF_LEN[16] = {
    2, 5, 6, 6, 6, 7, 7, 7,
    4, 6, 6, 6, 6, 7, 6, 1
};

struct CLIOptions {
    std::string input_file;
    std::string output_file;

    float    target_kbps   = SETTINGS.default_target_kbps;
    size_t   read_buffer   = 8192;
    int      is_base       = -1;     
    float    sbr_w0        = 1.0f;
    float    sbr_w1        = 9.0f;
    bool     speed_mode    = false;   

    int      block_mode    = 0;       
    bool     enable_ms     = true;
    bool     enable_is     = true;
    bool     enable_sbr    = true;

    uint32_t max_blocks    = 0;        
    bool     verbose       = false;
};

static void print_help_encoder()
{
    std::cout <<
R"(Usage: encoder <input.wav> [options]

Required:
  <input.wav>                  Path to input WAV file

Bitrate:
  --bitrate <kbps>             Target bitrate (default: 64, max: 452)

Block mode:
  --long-only                  Force 2048-sample long blocks (tonal / stable)
  --short-only                 Force 1024-sample short blocks (percussive)
  (default)                    Auto: short on detected transients

Stereo tools:
  --no-ms                      Disable mid/side for regular (low) bands
  --no-is                      Disable intensity stereo for high bands
                               (recommended at very low bitrates if phase issues)

SBR:
  --no-sbr                     Disable SBR high-frequency continuation
  --sbr-weights <w0> <w1>      SBR spectral shape weights (default 1 9).
                               Contrast = 20*log10(w1/w0) dB. Stored in file.

Adaptive IS base:
  --is-base <n>                -1 = adaptive (default), 0 = disabled,
                               >0 = fixed base band

Preprocessing:
  --speed, --no-pow            Disable POW preprocessing (faster encode).
                               Stored in file; decoder auto-detects.

I/O:
  --read-buffer <samples>      I/O chunk size (default 8192)
  --output <path>              Output file (default <input>.wha)
  --max-blocks <n>             Stop after n blocks (debug)

Logging:
  --quiet, -q                  Suppress progress (Default)
  --verbose, -v                Verbose 
  --help, -h                   Show this help

Examples:
  encoder music.wav --bitrate 96 --no-is
  encoder speech.wav --bitrate 32 --long-only --no-sbr -q
  encoder drum.wav --bitrate 128 --short-only --sbr-weights 1 12
)";
}

static bool parse_encoder_args(int argc, char** argv, CLIOptions& o)
{
    if (argc < 2) return false;
    o.input_file = argv[1];
    if (o.input_file == "--help" || o.input_file == "-h") { print_help_encoder(); return false; }
    o.enable_ms = SETTINGS.enable_ms;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> bool {
            if (i + 1 >= argc) { std::cerr << "Option " << what << " requires a value\n"; return false; }
            return true;
        };

        if      (a == "--help" || a == "-h") { print_help_encoder(); return false; }
        else if (a == "--bitrate")    { if(!need("--bitrate")) return false; o.target_kbps = std::stof(argv[++i]); }
        else if (a == "--read-buffer"){ if(!need("--read-buffer")) return false; o.read_buffer = std::stoul(argv[++i]); if (o.read_buffer < 256) o.read_buffer = 256; }
        else if (a == "--is-base")    { if(!need("--is-base")) return false; o.is_base = std::stoi(argv[++i]); }
        else if (a == "--output")     { if(!need("--output")) return false; o.output_file = argv[++i]; }
        else if (a == "--max-blocks") { if(!need("--max-blocks")) return false; o.max_blocks = (uint32_t)std::stoul(argv[++i]); }
        else if (a == "--sbr-weights") {
            if (i + 2 >= argc) { std::cerr << "--sbr-weights needs 2 values\n"; return false; }
            o.sbr_w0 = std::stof(argv[++i]);
            o.sbr_w1 = std::stof(argv[++i]);
        }
        else if (a == "--speed" || a == "--no-pow") o.speed_mode = true;
        else if (a == "--long-only")   o.block_mode = 1;
        else if (a == "--short-only")  o.block_mode = 2;
        else if (a == "--no-ms")       o.enable_ms  = false;
        else if (a == "--no-is")       o.enable_is  = false;
        else if (a == "--no-sbr")      o.enable_sbr = false;
        else if (a == "--quiet" || a == "-q")   o.verbose = false;
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else { std::cerr << "Unknown option: " << a << "\n"; return false; }
    }

    if (o.output_file.empty()) {
        o.output_file = o.input_file;
        if (o.output_file.size() > 3) o.output_file.resize(o.output_file.size() - 3);
        o.output_file += "wha";
    }
    return true;
}

static int get_adaptive_is_max_base(float target_kbps, const EncoderSettings& s)
{
    int base_default = get_is_base_original(target_kbps);
    if (s.adaptive_is_base_override > 0)
        return std::clamp(s.adaptive_is_base_override, base_default, 16);
    if (s.adaptive_is_base_override == 0)
        return base_default;
    if (target_kbps < s.adaptive_is_kbps_min || target_kbps > s.adaptive_is_kbps_max)
        return base_default;

    float k0, k1; int b0, b1;
    if (target_kbps <= 64.0f)       { k0 = k1 = target_kbps; b0 = b1 = s.adaptive_is_base_at_64; }
    else if (target_kbps <= 96.0f)  { k0 = 64.0f;  k1 = 96.0f;  b0 = s.adaptive_is_base_at_64;  b1 = s.adaptive_is_base_at_96; }
    else if (target_kbps <= 128.0f) { k0 = 96.0f;  k1 = 128.0f; b0 = s.adaptive_is_base_at_96;  b1 = s.adaptive_is_base_at_128; }
    else if (target_kbps <= 160.0f) { k0 = 128.0f; k1 = 160.0f; b0 = s.adaptive_is_base_at_128; b1 = s.adaptive_is_base_at_160; }
    else                            { k0 = 160.0f; k1 = 660.0f; b0 = s.adaptive_is_base_at_160; b1 = s.adaptive_is_base_at_660; }

    float t = 0.0f;
    if (std::abs(k1 - k0) > 1e-6f) t = (target_kbps - k0) / (k1 - k0);
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
    bool enable_sbr,
    bool enable_is,
    bool block_is_full,
    uint32_t block_index,
    PredContext& pred_ctx)
{
    const bool stereo = (num_channels == 2);
    std::vector<std::vector<float>> left_coeffs, right_coeffs;
    std::vector<int> coeff_counts;

    if (stereo) {
        size_t spc = block.size() / 2;
        std::vector<float> left(spc), right(spc);
        for (size_t i = 0; i < spc; ++i) { left[i] = block[i*2]; right[i] = block[i*2+1]; }
        left_coeffs  = wpt.wpt(left,  level, target_kbps, sr, num_channels);
        right_coeffs = wpt.wpt(right, level, target_kbps, sr, num_channels);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i) coeff_counts[i] = (int)left_coeffs[i].size();
    } else {
        left_coeffs = wpt.wpt(block, level, target_kbps, sr, num_channels);
        coeff_counts.resize(left_coeffs.size());
        for (size_t i = 0; i < left_coeffs.size(); ++i) coeff_counts[i] = (int)left_coeffs[i].size();
    }

    int band_count = (int)coeff_counts.size();
    int total_bands = 1 << level;
    ModeState& pred_state = (level == 5) ? pred_ctx.long_state : pred_ctx.short_state;
    pred_state.ensure(band_count);

    bool can_state_pred = false;
    if (pred_state.ready && block_index > pred_state.last_bi) {
        uint32_t age = block_index - pred_state.last_bi;
        if (age <= SAME_MODE_PRED_MAX_AGE) can_state_pred = true;
    }

    bool use_is = (stereo && enable_is && target_kbps < 510.0f);
    std::vector<uint8_t> mode_ms(band_count, 0);
    std::vector<uint8_t> is_band(band_count, 0);
    std::vector<std::vector<float>> ch0_bands(band_count);
    std::vector<std::vector<float>> ch1_bands(band_count);

    int is_start_default = band_count;
    int is_start_max = band_count;
    std::vector<std::array<float, 4>> is_r_vals(band_count);
    for (auto& a : is_r_vals) a.fill(0.5f);
    std::vector<std::array<bool, 4>> is_inv_flags(band_count);
    for (auto& a : is_inv_flags) a.fill(false);
    std::vector<bool> is_use_segmented(band_count, false);

    if (use_is) {
        is_start_default = get_is_start_band(target_kbps, total_bands);
        if (is_start_default >= band_count) use_is = false;
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

    int is_seg_threshold = get_is_segmented_threshold(target_kbps, total_bands);

    if (stereo) {
        int low_end = use_is ? is_start_default : band_count;
        for (int i = 0; i < low_end; ++i) {
            const auto& lb = left_coeffs[i];
            const auto& rb = right_coeffs[i];
            int n = (int)lb.size();
            if (n == 0) { ch0_bands[i] = lb; ch1_bands[i] = rb; continue; }
            float El = 0, Er = 0, cross = 0;
            for (int j = 0; j < n; ++j) { El += lb[j]*lb[j]; Er += rb[j]*rb[j]; cross += lb[j]*rb[j]; }
            float Em = (El + Er + 2.0f*cross) * 0.25f;
            float Es = (El + Er - 2.0f*cross) * 0.25f;
            if (use_mid_side(El, Er, Em, Es, enable_ms, target_kbps)) {
                mode_ms[i] = 1;
                ch0_bands[i].resize(n); ch1_bands[i].resize(n);
                for (int j = 0; j < n; ++j) {
                    ch0_bands[i][j] = (lb[j] + rb[j]) * 0.5f;
                    ch1_bands[i][j] = (lb[j] - rb[j]) * 0.5f;
                }
            } else {
                mode_ms[i] = 0; ch0_bands[i] = lb; ch1_bands[i] = rb;
            }
        }

        if (use_is) {
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
                    is_band[i] = 1; mode_ms[i] = 1;
                    std::array<float, 4> r_arr; std::array<bool, 4> inv_arr;
                    bool use_segmented;
                    std::vector<float> Y;
                    compute_is_parameters_ex(left_coeffs[i], right_coeffs[i], Y, r_arr,
                                             inv_arr, use_segmented, i, total_bands, is_seg_threshold);
                    is_r_vals[i] = r_arr;
                    is_inv_flags[i] = inv_arr;
                    is_use_segmented[i] = (i < is_seg_threshold);
                    ch0_bands[i] = std::move(Y);
                } else {
                    is_band[i] = 0; mode_ms[i] = 0;
                    int n = (int)left_coeffs[i].size();
                    ch0_bands[i].resize(n); ch1_bands[i].resize(n);
                    for (int j = 0; j < n; ++j) {
                        ch0_bands[i][j] = (left_coeffs[i][j] + right_coeffs[i][j]) * 0.5f;
                        ch1_bands[i][j] = (left_coeffs[i][j] - right_coeffs[i][j]) * 0.5f;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < band_count; ++i) { ch0_bands[i] = left_coeffs[i]; ch1_bands[i].clear(); }
    }

    BandEnergy be = compute_band_energy(ch0_bands, ch1_bands, is_band, stereo, band_count);
    std::vector<float> energy0 = std::move(be.e0);
    std::vector<float> energy1 = std::move(be.e1);

    apply_base_pow(energy0, energy1, band_count);
    bool has_transient = detect_strong_transient(
        energy0, energy1, band_count, stereo, SETTINGS.transient_ratio_threshold);
    apply_adaptive_pow(energy0, energy1, band_count, stereo, target_kbps, has_transient);
    apply_narrowband_correction(energy0, energy1, band_count, stereo);
    apply_highfreq_attenuation(energy0, energy1, band_count, target_kbps, num_channels, has_transient);

    static std::vector<std::vector<float>> prev_ch0, prev_ch1;
    static thread_local std::vector<const std::vector<float>*> ch0_views, ch1_views;
    ch0_views.resize(band_count); ch1_views.resize(band_count);
    if (stereo) {
        for (int i = 0; i < band_count; ++i) {
            ch0_views[i] = &ch0_bands[i];
            ch1_views[i] = is_band[i] ? &ch0_bands[i] : &ch1_bands[i];
        }
    } else {
        for (int i = 0; i < band_count; ++i) ch0_views[i] = &ch0_bands[i];
    }

    std::vector<float> priority0, priority1;
    if (stereo) {
        priority0 = compute_channel_priority(ch0_views, prev_ch0, coeff_counts, target_kbps, sr, level, false, num_channels);
        priority1 = compute_channel_priority(ch1_views, prev_ch1, coeff_counts, target_kbps, sr, level, true,  num_channels);
    } else {
        priority0 = compute_channel_priority(ch0_views, prev_ch0, coeff_counts, target_kbps, sr, level, false, num_channels);
        priority1.assign(band_count, 0.0f);
    }

    std::vector<int> min_bits(band_count, 0), max_bits(band_count, 10);
    if (use_is) {
        for (int i = 0; i < band_count; ++i) {
            if (is_band[i]) { min_bits[i] = 0; priority1[i] = 0.0f; energy1[i] = 0.0f; }
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

    float total_energy = 0.0f;
    for (int i = 0; i < band_count; ++i) {
        total_energy += energy0[i];
        if (stereo) total_energy += energy1[i];
    }
    bool is_low_bitrate = (target_kbps <= 80.0f);
    float silence_threshold = is_low_bitrate ? 1e-6f : 1e-7f;
    bool is_silent_block = (total_energy < silence_threshold);

    std::vector<bool> active0(band_count), active1(band_count);
    for (int i = 0; i < band_count; ++i) {
        if (stereo && is_band[i]) bits1[i] = 0;
        if (is_silent_block) { active0[i] = false; active1[i] = false; }
        else {
            active0[i] = (bits0[i] > 0);
            if (stereo) active1[i] = (!is_band[i] && bits1[i] > 0);
            else        active1[i] = false;
        }
        if (!active0[i]) bits0[i] = 0;
        if (!active1[i]) bits1[i] = 0;
    }

    int sbr_end = (enable_sbr && block_is_full) ? (14 * band_count / 16) : 0;

    SBREncodeResult sbr = sbr_analyze(left_coeffs, active0, sbr_end, band_count, SBR_W0, SBR_W1);

    std::vector<uint8_t> header;
    std::vector<std::vector<std::vector<float>>> for_quant0(1);
    for_quant0[0] = std::move(ch0_bands);
    QuantResult qres0 = quantize_levels(for_quant0, bits0, target_kbps, num_channels);

    QuantResult qres1;
    if (stereo) {
        std::vector<std::vector<std::vector<float>>> for_quant1(1);
        for_quant1[0] = std::move(ch1_bands);
        qres1 = quantize_levels(for_quant1, bits1, target_kbps, num_channels);
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

    int k_scale0_pred = 0, k_scale1_pred = 0;
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

    for (int i = 0; i < band_count; i += 4) {
        int nibble = 0;
        for (int s = 0; s < 4; ++s) { int idx = i + s; nibble <<= 1; if (idx < band_count && mode_ms[idx]) nibble |= 1; }
        payload_writer.write_bits(MODE_MS_HUFF_CODE[nibble], MODE_MS_HUFF_LEN[nibble]);
    }
    for (int i = 0; i < band_count; i += 4) {
        int nibble = 0;
        for (int s = 0; s < 4; ++s) { int idx = i + s; nibble <<= 1; if (idx < band_count && active0[idx]) nibble |= 1; }
        payload_writer.write_bits(ACTIVE_HUFF_CODE[nibble], ACTIVE_HUFF_LEN[nibble]);
    }

    if (stereo) {
        int is_start = use_is ? get_is_start_band(target_kbps, total_bands) : band_count;
        std::vector<int> idx_list;
        for (int i = 0; i < band_count; ++i) {
            bool is_is_band = use_is && (i >= is_start) && mode_ms[i];
            if (!is_is_band) idx_list.push_back(i);
        }
        int non_is_count = (int)idx_list.size();
        for (int j = 0; j < non_is_count; j += 4) {
            int nibble = 0;
            for (int s = 0; s < 4; ++s) {
                int idx = j + s; nibble <<= 1;
                if (idx < non_is_count && active1[idx_list[idx]]) nibble |= 1;
            }
            payload_writer.write_bits(ACTIVE_HUFF_CODE[nibble], ACTIVE_HUFF_LEN[nibble]);
        }
    }

    rice_encode(payload_writer, scale_vals0, k_scale0);
    if (stereo) rice_encode(payload_writer, scale_vals1, k_scale1);

    std::vector<uint32_t> k_sub_vals0, k_sub_vals1;
    std::vector<int> k_sub_per_band0(band_count, 0), k_sub_per_band1(band_count, 0);
    std::vector<std::vector<uint32_t>> uvals0(band_count), uvals1(band_count);
    k_sub_vals0.reserve(band_count);
    if (stereo) k_sub_vals1.reserve(band_count);

    for (int i = 0; i < band_count; ++i) {
        if (active0[i] && bits0[i] > 0) {
            const auto& qvals = qres0.quantized_per_level[i];
            auto& uv = uvals0[i]; uv.resize(qvals.size());
            for (size_t j = 0; j < qvals.size(); ++j) uv[j] = zigzag_encode(qvals[j]);
            int k_sub = compute_optimal_rice_k(uv);
            k_sub_per_band0[i] = k_sub; k_sub_vals0.push_back(k_sub);
        }
        if (stereo && active1[i] && bits1[i] > 0) {
            const auto& qvals = qres1.quantized_per_level[i];
            auto& uv = uvals1[i]; uv.resize(qvals.size());
            for (size_t j = 0; j < qvals.size(); ++j) uv[j] = zigzag_encode(qvals[j]);
            int k_sub = compute_optimal_rice_k(uv);
            k_sub_per_band1[i] = k_sub; k_sub_vals1.push_back(k_sub);
        }
    }

    int k_sub_k0 = k_sub_vals0.empty() ? 0 : std::min(compute_optimal_rice_k(k_sub_vals0), 3);
    int k_sub_k1 = 0;
    if (stereo && !k_sub_vals1.empty()) k_sub_k1 = std::min(compute_optimal_rice_k(k_sub_vals1), 3);

    payload_writer.write_bits(k_sub_k0, 2);
    if (stereo) payload_writer.write_bits(k_sub_k1, 2);
    rice_encode(payload_writer, k_sub_vals0, k_sub_k0);
    if (stereo) rice_encode(payload_writer, k_sub_vals1, k_sub_k1);

    for (int i = 0; i < band_count; ++i)
        if (active0[i] && bits0[i] > 0) rice_encode(payload_writer, uvals0[i], k_sub_per_band0[i]);
    if (stereo) {
        for (int i = 0; i < band_count; ++i)
            if (active1[i] && bits1[i] > 0) rice_encode(payload_writer, uvals1[i], k_sub_per_band1[i]);
    }

    if (use_is) {
        std::vector<int> is_indices;
        for (int i = 0; i < band_count; ++i) if (is_band[i]) is_indices.push_back(i);
        if (!is_indices.empty()) {
            int total_vals = 0;
            for (int idx : is_indices) total_vals += (idx < is_seg_threshold) ? 4 : 1;
            for (int i : is_indices) {
                if (i < is_seg_threshold) {
                    int nibble = 0;
                    for (int s = 0; s < 4; ++s) nibble = (nibble << 1) | (is_inv_flags[i][s] ? 1 : 0);
                    payload_writer.write_bits(IS_INV_HUFF_CODE[nibble], IS_INV_HUFF_LEN[nibble]);
                } else {
                    payload_writer.write_bits(is_inv_flags[i][0] ? 1u : 0u, 1);
                }
            }
            std::vector<uint32_t> is_zz_vals; is_zz_vals.reserve(total_vals);
            for (int i : is_indices) {
                int bits = get_r_bits(i);
                if (is_use_segmented[i]) {
                    for (int s = 0; s < 4; ++s)
                        is_zz_vals.push_back(zigzag_encode(quantize_r_centered(is_r_vals[i][s], bits)));
                } else {
                    is_zz_vals.push_back(zigzag_encode(quantize_r_centered(is_r_vals[i][0], bits)));
                }
            }
            int k_is = compute_optimal_rice_k(is_zz_vals);
            payload_writer.write_bits(k_is & 0x07, 3);
            rice_encode(payload_writer, is_zz_vals, k_is);
        }
    }

    if (sbr_end > 0) {
        std::vector<int> sbr_indices;
        for (int i = 0; i < sbr_end; ++i) if (!active0[i]) sbr_indices.push_back(i);
        if (!sbr_indices.empty())
            sbr_write_to_stream(payload_writer, sbr, sbr_indices, can_state_pred, pred_state);
    }

    payload_writer.flush();
    auto payload = payload_writer.data();
    if (payload.size() >= 0x8000) throw std::runtime_error("Block payload exceeds 32767 bytes");

    uint16_t payload_len_field = (uint16_t)payload.size();
    if (level != 5) payload_len_field |= 0x8000;
    header.push_back(payload_len_field & 0xFF);
    header.push_back((payload_len_field >> 8) & 0xFF);

    std::vector<uint8_t> out = header;
    out.insert(out.end(), payload.begin(), payload.end());

    pred_state.prev_scale0 = full_scale0;
    pred_state.prev_scale1 = full_scale1;
    pred_state.prev_active0.assign(band_count, 0);
    for (int i = 0; i < band_count; ++i) if (active0[i]) pred_state.prev_active0[i] = 1;
    if (stereo) {
        pred_state.prev_active1.assign(band_count, 0);
        for (int i = 0; i < band_count; ++i) if (active1[i]) pred_state.prev_active1[i] = 1;
    } else {
        pred_state.prev_active1.assign(band_count, 0);
    }

    pred_state.prev_sbr_valid.assign(band_count, 0);
    pred_state.prev_sbr_noise.assign(band_count, 0);
    for (int i = 0; i < sbr_end && i < band_count; ++i) {
        if (!active0[i]) {
            pred_state.prev_sbr_valid[i] = 1;
            pred_state.prev_sbr_rms[i]   = sbr.rms_idx[i];
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
    uint8_t coding_flags,
    float sbr_w0,
    float sbr_w1,
    size_t write_buffer_blocks = 8192)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot create output file");

    f.write("WHA1", 4);
    uint8_t version = 30;
    f.write((char*)&version, 1);
    f.write((char*)&coding_flags, 1);
    f.write((char*)&sbr_w0, 4);
    f.write((char*)&sbr_w1, 4);

    f.write((char*)&sr, 4);
    uint8_t ch = (uint8_t)num_channels; f.write((char*)&ch, 1);
    uint32_t block_count = (uint32_t)blocks.size(); f.write((char*)&block_count, 4);
    f.write((char*)&target_kbps, 4);
    uint8_t block_format_version = 29; f.write((char*)&block_format_version, 1);
    f.write((char*)&total_frames, 4);

    std::vector<uint8_t> write_buffer;
    write_buffer.reserve(write_buffer_blocks * 4096);
    auto flush_buffer = [&]() {
        if (!write_buffer.empty()) { f.write((char*)write_buffer.data(), write_buffer.size()); write_buffer.clear(); }
    };

    for (uint32_t i = 0; i < block_count; ++i) {
        const auto& blk = blocks[i];
        uint16_t len = (uint16_t)blk.size();
        write_buffer.insert(write_buffer.end(), (char*)&len, (char*)&len + 2);
        write_buffer.insert(write_buffer.end(), blk.begin(), blk.end());
        if (write_buffer.size() >= write_buffer_blocks * 4096) flush_buffer();
    }
    flush_buffer();
    f.close();

    int long_blocks = (int)std::count(block_modes.begin(), block_modes.end(), true);
    int short_blocks = (int)block_count - long_blocks;
    std::cout << "Saved container v30 to " << path << " ("
              << blocks.size() << " blocks, "
              << long_blocks << " long, "
              << short_blocks << " short, speed_mode=" << (speed_mode ? "yes" : "no")
              << ", flags=0x" << std::hex << (int)coding_flags << std::dec << ")" << std::endl;
}

std::tuple<std::vector<std::vector<uint8_t>>, uint32_t, int, float, uint32_t, std::vector<bool>, bool>
compress_audio_streaming(const std::string& input_path, const CLIOptions& opts)
{
    uint32_t sr, channels;
    drwav wav;
    if (!drwav_init_file(&wav, input_path.c_str(), nullptr))
        throw std::runtime_error("Cannot open WAV file");

    sr = wav.sampleRate;
    channels = wav.channels;
    uint64_t total_frames = wav.totalPCMFrameCount;
    int num_channels = (int)channels;

    float target_kbps = (opts.target_kbps > 0) ? opts.target_kbps : SETTINGS.default_target_kbps;
    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor = (per_channel_kbps <= 0.0f) ? 2.0/3.0f : 1.0f;

    PWPT wpt;
    std::vector<std::vector<uint8_t>> blocks_raw;
    std::vector<bool> block_modes;
    PredContext pred_ctx;

    std::unique_ptr<PowCodec> pow_enc;
    if (!opts.speed_mode)
        pow_enc = std::make_unique<PowCodec>(num_channels, 0.85f, target_kbps / float(num_channels));

    int current_pos = 0;
    int prev_block_size = 1024;
    float bits_per_sample = target_kbps * 1000.0f / sr;
    int reservoir = 0;
    int max_reservoir = (int)(bits_per_sample * 2048 * 0.3);

    std::vector<float> audio_buffer;
    size_t audio_head = 0;
    size_t buffer_start = 0;
    bool eof = false;
    bool pow_flushed = false;

    std::vector<float> analyze_buf, block;
    std::vector<float> read_chunk(opts.read_buffer * num_channels);
    std::vector<float> processed_buf, tail_buf;

    struct WindowCache { std::vector<float> buf; int bs = 0, lo = -1, ro = -1; };
    WindowCache win_long, win_short;

    auto get_window = [&](int bs, int lo, int ro) -> const std::vector<float>& {
        WindowCache& wc = (bs == 2048) ? win_long : win_short;
        if (wc.bs != bs || wc.lo != lo || wc.ro != ro) {
            wc.buf.assign(bs, 1.0f);
            if (lo > 1) for (int i = 0; i < lo; ++i) { float t = (float)i/lo; wc.buf[i] = t*t*(3.0f-2.0f*t); }
            if (ro > 1) for (int i = 0; i < ro; ++i) { float t = (float)i/ro; wc.buf[bs-1-i] = t*t*(3.0f-2.0f*t); }
            wc.bs = bs; wc.lo = lo; wc.ro = ro;
        }
        return wc.buf;
    };

    auto ensure_data_available = [&](int needed_pos) {
        while ((int)(buffer_start + (audio_buffer.size() - audio_head) / num_channels) < needed_pos) {
            if (eof) {
                if (!pow_flushed) {
                    pow_flushed = true;
                    if (!opts.speed_mode && pow_enc) {
                        tail_buf.clear(); pow_enc->flush(tail_buf);
                        audio_buffer.insert(audio_buffer.end(), tail_buf.begin(), tail_buf.end());
                        if (!tail_buf.empty()) continue;
                    }
                }
                break;
            }
            drwav_uint64 frames_read = drwav_read_pcm_frames_f32(
                &wav, opts.read_buffer, read_chunk.data());
            if (frames_read == 0) { eof = true; continue; }
            if (!opts.speed_mode) {
                processed_buf.clear();
                pow_enc->process(read_chunk.data(), (size_t)frames_read, processed_buf);
                audio_buffer.insert(audio_buffer.end(), processed_buf.begin(), processed_buf.end());
            } else {
                audio_buffer.insert(audio_buffer.end(), read_chunk.begin(),
                                    read_chunk.begin() + frames_read * num_channels);
            }
        }
    };

    ensure_data_available(2048);

    while (current_pos < (int)total_frames) {
        if (opts.max_blocks > 0 && blocks_raw.size() >= opts.max_blocks) break;

        int analyze_len = std::min(2048, (int)total_frames - current_pos);
        ensure_data_available(current_pos + analyze_len);
        if ((int)(buffer_start + (audio_buffer.size() - audio_head) / num_channels) <= current_pos) break;

        size_t need_an = (size_t)analyze_len * num_channels;
        if (analyze_buf.size() != need_an) analyze_buf.resize(need_an);
        std::memset(analyze_buf.data(), 0, need_an * sizeof(float));
        size_t offset = audio_head + (current_pos - (int)buffer_start) * num_channels;
        if (offset + analyze_buf.size() <= audio_buffer.size()) {
            std::copy(audio_buffer.begin() + offset, audio_buffer.begin() + offset + analyze_buf.size(), analyze_buf.begin());
        } else {
            size_t avail = audio_buffer.size() - offset;
            std::copy(audio_buffer.begin() + offset, audio_buffer.end(), analyze_buf.begin());
        }

        bool is_transient;
        if      (opts.block_mode == 1) is_transient = false;
        else if (opts.block_mode == 2) is_transient = true;
        else                           is_transient = detect_transient_block_stereo(analyze_buf, num_channels);

        int block_size, right_overlap, level;
        if (is_transient) { block_size = 1024; right_overlap = (int)(48 * overlap_factor); level = 4; }
        else              { block_size = 2048; right_overlap = (int)(96 * overlap_factor); level = 5; }

        int left_overlap = (prev_block_size == 2048)
            ? (int)(96 * overlap_factor)
            : (int)(48 * overlap_factor);
        int hop = block_size - right_overlap;

        ensure_data_available(current_pos + block_size);

        size_t need_blk = (size_t)block_size * num_channels;
        if (block.size() != need_blk) block.resize(need_blk);
        std::memset(block.data(), 0, need_blk * sizeof(float));

        size_t offset_block = audio_head + (current_pos - (int)buffer_start) * num_channels;
        size_t avail_block = (offset_block < audio_buffer.size()) ? (audio_buffer.size() - offset_block) : 0;
        if (avail_block > 0) {
            size_t copy_samples = std::min(avail_block, block.size());
            std::copy(audio_buffer.begin() + offset_block,
                      audio_buffer.begin() + offset_block + copy_samples, block.begin());
        }

        const std::vector<float>& window = get_window(block_size, left_overlap, right_overlap);
        if (num_channels == 2) {
            for (int i = 0; i < block_size; ++i) {
                float w = window[i];
                block[i*2] *= w; block[i*2+1] *= w;
            }
        } else {
            for (int i = 0; i < block_size; ++i) block[i] *= window[i];
        }

        bool block_is_full = (current_pos + block_size <= (int)total_frames);
        int target_bits = (int)(bits_per_sample * hop);
        if (target_bits < 256) target_bits = 256;
        int effective_budget = target_bits + reservoir;
        if (effective_budget > target_bits + max_reservoir) effective_budget = target_bits + max_reservoir;

        uint32_t block_index = (uint32_t)blocks_raw.size();
        auto comp = compress_block_adaptive_joint(
            block, wpt, num_channels, level, effective_budget, target_kbps, sr,
            block_size, opts.enable_ms, opts.enable_sbr, opts.enable_is,
            block_is_full, block_index, pred_ctx);

        int real_bits = (int)comp.size() * 8;
        reservoir += target_bits - real_bits;
        if (reservoir < 0) reservoir = 0;
        if (reservoir > max_reservoir) reservoir = max_reservoir;

        blocks_raw.push_back(std::move(comp));
        block_modes.push_back(!is_transient);

        if (opts.verbose && (blocks_raw.size() % 10 == 0)) {
            std::cout << "[Block " << blocks_raw.size() << "] size=" << block_size
                      << ", mode=" << (is_transient ? "short" : "long")
                      << ", bytes=" << blocks_raw.back().size()
                      << ", reservoir=" << reservoir << std::endl;
        }

        current_pos += hop;
        prev_block_size = block_size;

        int keep_start = current_pos - left_overlap;
        if (keep_start < 0) keep_start = 0;
        size_t remove_samples = (size_t)(keep_start - (int)buffer_start) * num_channels;
        if (remove_samples > 0 && audio_head + remove_samples <= audio_buffer.size()) {
            audio_head += remove_samples;
            buffer_start = keep_start;
            if (audio_head > 16384 && audio_head * 2 > audio_buffer.size()) {
                audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + audio_head);
                audio_head = 0;
            }
        }
        ensure_data_available(current_pos + 2048);
    }

    drwav_uninit(&wav);

    uint32_t frames_out = (opts.max_blocks > 0 && blocks_raw.size() >= opts.max_blocks)
                          ? 0u
                          : (uint32_t)current_pos;

    return {
        std::move(blocks_raw),
        sr,
        num_channels,
        target_kbps,
        frames_out,
        std::move(block_modes),
        opts.speed_mode
    };
}

int main(int argc, char** argv)
{
    CLIOptions opts;
    if (!parse_encoder_args(argc, argv, opts)) return 1;

    if (opts.target_kbps > 452.0f) {
        std::cerr << "Bitrate isn't supported (max 452 kbps)\n";
        return 1;
    }

    if (opts.is_base == 0) {
        SETTINGS.enable_adaptive_is = false;
        SETTINGS.adaptive_is_base_override = -1;
    } else if (opts.is_base > 0) {
        SETTINGS.enable_adaptive_is = true;
        SETTINGS.adaptive_is_base_override = opts.is_base;
    } else {
        SETTINGS.enable_adaptive_is = true;
        SETTINGS.adaptive_is_base_override = -1;
    }

    SBR_W0 = opts.sbr_w0;
    SBR_W1 = opts.sbr_w1;

    std::cout << "SBR Shape Weights: W0=" << SBR_W0
              << ", W1=" << SBR_W1
              << " (Contrast: " << (20.0f * std::log10(SBR_W1 / SBR_W0)) << " dB)\n";

    if (opts.speed_mode)
        std::cout << ">>> SPEED MODE: POW preprocessing DISABLED (saved to container) <<<\n";

    struct rusage usage_before, usage_after;
    getrusage(RUSAGE_SELF, &usage_before);

    auto comp = compress_audio_streaming(opts.input_file, opts);
    auto blocks = std::get<0>(comp);
    uint32_t sr = std::get<1>(comp);
    int num_channels = std::get<2>(comp);
    float tk = std::get<3>(comp);
    uint32_t total_samples = std::get<4>(comp);
    auto block_modes = std::get<5>(comp);
    bool final_speed_mode = std::get<6>(comp);

    uint8_t coding_flags = 0;
    if (final_speed_mode) coding_flags |= 0x01;
    if (!opts.enable_ms)  coding_flags |= 0x02;

    bool is_will_be_used = (num_channels == 2) && opts.enable_is && (opts.target_kbps < 510.0f);
    if (!is_will_be_used) coding_flags |= 0x04;

    if (!opts.enable_sbr) coding_flags |= 0x08;
    coding_flags |= (uint8_t)((opts.block_mode & 0x03) << 4);

    save_compressed_buffered(
        blocks, block_modes, opts.output_file, sr, num_channels, tk,
        total_samples, final_speed_mode, coding_flags,
        SBR_W0, SBR_W1, opts.read_buffer);

    getrusage(RUSAGE_SELF, &usage_after);
    double user_time_sec =
        (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
        (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;

    double audio_duration_sec = (total_samples > 0)
        ? (double)total_samples / (double)sr : 0.0;

    size_t total_file_bytes = 32;
    for (const auto& blk : blocks) total_file_bytes += 2 + blk.size();

    double actual_bitrate_kbps = (audio_duration_sec > 0.0)
        ? (double)total_file_bytes * 8.0 / 1000.0 / audio_duration_sec : 0.0;
    double realtime_speed = (user_time_sec > 0.0) ? (audio_duration_sec / user_time_sec) : 0.0;

    std::cout << "Compressed: " << blocks.size() << " blocks, user time=" << user_time_sec << "s\n";
    std::cout << "Audio duration: " << audio_duration_sec << " s\n";
    std::cout << "File size: " << total_file_bytes << " bytes\n";
    std::cout << "Target bitrate: " << opts.target_kbps << " kbps\n";
    std::cout << "Actual bitrate: " << actual_bitrate_kbps << " kbps\n";
    std::cout << "Encoding speed: " << realtime_speed << "x realtime\n";
    return 0;
}