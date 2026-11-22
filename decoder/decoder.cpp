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
#include "table.h"
#include "kidx_huff.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float SBR_W0 = 1.0f;
float SBR_W1 = 9.0f;

static const int IS_TBL_LO = -7;
static const int IS_TBL_HI =  7;
static const int IS_TBL_N  = 15;
static const int IS_TBL_SLOTS = 8;
static const uint16_t IS_TBL_CODE[IS_TBL_SLOTS][IS_TBL_N] = {
{ 0x681, 0x683, 0x0D1, 0x035, 0x00C, 0x003, 0x000, 0x002,
  0x007, 0x002, 0x01B, 0x069, 0x1A1, 0x680, 0x682 },
{ 0, 0, 0, 0, 0x034, 0x01B, 0x007, 0x000,
  0x002, 0x00C, 0x035, 0, 0, 0, 0 },
{ 0, 0, 0, 0x013, 0x001, 0x005, 0x007, 0x001,
  0x006, 0x000, 0x008, 0x012, 0, 0, 0 },
{ 0, 0, 0, 0, 0x004, 0x003, 0x006, 0x002,
  0x001, 0x007, 0x000, 0x005, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0x00D, 0x000,
  0x001, 0x002, 0x007, 0x00C, 0, 0, 0 },
{ 0, 0, 0x009, 0x003, 0x002, 0x000, 0x006, 0x007,
  0x005, 0, 0, 0x008, 0, 0, 0 },
{ 0, 0, 0, 0x01C, 0x006, 0x000, 0x001, 0x002,
  0x00F, 0x01D, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0x000, 0x001, 0x005, 0x003,
  0x001, 0x004, 0x001, 0, 0, 0, 0 },
};
static const uint8_t IS_TBL_LEN[IS_TBL_SLOTS][IS_TBL_N] = {
{ 11, 11, 8, 6, 4, 3, 2, 2, 3, 3, 5, 7, 9, 11, 11 },
{  0,  0, 0, 0, 6, 5, 3, 1, 2, 4, 6, 0, 0,  0,  0 },
{  0,  0, 0, 5, 3, 3, 3, 2, 3, 3, 4, 5, 0,  0,  0 },
{  0,  0, 0, 0, 5, 4, 3, 2, 2, 3, 3, 5, 0,  0,  0 },
{  0,  0, 0, 0, 0, 0, 4, 2, 2, 2, 3, 4, 0,  0,  0 },
{  0,  0, 5, 3, 2, 2, 3, 3, 4, 0, 0, 5, 0,  0,  0 },
{  0,  0, 0, 5, 3, 2, 2, 2, 4, 5, 0, 0, 0,  0,  0 },
{  0,  0, 0, 0, 4, 3, 3, 2, 2, 3, 4, 0, 0,  0,  0 },
};

static const uint16_t IS_INV_HUFF_CODE[16] = {
    0x1, 0x0, 0x5, 0x6, 0x4, 0x19, 0x1B, 0x9,
    0x7, 0x1A, 0x18, 0x1F, 0x5, 0x1E, 0x8, 0x0E
};
static const uint8_t IS_INV_HUFF_LEN[16] = {
    1, 3, 4, 5, 4, 6, 6, 6, 4, 6, 6, 7, 5, 7, 6, 6
};

static const uint16_t ACTIVE_HUFF_CODE[16] = {
    0x001, 0x002, 0x000, 0x003, 0x07D, 0x03F, 0x006, 0x001,
    0x01E, 0x0F8, 0x00E, 0x003, 0x0F9, 0x002, 0x002, 0x001
};
static const uint8_t ACTIVE_HUFF_LEN[16] = {
    2,  5,  6,  5, 12, 11, 8,  3, 10, 13, 9,  6, 13, 7,  6,  1
};

static const uint16_t MODE_MS_HUFF_CODE[16] = {
    0x01, 0x00, 0x0F, 0x0C, 0x0D, 0x04, 0x14, 0x15,
    0x01, 0x09, 0x08, 0x0B, 0x0E, 0x05, 0x03, 0x01
};
static const uint8_t MODE_MS_HUFF_LEN[16] = {
    2, 5, 6, 6, 6, 7, 7, 7, 4, 6, 6, 6, 6, 7, 6, 1
};

static const int WPT_HUFF_SLOT_MAP[8] = { 0, 1, 2, 4, 5, 6, 7, 8 };

struct SymCode { uint16_t code; uint8_t sym; };

static const SymCode ACTIVE_BY_LEN[14][8] = {
    {{0,0}}, {{0x1,15}}, {{0x1, 0}}, {{0x1, 7}}, {{0,0}},
    {{0x2, 1}, {0x3, 3}},
    {{0x0, 2}, {0x3, 11}, {0x2, 14}},
    {{0x2,13}}, {{0x6, 6}}, {{0xE,10}}, {{0x1E, 8}}, {{0x3F, 5}}, {{0x7D, 4}},
    {{0xF8, 9}, {0xF9, 12}}
};
static const int ACTIVE_COUNT[14] = {0,1,1,1,0,2,3,1,1,1,1,1,1,2};

static const SymCode MODE_MS_BY_LEN[8][8] = {
    {{0,0}}, {{0x1,15}}, {{0x1, 0}}, {{0,0}}, {{0x1, 8}}, {{0x0, 1}},
    {{0xF, 2}, {0xC, 3}, {0xD, 4}, {0x9, 9}, {0x8, 10}, {0xB, 11}, {0xE, 12}, {0x3, 14}},
    {{0x4, 5}, {0x14, 6}, {0x15, 7}, {0x5, 13}}
};
static const int MODE_MS_COUNT[8] = {0,1,1,0,1,1,8,4};

static const SymCode IS_INV_BY_LEN[8][8] = {
    {{0,0}}, {{0x1, 0}}, {{0,0}}, {{0x0, 1}},
    {{0x5, 2}, {0x4, 4}, {0x7, 8}},
    {{0x6, 3}, {0x5, 12}},
    {{0x19, 5}, {0x1B, 6}, {0x9, 7}, {0x1A, 9}, {0x18, 10}, {0x8, 14}, {0xE, 15}},
    {{0x1F, 11}, {0x1E, 13}}
};
static const int IS_INV_COUNT[8] = {0,1,0,1,3,2,7,2};

static int16_t g_active_lut  [1 << 14];
static int16_t g_mode_ms_lut [1 <<  8];
static int16_t g_is_inv_lut  [1 <<  8];
static int16_t g_is_coeff_lut[IS_TBL_SLOTS][1 << 12];
static int16_t g_kidx_lut    [8][1 << 10];

static void build_huffman_luts()
{
    static bool built = false;
    if (built) return;
    built = true;
    constexpr int16_t NONE = INT16_MIN;

    std::fill(g_active_lut, g_active_lut + (1 << 14), NONE);
    for (int len = 1; len <= 13; ++len) {
        int base = (1 << len) - 2;
        for (int k = 0; k < ACTIVE_COUNT[len]; ++k)
            g_active_lut[base + (int)ACTIVE_BY_LEN[len][k].code] =
                (int16_t)ACTIVE_BY_LEN[len][k].sym;
    }

    std::fill(g_mode_ms_lut, g_mode_ms_lut + (1 << 8), NONE);
    for (int len = 1; len <= 7; ++len) {
        int base = (1 << len) - 2;
        for (int k = 0; k < MODE_MS_COUNT[len]; ++k)
            g_mode_ms_lut[base + (int)MODE_MS_BY_LEN[len][k].code] =
                (int16_t)MODE_MS_BY_LEN[len][k].sym;
    }

    std::fill(g_is_inv_lut, g_is_inv_lut + (1 << 8), NONE);
    for (int len = 1; len <= 7; ++len) {
        int base = (1 << len) - 2;
        for (int k = 0; k < IS_INV_COUNT[len]; ++k)
            g_is_inv_lut[base + (int)IS_INV_BY_LEN[len][k].code] =
                (int16_t)IS_INV_BY_LEN[len][k].sym;
    }

    for (int t = 0; t < IS_TBL_SLOTS; ++t) {
        std::fill(g_is_coeff_lut[t], g_is_coeff_lut[t] + (1 << 12), NONE);
        for (int k = 0; k < IS_TBL_N; ++k) {
            int len = IS_TBL_LEN[t][k];
            if (len == 0) continue;
            int base = (1 << len) - 2;
            g_is_coeff_lut[t][base + (int)IS_TBL_CODE[t][k]] =
                (int16_t)(k + IS_TBL_LO);
        }
    }

    for (int s = 0; s < 8; ++s) {
        std::fill(g_kidx_lut[s], g_kidx_lut[s] + (1 << 10), NONE);
        for (int k = 0; k < 17; ++k) {
            int len = KIDX_COEFF_HUFF_LEN[s][k];
            if (len == 0) continue;
            int base = (1 << len) - 2;
            g_kidx_lut[s][base + (int)KIDX_COEFF_HUFF_CODE[s][k]] = (int16_t)k;
        }
    }
}

static inline int decode_is_coeff_with_table(BitReaderMSB& r, int table_idx)
{
    uint32_t c = 0;
    const int16_t* lut = g_is_coeff_lut[table_idx];
    for (int len = 1; len <= 11; ++len) {
        c = (c << 1) | r.read_bits(1);
        int16_t s = lut[((1 << len) - 2) + (int)c];
        if (s != INT16_MIN) return s;
    }
    throw std::runtime_error("Invalid Huffman code for IS coeff");
}

static inline int huffman_decode_is_inv(BitReaderMSB& r) {
    uint32_t c = 0;
    for (int len = 1; len <= 7; ++len) {
        c = (c << 1) | r.read_bits(1);
        int16_t s = g_is_inv_lut[((1 << len) - 2) + (int)c];
        if (s != INT16_MIN) return s;
    }
    throw std::runtime_error("Invalid Huffman code for IS inversion");
}

static inline int huffman_decode_active(BitReaderMSB& r) {
    uint32_t c = 0;
    for (int len = 1; len <= 13; ++len) {
        c = (c << 1) | r.read_bits(1);
        int16_t s = g_active_lut[((1 << len) - 2) + (int)c];
        if (s != INT16_MIN) return s;
    }
    throw std::runtime_error("Invalid Huffman code for active mask");
}

static inline int huffman_decode_mode_ms(BitReaderMSB& r) {
    uint32_t c = 0;
    for (int len = 1; len <= 7; ++len) {
        c = (c << 1) | r.read_bits(1);
        int16_t s = g_mode_ms_lut[((1 << len) - 2) + (int)c];
        if (s != INT16_MIN) return s;
    }
    throw std::runtime_error("Invalid Huffman code for mode_ms");
}

static inline int decode_huffman_small(BitReaderMSB& r) {
    if (r.read_bits(1) != 0) return 0;
    if (r.read_bits(1) == 0) return 1;
    if (r.read_bits(1) != 0) return -1;
    return r.read_bits(1) ? -2 : 2;
}
static inline void decode_huffman_small_vec(BitReaderMSB& r, int32_t* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = decode_huffman_small(r);
}

static inline int decode_huffman_large(BitReaderMSB& r) {
    if (r.read_bits(1) == 0) return 0;
    if (r.read_bits(1) == 1) {
        return r.read_bits(1) ? -1 : 1;
    }
    if (r.read_bits(1) == 0) return -2;
    if (r.read_bits(1) == 1) return 2;
    return r.read_bits(1) ? -3 : 3;
}
static inline void decode_huffman_large_vec(BitReaderMSB& r, int32_t* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = decode_huffman_large(r);
}

static inline int decode_kidx_huff(BitReaderMSB& r, int slot) {
    uint32_t c = 0;
    const int16_t* lut = g_kidx_lut[slot];
    for (int len = 1; len <= 9; ++len) {
        c = (c << 1) | r.read_bits(1);
        int16_t s = lut[((1 << len) - 2) + (int)c];
        if (s != INT16_MIN) return s;
    }
    throw std::runtime_error("Invalid kidx Huffman code");
}
static inline std::vector<uint32_t> decode_kidx_huff_vec(BitReaderMSB& r, size_t n, int slot) {
    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint32_t)decode_kidx_huff(r, slot);
    return out;
}

struct WptHuffSlotDecoder {
    static constexpr int SHORT_MAX = 12;
    int max_len = 0;
    std::vector<int16_t> lut_short;
    std::vector<std::vector<std::pair<int,int>>> by_len;
};

static void build_wpt_decoder(WptHuffSlotDecoder& dec, int slot) {
    dec.max_len = 0;
    dec.by_len.assign(23, {});
    dec.lut_short.assign(1 << (WptHuffSlotDecoder::SHORT_MAX + 1), INT16_MIN);
    for (int i = 0; i < 255; ++i) {
        int len = WPT_COEFF_HUFF_LEN[slot][i];
        if (len == 0) continue;
        int code = (int)WPT_COEFF_HUFF_CODE[slot][i];
        int sym  = i + WPT_COEFF_HUFF_LO;
        if (len <= WptHuffSlotDecoder::SHORT_MAX) {
            dec.lut_short[((1 << len) - 2) + code] = (int16_t)sym;
        } else {
            dec.by_len[len].push_back({code, sym});
        }
        if (len > dec.max_len) dec.max_len = len;
    }
}

static inline int decode_wpt_huff(BitReaderMSB& r, const WptHuffSlotDecoder& dec) {
    int code = 0;
    const int16_t* lut = dec.lut_short.data();
    for (int len = 1; len <= WptHuffSlotDecoder::SHORT_MAX; ++len) {
        code = (code << 1) | (int)r.read_bits(1);
        int16_t s = lut[((1 << len) - 2) + code];
        if (s != INT16_MIN) return s;
    }
    for (int len = WptHuffSlotDecoder::SHORT_MAX + 1; len <= dec.max_len; ++len) {
        code = (code << 1) | (int)r.read_bits(1);
        for (const auto& p : dec.by_len[len])
            if (p.first == code) return p.second;
    }
    throw std::runtime_error("Invalid WPT Huffman code");
}

struct DecoderOptions {
    std::string input_wha;
    std::string output_wav;
    size_t buffer = 8192;
    bool sbr_override = false;
    float sbr_w0 = 1.0f, sbr_w1 = 9.0f;
};

static void print_help_decoder()
{
    std::cout <<
R"(Usage: decoder <input.wha> [options]
Arguments:
<input.wha>                  Path to compressed file
Options:
--output <path>              Output WAV (default: <input>.wav)
--buffer <samples>           Write batch size (default 8192, max 65536)
--sbr-weights <w0> <w1>      Override SBR weights
--help, -h                   Show this help
)";
}

static bool parse_decoder_args(int argc, char** argv, DecoderOptions& o)
{
    if (argc < 2) return false;
    o.input_wha = argv[1];
    if (o.input_wha == "--help" || o.input_wha == "-h") { print_help_decoder(); return false; }
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> bool {
            if (i + 1 >= argc) { std::cerr << "Option " << what << " requires a value\n"; return false; }
            return true;
        };
        if      (a == "--help" || a == "-h") { print_help_decoder(); return false; }
        else if (a == "--output")      { if(!need("--output")) return false; o.output_wav = argv[++i]; }
        else if (a == "--buffer")      { if(!need("--buffer")) return false; o.buffer = std::stoul(argv[++i]); if (o.buffer == 0) o.buffer = 8192; if (o.buffer > 65536) o.buffer = 65536; }
        else if (a == "--sbr-weights") {
            if (i + 2 >= argc) { std::cerr << "--sbr-weights needs 2 values\n"; return false; }
            o.sbr_w0 = std::stof(argv[++i]);
            o.sbr_w1 = std::stof(argv[++i]);
            o.sbr_override = true;
        }
        else { std::cerr << "Unknown option: " << a << "\n"; return false; }
    }
    if (o.output_wav.empty()) {
        o.output_wav = o.input_wha;
        if (o.output_wav.size() > 4) o.output_wav.resize(o.output_wav.size() - 4);
        o.output_wav += ".wav";
    }
    return true;
}

void decompress_wha_to_wav(const DecoderOptions& opts)
{
    build_huffman_luts();

    std::ifstream f(opts.input_wha, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + opts.input_wha);

    auto magic = read_n(f, 4);
    if (magic.size() != 4 || std::string((char*)magic.data(), 4) != "WHA1")
        throw std::runtime_error("Not a WHA container");

    uint8_t version = read_u8(f);
    if (version != 30)
        throw std::runtime_error("Unsupported container version (only v30 supported)");

    uint8_t coding_flags    = read_u8(f);
    float   container_sbr_w0 = read_f32(f);
    float   container_sbr_w1 = read_f32(f);

    uint32_t sr = read_u32(f);
    uint8_t  num_channels = read_u8(f);
    uint32_t block_count = read_u32(f);
    float    target_kbps = read_f32(f);
    uint8_t  block_format_version = read_u8(f);
    if (block_format_version != 36)
        throw std::runtime_error("Unsupported block format version (only v36 supported)");

    uint32_t total_frames = read_u32(f);

    bool speed_mode        = (coding_flags & 0x01) != 0;
    bool decoder_enable_ms  = (coding_flags & 0x02) == 0;
    bool decoder_enable_is  = (coding_flags & 0x04) == 0;
    bool decoder_enable_sbr = (coding_flags & 0x08) == 0;
    int  decoder_block_mode = (coding_flags >> 4) & 0x03;

    if (speed_mode)
        std::cout << ">>> Container indicates SPEED MODE: post-processing will be skipped <<<\n";
    if (!decoder_enable_ms)  std::cout << ">>> Container: MS disabled at encode time <<<\n";
    if (!decoder_enable_is)  std::cout << ">>> Container: IS disabled at encode time <<<\n";
    if (!decoder_enable_sbr) std::cout << ">>> Container: SBR disabled at encode time <<<\n";
    if (decoder_block_mode == 1) std::cout << ">>> Container: long blocks only <<<\n";
    if (decoder_block_mode == 2) std::cout << ">>> Container: short blocks only <<<\n";

    if (opts.sbr_override) {
        SBR_W0 = opts.sbr_w0;
        SBR_W1 = opts.sbr_w1;
    } else {
        SBR_W0 = container_sbr_w0;
        SBR_W1 = container_sbr_w1;
    }
    std::cout << "SBR Shape Weights: W0=" << SBR_W0
              << ", W1=" << SBR_W1
              << " (Contrast: " << (20.0f * std::log10(SBR_W1 / SBR_W0)) << " dB)\n";

    float per_channel_kbps = target_kbps / num_channels;
    float overlap_factor = (per_channel_kbps <= 0.0f) ? 2.0f/3.0f : 1.0f;

    float step_multiplier = (per_channel_kbps < 96.0f) ? 16.0f : 1.0f;

    drwav_data_format fmt;
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = num_channels;
    fmt.sampleRate    = sr;
    fmt.bitsPerSample = 32;
    drwav wav;
    if (!drwav_init_file_write(&wav, opts.output_wav.c_str(), &fmt, nullptr))
        throw std::runtime_error("dr_wav failed to init");

    std::vector<WptHuffSlotDecoder> wpt_decoders(WPT_COEFF_HUFF_SLOTS);
    for (int s = 0; s < WPT_COEFF_HUFF_SLOTS; ++s)
        build_wpt_decoder(wpt_decoders[s], s);

    PWPT wpt;
    ModeState long_state, short_state;
    bool stereo = (num_channels == 2);

    std::unique_ptr<PowCodec> pow_dec;
    if (!speed_mode)
        pow_dec = std::make_unique<PowCodec>((int)num_channels, 0.85f, false,
                                             target_kbps / float(num_channels));

    OverlapAddState ola;
    ola.init(num_channels, opts.buffer);

    std::vector<float> write_buf, decoded;
    write_buf.reserve(opts.buffer * num_channels);
    decoded.reserve(opts.buffer * num_channels);

    std::unique_ptr<SpectralFloorPostProcessor> postproc;
    if (!speed_mode && shouldApplySpectralFloor((int)num_channels, target_kbps)) {
        SpectralFloorPostFX::Config pp_cfg;
        postproc = std::make_unique<SpectralFloorPostProcessor>((int)num_channels, pp_cfg);
    }

    std::vector<int> band_shapes_long  = compute_band_shapes(2048, 5);
    std::vector<int> band_shapes_short = compute_band_shapes(1024, 4);

    struct WindowCache { std::vector<float> window, window_sq; int bs = 0, lo = -1, ro = -1; };
    WindowCache win_long, win_short;

    std::vector<uint8_t> blk, mode_ms, active0, active1;
    std::vector<std::array<float,4>> is_r;
    std::vector<std::array<bool,4>>  is_inv;
    std::vector<bool> is_use_segmented;
    std::vector<float> steps0, steps1;
    std::vector<std::vector<int32_t>> quant0, quant1;
    std::vector<std::vector<float>>   ch0_bands, ch1_bands;
    quant0.reserve(32); quant1.reserve(32);
    ch0_bands.reserve(32); ch1_bands.reserve(32);

    std::vector<std::vector<float>> recon_chs(num_channels);
    uint64_t frames_written = 0;
    SBRDecodeResult sbr;
    std::vector<int> idx_list_scratch;

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
            if (n_out > 0) { data_to_write = postproc->out_data(); frames_to_write = n_out; }
            else return;
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
        uint16_t len16;
        f.read((char*)&len16, 2);
        size_t block_len = len16;

        blk.resize(block_len);
        f.read((char*)blk.data(), block_len);
        if ((size_t)f.gcount() < block_len)
            throw std::runtime_error("Block truncated");

        if (blk.size() < 3) throw std::runtime_error("Block too small");

        uint16_t payload_len_field;
        memcpy(&payload_len_field, &blk[1], 2);
        bool     use_long_block = (payload_len_field & 0x8000) == 0;
        uint16_t payload_len    = payload_len_field & 0x7FFF;
        uint8_t  k_scale_byte   = blk[0];
        size_t   ptr = 3;

        int level       = use_long_block ? 5 : 4;
        int block_size  = use_long_block ? 2048 : 1024;
        int right_overlap = use_long_block ? (int)(96*overlap_factor) : (int)(48*overlap_factor);
        int left_overlap  = (prev_block_size == 2048) ? (int)(96*overlap_factor) : (int)(48*overlap_factor);
        if (bi == 0) left_overlap = 0;
        int hop = block_size - right_overlap;
        int total_bands = 1 << level;
        const std::vector<int>& band_shapes = use_long_block ? band_shapes_long : band_shapes_short;
        int expected_band_count = total_bands;

        ModeState& pred_state = use_long_block ? long_state : short_state;
        pred_state.ensure(expected_band_count);
        bool can_state_pred = false;
        if (pred_state.ready && bi > pred_state.last_bi) {
            uint32_t age = bi - pred_state.last_bi;
            if (age <= SAME_MODE_PRED_MAX_AGE) can_state_pred = true;
        }

        WindowCache& wc = use_long_block ? win_long : win_short;
        if (wc.bs != block_size || wc.lo != left_overlap || wc.ro != right_overlap) {
            make_window(wc.window, wc.window_sq, block_size, left_overlap, right_overlap);
            wc.bs = block_size; wc.lo = left_overlap; wc.ro = right_overlap;
        }
        const std::vector<float>& window    = wc.window;
        const std::vector<float>& window_sq = wc.window_sq;

        int  k_scale0 = k_scale_byte & 0x07;
        int  k_scale1 = stereo ? ((k_scale_byte >> 3) & 0x07) : 0;
        bool block_is_full = (k_scale_byte & 0x80) != 0;
        bool use_scale_pred = true;  

        int sbr_end = (decoder_enable_sbr && block_is_full)
                      ? (14 * expected_band_count / 16) : 0;

        mode_ms.resize(expected_band_count);
        active0.resize(expected_band_count);
        if (stereo) active1.resize(expected_band_count);

        if (ptr + payload_len > blk.size()) throw std::runtime_error("Payload truncated");
        BitReaderMSB payload_reader(blk.data() + ptr, payload_len);
        ptr += payload_len;

        for (int i = 0; i < expected_band_count; i += 4) {
            int nibble = huffman_decode_mode_ms(payload_reader);
            for (int s = 0; s < 4; ++s) {
                int idx = i + s;
                if (idx < expected_band_count) mode_ms[idx] = (nibble >> (3 - s)) & 1;
            }
        }
        for (int i = 0; i < expected_band_count; i += 4) {
            int nibble = huffman_decode_active(payload_reader);
            for (int s = 0; s < 4; ++s) {
                int idx = i + s;
                if (idx < expected_band_count) active0[idx] = (nibble >> (3 - s)) & 1;
            }
        }
        if (stereo) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            auto& idx_list = idx_list_scratch;
            idx_list.clear();
            for (int i = 0; i < expected_band_count; ++i) {
                bool is_is_band = decoder_enable_is && (i >= is_start) && mode_ms[i];
                if (!is_is_band) idx_list.push_back(i);
            }
            int non_is_count = (int)idx_list.size();
            std::fill_n(active1.begin(), expected_band_count, (uint8_t)0);
            int bit_idx = 0;
            int n_nibbles = (non_is_count + 3) / 4;
            for (int j = 0; j < n_nibbles; ++j) {
                int nibble = huffman_decode_active(payload_reader);
                for (int s = 0; s < 4; ++s) {
                    if (bit_idx < non_is_count) {
                        active1[idx_list[bit_idx]] = (nibble >> (3 - s)) & 1;
                        bit_idx++;
                    }
                }
            }
        }

        steps0.assign(expected_band_count, 0.0f);
        steps1.assign(expected_band_count, 0.0f);

        auto decode_scales_batch = [&](
            std::vector<float>& steps,
            const std::vector<uint8_t>& active,
            int k_scale,
            std::vector<uint32_t>& prev_scale,
            const std::vector<uint8_t>& prev_active)
        {
            int active_count = 0;
            for (int i = 0; i < expected_band_count; ++i) if (active[i]) active_count++;
            if (active_count == 0) return;

            auto all_scales = rice_decode(payload_reader, (size_t)active_count, k_scale);
            if ((int)all_scales.size() != active_count) throw std::runtime_error("Scale count mismatch");

            int scale_idx = 0;
            for (int i = 0; i < expected_band_count; ++i) {
                if (!active[i]) continue;
                uint32_t val = all_scales[scale_idx++];
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

        decode_scales_batch(steps0, active0, k_scale0, pred_state.prev_scale0, pred_state.prev_active0);
        if (stereo)
            decode_scales_batch(steps1, active1, k_scale1, pred_state.prev_scale1, pred_state.prev_active1);

        if (step_multiplier != 1.0f) {
            for (float& s : steps0) s *= step_multiplier;
            if (stereo) {
                for (float& s : steps1) s *= step_multiplier;
            }
        }

        int active_count0 = 0, active_count1 = 0;
        for (int i = 0; i < expected_band_count; ++i) {
            if (active0[i]) active_count0++;
            if (stereo && active1[i]) active_count1++;
        }

        int kidx_slot0 = (int)payload_reader.read_bits(3);
        int kidx_slot1 = stereo ? (int)payload_reader.read_bits(3) : 0;
        std::vector<uint32_t> k_sub_vals0 = decode_kidx_huff_vec(payload_reader, (size_t)active_count0, kidx_slot0);
        std::vector<uint32_t> k_sub_vals1;
        if (stereo) k_sub_vals1 = decode_kidx_huff_vec(payload_reader, (size_t)active_count1, kidx_slot1);

        quant0.resize(expected_band_count);
        quant1.resize(expected_band_count);
        ch0_bands.resize(expected_band_count);
        ch1_bands.resize(expected_band_count);

        int k_idx0 = 0;
        for (int i = 0; i < expected_band_count; ++i) {
            quant0[i].clear();
            if (!active0[i]) continue;
            if (k_idx0 >= (int)k_sub_vals0.size())
                throw std::runtime_error("k_sub_vals0 OOB");
            uint32_t sym = k_sub_vals0[k_idx0++];

            bool is_huff_small = false, is_huff_large = false, is_wpt = false;
            int  wpt_slot = 0;
            int  k_rice = 0;
            if      (sym == 0)   is_huff_small = true;
            else if (sym == 1)   is_huff_large = true;
            else if (sym < 10) { is_wpt = true; wpt_slot = WPT_HUFF_SLOT_MAP[sym - 2]; }
            else                 k_rice = (int)sym - 10;

            if (is_huff_small) {
                quant0[i].resize(band_shapes[i]);
                decode_huffman_small_vec(payload_reader, quant0[i].data(), band_shapes[i]);
            } else if (is_huff_large) {
                quant0[i].resize(band_shapes[i]);
                decode_huffman_large_vec(payload_reader, quant0[i].data(), band_shapes[i]);
            } else if (is_wpt) {
                quant0[i].resize(band_shapes[i]);
                int32_t* out = quant0[i].data();
                const WptHuffSlotDecoder& dec = wpt_decoders[wpt_slot];
                for (int j = 0; j < band_shapes[i]; ++j)
                    out[j] = decode_wpt_huff(payload_reader, dec);
            } else {
                if (k_rice < 0) throw std::runtime_error("Invalid k_rice");
                if (k_rice > 7) k_rice = 7;
                auto uvals = rice_decode(payload_reader, (size_t)band_shapes[i], k_rice);
                if ((int)uvals.size() != band_shapes[i])
                    throw std::runtime_error("uvals size mismatch ch0");
                quant0[i].resize(uvals.size());
                for (size_t j = 0; j < uvals.size(); ++j)
                    quant0[i][j] = zigzag_decode(uvals[j]);
            }
        }

        if (stereo) {
            int k_idx1 = 0;
            for (int i = 0; i < expected_band_count; ++i) {
                quant1[i].clear();
                if (!active1[i]) continue;
                if (k_idx1 >= (int)k_sub_vals1.size())
                    throw std::runtime_error("k_sub_vals1 OOB");
                uint32_t sym = k_sub_vals1[k_idx1++];

                bool is_huff_small = false, is_huff_large = false, is_wpt = false;
                int  wpt_slot = 0;
                int  k_rice = 0;
                if      (sym == 0)   is_huff_small = true;
                else if (sym == 1)   is_huff_large = true;
                else if (sym < 10) { is_wpt = true; wpt_slot = WPT_HUFF_SLOT_MAP[sym - 2]; }
                else                 k_rice = (int)sym - 10;

                if (is_huff_small) {
                    quant1[i].resize(band_shapes[i]);
                    decode_huffman_small_vec(payload_reader, quant1[i].data(), band_shapes[i]);
                } else if (is_huff_large) {
                    quant1[i].resize(band_shapes[i]);
                    decode_huffman_large_vec(payload_reader, quant1[i].data(), band_shapes[i]);
                } else if (is_wpt) {
                    quant1[i].resize(band_shapes[i]);
                    int32_t* out = quant1[i].data();
                    const WptHuffSlotDecoder& dec = wpt_decoders[wpt_slot];
                    for (int j = 0; j < band_shapes[i]; ++j)
                        out[j] = decode_wpt_huff(payload_reader, dec);
                } else {
                    if (k_rice < 0) throw std::runtime_error("Invalid k_rice");
                    if (k_rice > 7) k_rice = 7;
                    auto uvals = rice_decode(payload_reader, (size_t)band_shapes[i], k_rice);
                    if ((int)uvals.size() != band_shapes[i])
                        throw std::runtime_error("uvals size mismatch ch1");
                    quant1[i].resize(uvals.size());
                    for (size_t j = 0; j < uvals.size(); ++j)
                        quant1[i][j] = zigzag_decode(uvals[j]);
                }
            }
        }

        is_r.resize(expected_band_count);
        for (auto& a : is_r) a.fill(0.5f);
        is_inv.assign(expected_band_count, std::array<bool,4>{});
        for (auto& a : is_inv) a.fill(false);
        is_use_segmented.assign(expected_band_count, false);

        bool is_used = false;
        if (decoder_enable_is && stereo && target_kbps < 510.0f) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            if (is_start < expected_band_count) {
                auto& is_indices = idx_list_scratch;
                is_indices.clear();
                for (int i = is_start; i < expected_band_count; ++i)
                    if (mode_ms[i]) is_indices.push_back(i);

                if (!is_indices.empty()) {
                    is_used = true;
                    int segmented_threshold = get_is_segmented_threshold(target_kbps, total_bands);

                    for (int idx : is_indices) {
                        if (idx < segmented_threshold) {
                            int nibble = huffman_decode_is_inv(payload_reader);
                            for (int s = 0; s < 4; ++s)
                                is_inv[idx][s] = (((nibble >> (3 - s)) & 1) != 0);
                        } else {
                            is_inv[idx][0] = (payload_reader.read_bits(1) != 0);
                        }
                    }

                    int table_idx = (int)payload_reader.read_bits(3);
                    if (table_idx >= IS_TBL_SLOTS)
                        throw std::runtime_error("Invalid IS table index");
                    for (int idx : is_indices) {
                        int bits = get_r_bits(idx);
                        if (idx < segmented_threshold) {
                            is_use_segmented[idx] = true;
                            for (int s = 0; s < 4; ++s) {
                                int sym = decode_is_coeff_with_table(payload_reader, table_idx);
                                is_r[idx][s] = dequantize_r_centered(sym, bits);
                            }
                        } else {
                            is_use_segmented[idx] = false;
                            int sym = decode_is_coeff_with_table(payload_reader, table_idx);
                            is_r[idx].fill(dequantize_r_centered(sym, bits));
                        }
                    }
                }
            }
        }

        sbr.indices.clear();
        if (decoder_enable_sbr) {
            sbr_decode_v23(sbr, payload_reader, active0, sbr_end, expected_band_count,
                           can_state_pred, pred_state);
        }

        for (int i = 0; i < expected_band_count; ++i) {
            ch0_bands[i].clear(); ch0_bands[i].resize(band_shapes[i], 0.0f);
            if (active0[i]) dequantize_band(quant0[i], steps0[i], ch0_bands[i]); 
            
            if (stereo) {
                ch1_bands[i].clear(); ch1_bands[i].resize(band_shapes[i], 0.0f);
                if (active1[i]) dequantize_band(quant1[i], steps1[i], ch1_bands[i]); 
            }
        }

        if (sbr_end > 0) {
            sbr_synthesize(ch0_bands, active0, band_shapes, sbr, sbr_end, bi,
                           SBR_W0, SBR_W1, window, block_size, left_overlap, right_overlap);
        }

        if (stereo) {
            int is_start = get_is_start_band(target_kbps, total_bands);
            for (int i = 0; i < expected_band_count; ++i) {
                if (mode_ms[i]) {
                    if (decoder_enable_is && is_used && i >= is_start && is_start < expected_band_count) {
                        apply_is_inplace(ch0_bands[i], ch1_bands[i], is_r[i], is_inv[i], is_use_segmented[i]);
                    } else {
                        ms_to_lr_inplace(ch0_bands[i], ch1_bands[i]);
                    }
                } else {
                    if (decoder_enable_is && is_used && i >= is_start && is_start < expected_band_count) {
                        ms_to_lr_inplace(ch0_bands[i], ch1_bands[i]);
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

        if (write_buf.size() >= opts.buffer * num_channels) flush_write_buffer();

        pred_state.prev_active0.assign(expected_band_count, 0);
        for (int i = 0; i < expected_band_count; ++i) if (active0[i]) pred_state.prev_active0[i] = 1;
        if (stereo) {
            pred_state.prev_active1.assign(expected_band_count, 0);
            for (int i = 0; i < expected_band_count; ++i) if (active1[i]) pred_state.prev_active1[i] = 1;
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
        pred_state.ready = true;
        pred_state.last_bi = bi;

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
    std::cout << "Decompressed: " << opts.output_wav
              << " (frames=" << ola.output_pos
              << ", sr=" << sr
              << ", ch=" << (int)num_channels << ")" << std::endl;
}

int main(int argc, char** argv)
{
    DecoderOptions opts;
    if (!parse_decoder_args(argc, argv, opts)) return 1;
    try {
        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);
        decompress_wha_to_wav(opts);
        getrusage(RUSAGE_SELF, &usage_after);
        double user_time_sec =
            (usage_after.ru_utime.tv_sec - usage_before.ru_utime.tv_sec) +
            (usage_after.ru_utime.tv_usec - usage_before.ru_utime.tv_usec) / 1000000.0;

        drwav wav_info;
        if (drwav_init_file(&wav_info, opts.output_wav.c_str(), nullptr)) {
            double duration = (double)wav_info.totalPCMFrameCount / (double)wav_info.sampleRate;
            double speed    = (user_time_sec > 0.0) ? (duration / user_time_sec) : 0.0;
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