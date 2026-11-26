#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline float powf(float x, float y) {
    if (x <= 0.0f) return 0.0f;
    return std::exp2f(y * std::log2f(x));
}

class PowCodec {
public:
    static constexpr int NFFT  = 2048;
    static constexpr int HOP   = 1024;
    static constexpr int FFT_BATCH = 1;
    static constexpr int NBINS = NFFT / 2 + 1;
    static constexpr int PAD   = NFFT / 2;

    PowCodec(int num_channels, float gamma, float kbps_per_channel)
       : num_channels_(num_channels), gamma_(gamma), 
         kbps_per_channel_(kbps_per_channel),
         channels_(num_channels), ch_out_(num_channels)
     {
         if (num_channels_ <= 0) throw std::runtime_error("PowCodec: bad channels");
         if (std::fabs(gamma_ - 0.85f) > 1e-6f)
             throw std::runtime_error("PowCodec: this optimized version requires gamma = 0.85");
         
         if (kbps_per_channel_ < 48.0f) {
             max_mag_multiplier_ = 16.0f;
         } else if (kbps_per_channel_ < 64.0f) {
             max_mag_multiplier_ = 4.0f;
         } else {
             max_mag_multiplier_ = 1.0f;
         }

         in_     = fftwf_alloc_real(NFFT);
         out_    = fftwf_alloc_complex(NBINS);
         
         if (!in_ || !out_) throw std::runtime_error("PowCodec: FFTW alloc failed");
         
         const char* wisdom_file = "fftw_wisdom.dat";
         fftwf_import_wisdom_from_filename(wisdom_file);
         r2c_ = fftwf_plan_dft_r2c_1d(NFFT, in_, out_, FFTW_MEASURE);
         c2r_ = fftwf_plan_dft_c2r_1d(NFFT, out_, in_, FFTW_MEASURE);
         
         if (!r2c_ || !c2r_) throw std::runtime_error("PowCodec: FFTW plan failed");
         fftwf_export_wisdom_to_filename(wisdom_file);
         
         win_.resize(NFFT);
         win_sq_.resize(NFFT);
         for (int i = 0; i < NFFT; ++i) {
             float s = std::sin((float)M_PI * (float)i / (float)NFFT);
             win_[i] = std::sqrt(std::sin((float)(M_PI / 2.0) * s * s));
             win_sq_[i] = win_[i] * win_[i];
         }

         for (auto& ch : channels_) {
             ch.queue.reserve(1024 * 1024); 
             ch.acc.reserve(1024 * 1024);
         }
     }

    ~PowCodec() {
        if (r2c_) fftwf_destroy_plan(r2c_);
        if (c2r_) fftwf_destroy_plan(c2r_);
        fftwf_free(in_); fftwf_free(out_);
    }

    PowCodec(const PowCodec&) = delete;
    PowCodec& operator=(const PowCodec&) = delete;

    void process(const float* in, size_t frames, std::vector<float>& out) {
        if (frames == 0) return;
        out.reserve(out.size() + frames * num_channels_);
        for (int c = 0; c < num_channels_; ++c) {
            size_t old_size = channels_[c].queue.size();
            channels_[c].queue.resize(old_size + frames);
            for (size_t f = 0; f < frames; ++f) {
                channels_[c].queue[old_size + f] = in[f * num_channels_ + c];
            }
            channels_[c].total_in += frames;
        }
        for (auto& channel_out : ch_out_)
            channel_out.clear();
        processChannels(false);
        const size_t n_out = ch_out_.empty() ? 0 : ch_out_[0].size();
        size_t out_old_size = out.size();
        out.resize(out_old_size + n_out * num_channels_);
        for (size_t f = 0; f < n_out; ++f) {
            for (int c = 0; c < num_channels_; ++c) {
                out[out_old_size + f * num_channels_ + c] = ch_out_[c][f];
            }
        }
    }

    void flush(std::vector<float>& out) {
        for (auto& channel_out : ch_out_)
            channel_out.clear();
        processChannels(true);
        const size_t n_out = ch_out_.empty() ? 0 : ch_out_[0].size();
        size_t out_old_size = out.size();
        out.resize(out_old_size + n_out * num_channels_);
        for (size_t f = 0; f < n_out; ++f) {
            for (int c = 0; c < num_channels_; ++c) {
                out[out_old_size + f * num_channels_ + c] = ch_out_[c][f];
            }
        }
    }

    void reset() {
        for (auto& cs : channels_) cs = ChannelState{};
    }

private:
    struct ChannelState {
        std::vector<float> queue;
        std::vector<float> acc;
        int64_t queue_start = 0;
        int64_t acc_start   = 0;
        int64_t next_frame  = 0;
        int64_t emitted     = 0;
        int64_t total_in    = 0;
        bool seeded      = false;
        bool tail_padded = false;
    };

    int num_channels_;
    float gamma_;
    float kbps_per_channel_; 
    float max_mag_multiplier_ = 1.0f; 
    
    std::vector<ChannelState> channels_;
    std::vector<std::vector<float>> ch_out_; 
    fftwf_plan r2c_{};
    fftwf_plan c2r_{};
    float* in_ = nullptr;
    fftwf_complex* out_ = nullptr;
    std::vector<float> win_;
    std::vector<float> win_sq_;

    void processFrame(const float* frame) {
        constexpr float EPS = 1e-20f;
        constexpr float EPS_SQ = EPS * EPS;
        constexpr float SCALE_EXPONENT = (0.85f - 1.0f) * 0.5f;
        for (int i = 0; i < NFFT; ++i)
            in_[i] = frame[i] * win_[i];
        fftwf_execute(r2c_);

        float max_mag_sq = 0.0f;
        for (int i = 0; i < NBINS; ++i) {
            const float re = out_[i][0];
            const float im = out_[i][1];
            const float mag_sq = re * re + im * im;
            if (mag_sq > max_mag_sq)
                max_mag_sq = mag_sq;
        }

        float max_mag = EPS;
        if (max_mag_sq > EPS_SQ)
            max_mag = std::sqrt(max_mag_sq);
        max_mag *= max_mag_multiplier_;
        const float inv_max_sq = 1.0f / (max_mag * max_mag);

        for (int i = 0; i < NBINS; ++i) {
            const float re = out_[i][0];
            const float im = out_[i][1];
            const float mag_sq = re * re + im * im;
            if (mag_sq > EPS_SQ) {
                const float normalized_sq = mag_sq * inv_max_sq;
                const float scale = powf(normalized_sq, SCALE_EXPONENT);
                out_[i][0] = re * scale;
                out_[i][1] = im * scale;
            } else {
                out_[i][0] = 0.0f;
                out_[i][1] = 0.0f;
            }
        }
        fftwf_execute(c2r_);
    }

    void ensureAcc(ChannelState& cs, int64_t end_pos) {
        const int64_t cur_end = cs.acc_start + (int64_t)cs.acc.size();
        if (end_pos <= cur_end) return;
        const size_t new_size = (size_t)(end_pos - cs.acc_start);
        cs.acc.resize(new_size, 0.0f);
    }

    float overlapWeight(int64_t position) const {
        const size_t phase = static_cast<size_t>(position % HOP);
        if (position < HOP)
            return win_sq_[phase];
        return win_sq_[phase + HOP] + win_sq_[phase];
    }

    void prepareChannel(ChannelState& cs, bool final) {
        if (!cs.seeded) {
            if (cs.queue.size() < (size_t)PAD + 1 && !final) return;
            const int have = (int)cs.queue.size();
            std::vector<float> new_queue;
            new_queue.reserve(PAD + have);
            if (have >= PAD + 1) {
                for (int i = 0; i < PAD; ++i) new_queue.push_back(cs.queue[PAD - i]); 
            } else {
                for (int i = 0; i < PAD; ++i) new_queue.push_back(0.0f);
            }
            for (int i = 0; i < have; ++i) new_queue.push_back(cs.queue[i]);
            cs.queue = std::move(new_queue);
            cs.queue_start = 0;
            cs.next_frame = 0;
            cs.seeded = true;
        }

        if (final && !cs.tail_padded) {
            cs.tail_padded = true;
            const int64_t real_end = (int64_t)PAD + cs.total_in;
            if ((int64_t)cs.queue.size() < real_end) cs.queue.resize((size_t)real_end, 0.0f);
            if (cs.total_in >= (int64_t)PAD + 2) {
                for (int i = 0; i < PAD; ++i) {
                    const int64_t idx = real_end - 2 - i;
                    float v = (idx >= 0 && idx < (int64_t)cs.queue.size()) ? cs.queue[(size_t)idx] : 0.0f;
                    cs.queue.push_back(v);
                }
            } else {
                cs.queue.resize(cs.queue.size() + (size_t)PAD, 0.0f);
            }
        }
    }

    void processChannels(bool final) {
        for (auto& cs : channels_)
            prepareChannel(cs, final);

        while (true) {
            const int64_t next_frame = channels_[0].next_frame;
            const int64_t input_end = PAD + channels_[0].total_in;
            if (final && next_frame >= input_end)
                break;

            int batch_frames = 0;
            while (batch_frames < FFT_BATCH) {
                const int64_t frame_pos = next_frame +
                    static_cast<int64_t>(batch_frames) * HOP;
                if (final && frame_pos >= input_end)
                    break;

                bool frame_available = true;
                for (int channel = 0; channel < num_channels_; ++channel) {
                    ChannelState& cs = channels_[channel];
                    if (cs.next_frame != next_frame ||
                        cs.total_in != channels_[0].total_in)
                        throw std::runtime_error("POW channel frame states diverged");

                    int64_t q_idx = frame_pos - cs.queue_start;
                    if (q_idx < 0) {
                        cs.queue_start = frame_pos;
                        q_idx = 0;
                    }
                    const int64_t avail =
                        static_cast<int64_t>(cs.queue.size()) - q_idx;
                    if (avail < NFFT) {
                        if (!final) {
                            frame_available = false;
                            break;
                        }
                        cs.queue.resize(static_cast<size_t>(q_idx + NFFT), 0.0f);
                    }

                }
                if (!frame_available)
                    break;
                ++batch_frames;
            }

            if (batch_frames == 0 || (!final && batch_frames < FFT_BATCH))
                break;

            constexpr float norm = 1.0f / static_cast<float>(NFFT);
            for (int channel = 0; channel < num_channels_; ++channel) {
                ChannelState& cs = channels_[channel];
                for (int frame = 0; frame < batch_frames; ++frame) {
                    const int64_t frame_pos = next_frame +
                        static_cast<int64_t>(frame) * HOP;
                    const int64_t q_idx = frame_pos - cs.queue_start;
                    processFrame(cs.queue.data() + q_idx);
                    ensureAcc(cs, frame_pos + NFFT);
                    const int64_t a_idx = frame_pos - cs.acc_start;
                    float* acc_ptr = cs.acc.data() + a_idx;
                    for (int i = 0; i < NFFT; ++i)
                        acc_ptr[i] += in_[i] * norm * win_[i];
                }
                cs.next_frame += static_cast<int64_t>(batch_frames) * HOP;
            }
        }

        for (int channel = 0; channel < num_channels_; ++channel) {
            ChannelState& cs = channels_[channel];
            std::vector<float>& out = ch_out_[channel];
            const int64_t input_end = PAD + cs.total_in;
            const int64_t finalized = cs.next_frame;
            int64_t emit_end = std::min(finalized, input_end);
            if (emit_end < cs.emitted)
                emit_end = cs.emitted;
            const int64_t emit_start =
                std::min(emit_end, std::max<int64_t>(cs.emitted, PAD));
            const size_t output_start = out.size();
            const size_t output_count = static_cast<size_t>(emit_end - emit_start);
            out.resize(output_start + output_count);
            size_t output_idx = output_start;
            for (int64_t p = emit_start; p < emit_end; ++p) {
                const int64_t idx = p - cs.acc_start;
                out[output_idx++] =
                    cs.acc[static_cast<size_t>(idx)] / overlapWeight(p);
            }
            cs.emitted = emit_end;

            int64_t q_drop = cs.next_frame - cs.queue_start;
            if (q_drop > 4096 && q_drop < static_cast<int64_t>(cs.queue.size())) {
                const size_t drop = static_cast<size_t>(q_drop);
                const size_t remaining = cs.queue.size() - drop;
                std::memmove(cs.queue.data(), cs.queue.data() + drop,
                             remaining * sizeof(float));
                cs.queue.resize(remaining);
                cs.queue_start += static_cast<int64_t>(drop);
            }

            const int64_t a_drop = cs.emitted - cs.acc_start;
            if (a_drop > 4096 && a_drop < static_cast<int64_t>(cs.acc.size())) {
                const size_t drop = static_cast<size_t>(a_drop);
                const size_t remaining = cs.acc.size() - drop;
                std::memmove(cs.acc.data(), cs.acc.data() + drop,
                             remaining * sizeof(float));
                cs.acc.resize(remaining);
                cs.acc_start += static_cast<int64_t>(drop);
            }
        }
    }

};