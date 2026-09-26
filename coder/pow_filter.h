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

class PowCodec {
public:
    static constexpr int NFFT  = 2048;
    static constexpr int HOP   = 1024;
    static constexpr int NBINS = NFFT / 2 + 1;
    static constexpr int PAD   = NFFT / 2;

    PowCodec(int num_channels, float gamma, bool encode, float kbps_per_channel)
         : num_channels_(num_channels), gamma_(gamma), encode_(encode), 
           kbps_per_channel_(kbps_per_channel),
           channels_(num_channels), ch_out_(num_channels)
     {
         if (num_channels_ <= 0) throw std::runtime_error("PowCodec: bad channels");
         if (std::fabs(gamma_ - 0.85f) > 1e-6f)
             throw std::runtime_error("PowCodec: this optimized version requires gamma = 0.85");
         
         in_     = fftwf_alloc_real(NFFT);
         out_c_  = fftwf_alloc_complex(NBINS);
         in_c2r_ = fftwf_alloc_complex(NBINS);
         out_    = fftwf_alloc_real(NFFT);
         
         if (!in_ || !out_c_ || !in_c2r_ || !out_) throw std::runtime_error("PowCodec: FFTW alloc failed");
         
         const char* wisdom_file = "fftw_wisdom.dat";
         fftwf_import_wisdom_from_filename(wisdom_file);
         
         r2c_ = fftwf_plan_dft_r2c_1d(NFFT, in_, out_c_, FFTW_MEASURE);
         c2r_ = fftwf_plan_dft_c2r_1d(NFFT, in_c2r_, out_, FFTW_MEASURE);
         
         if (!r2c_ || !c2r_) throw std::runtime_error("PowCodec: FFTW plan failed");
         
         fftwf_export_wisdom_to_filename(wisdom_file);
         
         win_.resize(NFFT);
         win_sq_.resize(NFFT);
         for (int i = 0; i < NFFT; ++i) {
             float s = std::sin((float)M_PI * (float)i / (float)NFFT);
             win_[i] = std::sqrt(std::sin((float)(M_PI / 2.0) * s * s));
             win_sq_[i] = win_[i] * win_[i];
         }
     }

     ~PowCodec() {
         if (r2c_) fftwf_destroy_plan(r2c_);
         if (c2r_) fftwf_destroy_plan(c2r_);
         fftwf_free(in_); fftwf_free(out_c_); fftwf_free(in_c2r_); fftwf_free(out_);
     }

     PowCodec(const PowCodec&) = delete;
     PowCodec& operator=(const PowCodec&) = delete;

     void process(const float* in, size_t frames, std::vector<float>& out) {
         if (frames == 0) return;
         out.reserve(out.size() + frames * num_channels_);
         for (size_t f = 0; f < frames; ++f) {
             for (int c = 0; c < num_channels_; ++c) {
                 channels_[c].queue.push_back(in[f * num_channels_ + c]);
                 channels_[c].total_in++;
             }
         }
         for (int c = 0; c < num_channels_; ++c) {
             ch_out_[c].clear();
             processChannel(channels_[c], ch_out_[c], false);
         }
         const size_t n_out = ch_out_.empty() ? 0 : ch_out_[0].size();
         for (size_t f = 0; f < n_out; ++f) {
             for (int c = 0; c < num_channels_; ++c) {
                 out.push_back(ch_out_[c][f]);
             }
         }
     }

     void flush(std::vector<float>& out) {
         for (int c = 0; c < num_channels_; ++c) {
             ch_out_[c].clear();
             processChannel(channels_[c], ch_out_[c], true);
         }
         const size_t n_out = ch_out_.empty() ? 0 : ch_out_[0].size();
         out.reserve(out.size() + n_out * num_channels_);
         for (size_t f = 0; f < n_out; ++f) {
             for (int c = 0; c < num_channels_; ++c) {
                 out.push_back(ch_out_[c][f]);
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
        std::vector<float> wacc;
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
    bool encode_;
    float kbps_per_channel_; 
    std::vector<ChannelState> channels_;
    std::vector<std::vector<float>> ch_out_; 
    
    fftwf_plan r2c_{};
    fftwf_plan c2r_{};
    float* in_ = nullptr;
    fftwf_complex* out_c_ = nullptr;
    fftwf_complex* in_c2r_ = nullptr;
    float* out_ = nullptr;
    std::vector<float> win_;
    std::vector<float> win_sq_;

    void processFrameInPlace(float* frame) {
        for (int i = 0; i < NFFT; ++i) in_[i] = frame[i] * win_[i];
        fftwf_execute(r2c_);
        
        float mag[NBINS];
        float max_mag = 0.0f;
        for (int i = 0; i < NBINS; ++i) {
            const float re = out_c_[i][0];
            const float im = out_c_[i][1];
            const float m = std::sqrt(re * re + im * im);
            mag[i] = m;
            if (m > max_mag) max_mag = m;
        }
        
        if (kbps_per_channel_ < 64.0f) {
            max_mag *= 4.0f;
        }
        if (max_mag > 1e-20f) {
            const float inv_max = 1.0f / max_mag;
            if (encode_) {
                constexpr float GAMMA = 0.85f;
                for (int i = 0; i < NBINS; ++i) {
                    float n = mag[i] * inv_max;
                    n = powf(n, GAMMA);
                    mag[i] = n * max_mag;
                }
            } else {
                constexpr float INV_GAMMA = 1.1764705882352941f;
                for (int i = 0; i < NBINS; ++i) {
                    float n = mag[i] * inv_max;
                    n = powf(n, INV_GAMMA);
                    mag[i] = n * max_mag;
                }

                if (kbps_per_channel_ < 64.0f) {
                    constexpr int NUM_SUBBANDS = 32;
                    constexpr float VALLEY_STRENGTH = 0.5f;
                    constexpr float VALLEY_EXPONENT = 1.0f + VALLEY_STRENGTH;
                    
                    float enhanced[NBINS];
                    const int base_size = NBINS / NUM_SUBBANDS;
                    const int remainder = NBINS % NUM_SUBBANDS;
                    int offset = 0;
                    
                    for (int band = 0; band < NUM_SUBBANDS; ++band) {
                        const int band_size = base_size + (band < remainder ? 1 : 0);
                        const int begin = offset;
                        const int end = begin + band_size;
                        offset = end;
                        
                        if (band_size <= 0) continue;
                        
                        float sum = 0.0f;
                        for (int i = begin; i < end; ++i) {
                            sum += mag[i];
                        }
                        
                        const float level = sum / static_cast<float>(band_size);
                        if (level <= 1e-20f) {
                            for (int i = begin; i < end; ++i) {
                                enhanced[i] = mag[i];
                            }
                            continue;
                        }
                        
                        for (int i = begin; i < end; ++i) {
                            const float x = std::max(mag[i], 0.0f);
                            if (x < level) {
                                const float ratio = x / level;
                                enhanced[i] = level * powf(ratio, VALLEY_EXPONENT);
                            } else {
                                enhanced[i] = x;
                            }
                        }
                    }
                    
                    for (int i = 0; i < NBINS; ++i) {
                        mag[i] = enhanced[i];
                    }
                }
            }
        }
        
        for (int i = 0; i < NBINS; ++i) {
            const float re = out_c_[i][0];
            const float im = out_c_[i][1];
            const float old_mag = std::sqrt(re * re + im * im);
            if (old_mag > 1e-20f) {
                const float scale = mag[i] / old_mag;
                in_c2r_[i][0] = re * scale;
                in_c2r_[i][1] = im * scale;
            } else {
                in_c2r_[i][0] = 0.0f;
                in_c2r_[i][1] = 0.0f;
            }
        }
        
        fftwf_execute(c2r_);
        constexpr float norm = 1.0f / (float)NFFT;
        for (int i = 0; i < NFFT; ++i) frame[i] = out_[i] * norm * win_[i];
    }

    void ensureAcc(ChannelState& cs, int64_t end_pos) {
        const int64_t cur_end = cs.acc_start + (int64_t)cs.acc.size();
        if (end_pos <= cur_end) return;
        const size_t new_size = (size_t)(end_pos - cs.acc_start);
        cs.acc.resize(new_size, 0.0f);
        cs.wacc.resize(new_size, 0.0f);
    }

    void processChannel(ChannelState& cs, std::vector<float>& out, bool final) {
        if (!cs.seeded) {
            if (cs.queue.size() < (size_t)PAD + 1 && !final) return;
            const int have = (int)cs.queue.size();
            std::vector<float> new_queue;
            new_queue.reserve(PAD + have);
            if (have >= PAD + 1) {
                for (int i = 0; i < PAD; ++i) {
                    new_queue.push_back(cs.queue[PAD - i]); 
                }
            } else {
                for (int i = 0; i < PAD; ++i) {
                    new_queue.push_back(0.0f);
                }
            }
            for (int i = 0; i < have; ++i) {
                new_queue.push_back(cs.queue[i]);
            }
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
        
        const int64_t input_end = (int64_t)PAD + cs.total_in;
        while (true) {
            if (final && cs.next_frame >= input_end) break;
            int64_t q_idx = cs.next_frame - cs.queue_start;
            if (q_idx < 0) { cs.queue_start = cs.next_frame; q_idx = 0; }
            int64_t avail = (int64_t)cs.queue.size() - q_idx;
            if (avail < NFFT) {
                if (!final) break;
                if (avail <= 0 && cs.next_frame >= input_end) break;
                cs.queue.resize((size_t)(q_idx + NFFT), 0.0f);
            }
            float frame[NFFT];
            for (int i = 0; i < NFFT; ++i) frame[i] = cs.queue[(size_t)(q_idx + i)];
            processFrameInPlace(frame);
            ensureAcc(cs, cs.next_frame + NFFT);
            const int64_t a_idx = cs.next_frame - cs.acc_start;
            for (int i = 0; i < NFFT; ++i) {
                cs.acc[(size_t)(a_idx + i)] += frame[i];
                cs.wacc[(size_t)(a_idx + i)] += win_sq_[i];
            }
            cs.next_frame += HOP;
        }
        
        const int64_t finalized = cs.next_frame;
        int64_t emit_end = std::min(finalized, input_end);
        if (emit_end < cs.emitted) emit_end = cs.emitted;
        out.reserve(out.size() + (emit_end - cs.emitted));
        for (int64_t p = cs.emitted; p < emit_end; ++p) {
            if (p < PAD) continue;
            const int64_t idx = p - cs.acc_start;
            float v = 0.0f;
            if (idx >= 0 && idx < (int64_t)cs.acc.size() && cs.wacc[(size_t)idx] > 1e-12f) {
                v = cs.acc[(size_t)idx] / cs.wacc[(size_t)idx];
            }
            out.push_back(v);
        }
        cs.emitted = emit_end;
        
        int64_t q_drop = cs.next_frame - cs.queue_start;
        if (q_drop > 4096 && q_drop < (int64_t)cs.queue.size()) {
            size_t drop = (size_t)q_drop;
            size_t remaining = cs.queue.size() - drop;
            std::memmove(cs.queue.data(), cs.queue.data() + drop, remaining * sizeof(float));
            cs.queue.resize(remaining);
            cs.queue_start += drop;
        }
        
        int64_t a_drop = cs.emitted - cs.acc_start;
        if (a_drop > 4096 && a_drop < (int64_t)cs.acc.size()) {
            size_t drop = (size_t)a_drop;
            size_t remaining = cs.acc.size() - drop;
            std::memmove(cs.acc.data(), cs.acc.data() + drop, remaining * sizeof(float));
            std::memmove(cs.wacc.data(), cs.wacc.data() + drop, remaining * sizeof(float));
            cs.acc.resize(remaining);
            cs.wacc.resize(remaining);
            cs.acc_start += drop; 
        }
    }
};