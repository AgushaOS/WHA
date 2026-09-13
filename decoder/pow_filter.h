#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
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

    PowCodec(int num_channels, float gamma, bool encode)
        : num_channels_(num_channels), gamma_(gamma), encode_(encode),
          channels_(num_channels)
    {
        if (num_channels_ <= 0) throw std::runtime_error("PowCodec: bad channels");

        in_     = fftwf_alloc_real(NFFT);
        out_c_  = fftwf_alloc_complex(NBINS);
        in_c2r_ = fftwf_alloc_complex(NBINS);
        out_    = fftwf_alloc_real(NFFT);

        r2c_ = fftwf_plan_dft_r2c_1d(NFFT, in_,     out_c_, FFTW_ESTIMATE);
        c2r_ = fftwf_plan_dft_c2r_1d(NFFT, in_c2r_, out_,   FFTW_ESTIMATE);

        win_.resize(NFFT);
        win_sq_.resize(NFFT);
        for (int i = 0; i < NFFT; ++i) {
            float s = std::sin((float)M_PI * i / NFFT);
            win_[i]    = std::sqrt(std::sin((float)(M_PI / 2.0) * s * s));
            win_sq_[i] = win_[i] * win_[i];
        }
    }

    ~PowCodec() {
        fftwf_destroy_plan(r2c_);
        fftwf_destroy_plan(c2r_);
        fftwf_free(in_);
        fftwf_free(out_c_);
        fftwf_free(in_c2r_);
        fftwf_free(out_);
    }

    PowCodec(const PowCodec&)            = delete;
    PowCodec& operator=(const PowCodec&) = delete;

    void process(const float* in, size_t frames, std::vector<float>& out) {
        if (frames == 0) return;

        for (size_t f = 0; f < frames; ++f) {
            for (int c = 0; c < num_channels_; ++c) {
                channels_[c].queue.push_back(in[f * num_channels_ + c]);
            }
            for (int c = 0; c < num_channels_; ++c) {
                channels_[c].total_in += 1;
            }
        }

        std::vector<std::vector<float>> ch_out(num_channels_);
        for (int c = 0; c < num_channels_; ++c)
            processChannel(channels_[c], ch_out[c], false);

        size_t n_out = ch_out.empty() ? 0 : ch_out[0].size();
        for (size_t f = 0; f < n_out; ++f)
            for (int c = 0; c < num_channels_; ++c)
                out.push_back(ch_out[c][f]);
    }

    void flush(std::vector<float>& out) {
        std::vector<std::vector<float>> ch_out(num_channels_);
        for (int c = 0; c < num_channels_; ++c)
            processChannel(channels_[c], ch_out[c], true);

        size_t n_out = ch_out.empty() ? 0 : ch_out[0].size();
        for (size_t f = 0; f < n_out; ++f)
            for (int c = 0; c < num_channels_; ++c)
                out.push_back(ch_out[c][f]);
    }

    void reset() {
        for (auto& cs : channels_) cs = ChannelState{};
    }

private:
    struct ChannelState {
        std::vector<float> queue;     
        std::vector<float> acc;         
        std::vector<float> wacc;        
        int64_t queue_start   = 0;      
        int64_t acc_start     = 0;      
        int64_t next_frame    = 0;      
        int64_t emitted       = 0;      
        int64_t total_in      = 0;      
        bool    seeded        = false;  
        bool    tail_padded   = false;  
    };

    int    num_channels_;
    float  gamma_;
    bool   encode_;
    std::vector<ChannelState> channels_;

    fftwf_plan      r2c_{};
    fftwf_plan      c2r_{};
    float*          in_     = nullptr;
    fftwf_complex*  out_c_  = nullptr;
    fftwf_complex*  in_c2r_ = nullptr;
    float*          out_    = nullptr;
    std::vector<float> win_, win_sq_;

    void processFrameInPlace(float* frame) {
        for (int i = 0; i < NFFT; ++i) in_[i] = frame[i] * win_[i];
        fftwf_execute(r2c_);

        float mag[NBINS];
        float phase[NBINS];
        float max_mag = 0.0f;

        for (int i = 0; i < NBINS; ++i) {
            float re = out_c_[i][0];
            float im = out_c_[i][1];
            float m  = std::sqrt(re * re + im * im);
            mag[i]   = m;
            phase[i] = std::atan2(im, re);
            if (m > max_mag) max_mag = m;
        }

        if (gamma_ != 1.0f && max_mag > 1e-20f) {
            float inv_max = 1.0f / max_mag;
            if (encode_) {
                for (int i = 0; i < NBINS; ++i) {
                    float n = mag[i] * inv_max;
                    mag[i]  = std::pow(n, gamma_) * max_mag;
                }
            } else {
                float inv_g = 1.0f / gamma_;
                for (int i = 0; i < NBINS; ++i) {
                    float n = mag[i] * inv_max;
                    mag[i]  = std::pow(n, inv_g) * max_mag;
                }
            }
        }

        for (int i = 0; i < NBINS; ++i) {
            float m = mag[i];
            float p = phase[i];
            in_c2r_[i][0] = m * std::cos(p);
            in_c2r_[i][1] = m * std::sin(p);
        }
        fftwf_execute(c2r_);

        const float norm = 1.0f / (float)NFFT;
        for (int i = 0; i < NFFT; ++i)
            frame[i] = out_[i] * norm * win_[i];
    }

    void ensureAcc(ChannelState& cs, int64_t end_pos) {
        int64_t cur_end = cs.acc_start + (int64_t)cs.acc.size();
        if (end_pos <= cur_end) return;
        size_t new_size = (size_t)(end_pos - cs.acc_start);
        cs.acc.resize(new_size, 0.0f);
        cs.wacc.resize(new_size, 0.0f);
    }

    void processChannel(ChannelState& cs, std::vector<float>& out, bool final) {
        if (!cs.seeded) {
            if (cs.queue.size() < (size_t)PAD + 1 && !final) return;

            int have = (int)cs.queue.size();
            std::vector<float> prefix((size_t)PAD, 0.0f);
            if (have >= PAD + 1) {
                for (int i = 0; i < PAD; ++i) {
                    int idx = PAD - i;                 
                    prefix[i] = cs.queue[(size_t)idx];
                }
            }
            std::vector<float> nq;
            nq.reserve(prefix.size() + cs.queue.size());
            nq.insert(nq.end(), prefix.begin(), prefix.end());
            nq.insert(nq.end(), cs.queue.begin(), cs.queue.end());
            cs.queue        = std::move(nq);
            cs.queue_start  = 0;
            cs.next_frame   = 0;
            cs.seeded       = true;
        }

        if (final && !cs.tail_padded) {
            cs.tail_padded = true;
            int64_t real_end = (int64_t)PAD + cs.total_in;   

            if ((int64_t)cs.queue.size() < real_end)
                cs.queue.resize((size_t)real_end, 0.0f);

            if (cs.total_in >= (int64_t)PAD + 2) {
                for (int i = 0; i < PAD; ++i) {
                    int64_t idx = real_end - 2 - i;
                    float v = 0.0f;
                    if (idx >= 0 && idx < (int64_t)cs.queue.size())
                        v = cs.queue[(size_t)idx];
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
            for (int i = 0; i < NFFT; ++i)
                frame[i] = cs.queue[(size_t)(q_idx + i)];

            processFrameInPlace(frame);

            ensureAcc(cs, cs.next_frame + NFFT);
            int64_t a_idx = cs.next_frame - cs.acc_start;
            for (int i = 0; i < NFFT; ++i) {
                cs.acc [(size_t)(a_idx + i)] += frame[i];
                cs.wacc[(size_t)(a_idx + i)] += win_sq_[i];
            }
            cs.next_frame += HOP;
        }

        int64_t finalized = cs.next_frame;
        int64_t emit_end  = std::min(finalized, input_end);
        if (emit_end < cs.emitted) emit_end = cs.emitted;

        for (int64_t p = cs.emitted; p < emit_end; ++p) {
            if (p < PAD) continue;                       
            int64_t idx = p - cs.acc_start;
            float v = 0.0f;
            if (idx >= 0 && idx < (int64_t)cs.acc.size() && cs.wacc[(size_t)idx] > 1e-12f)
                v = cs.acc[(size_t)idx] / cs.wacc[(size_t)idx];
            out.push_back(v);
        }
        cs.emitted = emit_end;

        int64_t q_drop = cs.next_frame - cs.queue_start;
        if (q_drop > 0 && q_drop <= (int64_t)cs.queue.size()) {
            cs.queue.erase(cs.queue.begin(), cs.queue.begin() + (size_t)q_drop);
            cs.queue_start += q_drop;
        }

        int64_t a_drop = cs.emitted - cs.acc_start;
        if (a_drop > 0 && a_drop <= (int64_t)cs.acc.size()) {
            cs.acc .erase(cs.acc .begin(), cs.acc .begin() + (size_t)a_drop);
            cs.wacc.erase(cs.wacc.begin(), cs.wacc.begin() + (size_t)a_drop);
            cs.acc_start += a_drop;
        }
    }
};