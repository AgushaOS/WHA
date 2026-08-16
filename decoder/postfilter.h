#ifndef POSTFILTER_H
#define POSTFILTER_H

#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fastmath {

static inline float log10f_approx(float x) {
    if (x <= 0.0f) return -10.0f;
    union { float f; uint32_t i; } u;
    u.f = x;
    int32_t e = (int32_t)((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;
    float m = u.f - 1.0f;
    float log2_val = (float)e + m * (1.442695f + m * (-0.721347f + m * 0.478652f));
    return log2_val * 0.301030f;  
}

static inline float pow10f_approx(float x) {
    float log2_val = x * 3.321928f;
    int32_t e = (int32_t)log2_val;
    float frac = log2_val - (float)e;
    if (frac < 0.0f) { frac += 1.0f; e--; }
    float pow2_frac = 1.0f + frac * (0.693147f + frac * (0.240226f + frac * 0.055504f));
    union { float f; int32_t i; } u;
    u.i = (e + 127) << 23;
    return u.f * pow2_frac;
}

}

class SpectralFloorPostFX {
public:
    struct Config {
        int   nfft = 2048;
        int   hop  = 1024;
        int   floor_size = 15;
        float floor_percentile = 10.f;
        float t_lo = 6.f, t_hi = 16.f;
        float strength = 0.5f;
        int   smooth_freq = 5;
        float temporal_strength = 0.35f;
        float max_temporal_drop = 4.f;
        bool  use_measure = 0;
    };

    explicit SpectralFloorPostFX(const Config& cfg) : c(cfg) {
        nfreq = c.nfft / 2 + 1;
        unsigned flag = c.use_measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        time_in  = (float*)fftwf_malloc(sizeof(float) * c.nfft);
        time_out = (float*)fftwf_malloc(sizeof(float) * c.nfft);
        freq     = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfreq);
        freq_mod = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfreq);

        plan_fwd = fftwf_plan_dft_r2c_1d(c.nfft, time_in,  freq,     flag);
        plan_inv = fftwf_plan_dft_c2r_1d(c.nfft, freq_mod, time_out, flag);
        if (!plan_fwd || !plan_inv)
            throw std::runtime_error("FFTW: failed to create plan");

        window.resize(c.nfft); window2.resize(c.nfft);
        for (int i = 0; i < c.nfft; i++) {
            window[i]  = 0.5f * (1.f - cosf(2.f * (float)M_PI * i / (c.nfft - 1)));
            window2[i] = window[i] * window[i];
        }

        acc.assign(c.nfft, 0.f); norm.assign(c.nfft, 0.f);
        prev_gain.assign(nfreq, 1.f);
        noise_memory.assign(nfreq, 0.f);
        floor_cur.assign(nfreq, 0.f); prev_floor.assign(nfreq, 0.f);
        has_prev_floor = false;

        s_power.resize(nfreq); s_contrast.resize(nfreq); s_gaindb.resize(nfreq);
        s_valley.resize(nfreq); s_smooth.resize(nfreq); s_gain.resize(nfreq);

        int half = c.floor_size / 2;
        db_padded.resize(nfreq + 2 * half);
        if (!(c.floor_size == 15 && fabsf(c.floor_percentile - 10.f) < 1e-6f)) {
            wv.resize(c.floor_size);
        }
    }

    ~SpectralFloorPostFX() {
        if (plan_fwd) fftwf_destroy_plan(plan_fwd);
        if (plan_inv) fftwf_destroy_plan(plan_inv);
        if (time_in)  fftwf_free(time_in);
        if (time_out) fftwf_free(time_out);
        if (freq)     fftwf_free(freq);
        if (freq_mod) fftwf_free(freq_mod);
    }

    SpectralFloorPostFX(const SpectralFloorPostFX&) = delete;
    SpectralFloorPostFX& operator=(const SpectralFloorPostFX&) = delete;

    int process(const float* in, int n_in, float* out) {
        in_fifo.insert(in_fifo.end(), in, in + n_in);
        int produced = 0;
        while ((int)in_fifo.size() >= c.nfft) {
            processFrame(in_fifo.data(), out + produced);
            produced += c.hop;

            std::memmove(in_fifo.data(), in_fifo.data() + c.hop,
                         (in_fifo.size() - c.hop) * sizeof(float));
            in_fifo.resize(in_fifo.size() - c.hop);
        }
        return produced;
    }

    int flush(float* out) {
        std::vector<float> zeros(2 * c.nfft, 0.f);
        int produced = process(zeros.data(), (int)zeros.size(), out);
        reset();
        return produced;
    }

    void reset() {
        std::fill(acc.begin(), acc.end(), 0.f);
        std::fill(norm.begin(), norm.end(), 0.f);
        std::fill(prev_gain.begin(), prev_gain.end(), 1.f);
        std::fill(noise_memory.begin(), noise_memory.end(), 0.f);
        has_prev_floor = false;
        in_fifo.clear();
    }

private:
    Config c;
    int nfreq;
    fftwf_plan plan_fwd = nullptr, plan_inv = nullptr;
    float* time_in = nullptr;  float* time_out = nullptr;
    fftwf_complex* freq = nullptr; fftwf_complex* freq_mod = nullptr;

    std::vector<float> window, window2, acc, norm;
    std::vector<float> prev_gain, noise_memory, floor_cur, prev_floor;
    bool has_prev_floor;
    std::vector<float> in_fifo;
    std::vector<float> s_power, s_contrast, s_gaindb, s_valley, s_smooth, s_gain;
    std::vector<float> db_padded;
    std::vector<float> wv;

    void processFrame(const float* frame_in, float* frame_out) {
        const int N = c.nfft, H = c.hop, NF = nfreq;

        for (int i = 0; i < N; i++) time_in[i] = frame_in[i] * window[i];
        fftwf_execute_dft_r2c(plan_fwd, time_in, freq);

        float* power = s_power.data();
        float* fl = floor_cur.data();
        for (int i = 0; i < NF; i++) {
            float re = freq[i][0], im = freq[i][1];
            power[i] = re*re + im*im;  
        }

        bool fast_path = (c.floor_size == 15 && fabsf(c.floor_percentile - 10.f) < 1e-6f);
        if (fast_path) {
            const int half = c.floor_size / 2;
            float* pad = db_padded.data();

            for (int i = 0; i < half; i++) {
                pad[i] = power[0];
                pad[half + NF + i] = power[NF - 1];
            }
            memcpy(pad + half, power, NF * sizeof(float));

            float m0, m1, m2;
            int p0, p1, p2;
            m0 = m1 = m2 = 1e30f;
            p0 = p1 = p2 = -1;
            for (int j = 0; j < 15; j++) {
                float val = pad[j];
                if (val < m0) {
                    m2 = m1; p2 = p1;
                    m1 = m0; p1 = p0;
                    m0 = val; p0 = j;
                } else if (val < m1) {
                    m2 = m1; p2 = p1;
                    m1 = val; p1 = j;
                } else if (val < m2) {
                    m2 = val; p2 = j;
                }
            }
            fl[0] = 0.6f * m1 + 0.4f * m2;

            for (int i = 1; i < NF; i++) {
                bool need_recalc = false;
                if (p0 == 0) { p0 = -1; need_recalc = true; } else if (p0 > 0) p0--;
                if (p1 == 0) { p1 = -1; need_recalc = true; } else if (p1 > 0) p1--;
                if (p2 == 0) { p2 = -1; need_recalc = true; } else if (p2 > 0) p2--;

                if (need_recalc) {
                    m0 = m1 = m2 = 1e30f;
                    p0 = p1 = p2 = -1;
                    for (int j = 0; j < 15; j++) {
                        float val = pad[i + j];
                        if (val < m0) {
                            m2 = m1; p2 = p1;
                            m1 = m0; p1 = p0;
                            m0 = val; p0 = j;
                        } else if (val < m1) {
                            m2 = m1; p2 = p1;
                            m1 = val; p1 = j;
                        } else if (val < m2) {
                            m2 = val; p2 = j;
                        }
                    }
                } else {
                    float new_val = pad[i + 14];
                    if (new_val < m2) {
                        if (new_val < m0) {
                            m2 = m1; p2 = p1;
                            m1 = m0; p1 = p0;
                            m0 = new_val; p0 = 14;
                        } else if (new_val < m1) {
                            m2 = m1; p2 = p1;
                            m1 = new_val; p1 = 14;
                        } else {
                            m2 = new_val; p2 = 14;
                        }
                    }
                }
                fl[i] = 0.6f * m1 + 0.4f * m2;
            }
        } else {
            int half = c.floor_size / 2;
            float* pad = db_padded.data();
            for (int i = 0; i < half; i++) {
                pad[i] = power[0];
                pad[half + NF + i] = power[NF - 1];
            }
            memcpy(pad + half, power, NF * sizeof(float));

            for (int i = 0; i < NF; i++) {
                int cnt = 0;
                for (int j = i - half; j <= i + half; j++) {
                    int idx = j; if (idx < 0) idx = 0; if (idx >= NF) idx = NF - 1;
                    wv[cnt++] = power[idx];
                }
                std::sort(wv.begin(), wv.begin() + cnt);
                float rank = (cnt - 1) * c.floor_percentile * 0.01f;
                int i1 = (int)rank; int i2 = std::min(i1 + 1, cnt - 1);
                float frac = rank - i1;
                fl[i] = wv[i1] * (1 - frac) + wv[i2] * frac;
            }
        }

        float* contrast = s_contrast.data();
        for (int i = 0; i < NF; i++) {
            float power_ratio = power[i] / fmaxf(fl[i], 1e-20f);
            contrast[i] = 10.f * fastmath::log10f_approx(power_ratio);
        }

        float* valley = s_valley.data();
        float inv_range = 1.f / fmaxf(1e-6f, c.t_hi - c.t_lo);
        for (int i = 0; i < NF; i++) {
            float t = (contrast[i] - c.t_lo) * inv_range;
            t = fminf(fmaxf(t, 0.f), 1.f);
            float peak = t * t * (3.f - 2.f * t);
            valley[i] = 1.f - peak;
        }

        float* gdb = s_gaindb.data();
        for (int i = 0; i < NF; i++) {
            float drop = fminf(fmaxf(contrast[i] * c.strength, 0.f), 12.f);
            gdb[i] = -drop * valley[i];
        }

        if (has_prev_floor) {
            for (int i = 0; i < NF; i++) {
                float fl_db = 10.f * fastmath::log10f_approx(fmaxf(fl[i], 1e-20f));
                float prev_fl_db = 10.f * fastmath::log10f_approx(fmaxf(prev_floor[i], 1e-20f));
                float fd = fabsf(fl_db - prev_fl_db);
                float stable = expf(-fd * 0.5f) * valley[i];
                noise_memory[i] = 0.85f * noise_memory[i] + 0.15f * stable;
            }
        }
        for (int i = 0; i < NF; i++) {
            float cc = fminf(fmaxf(contrast[i], 0.f), 12.f);
            float td = c.temporal_strength * noise_memory[i] * cc;
            td = fminf(td, c.max_temporal_drop) * valley[i];
            gdb[i] -= td;
        }

        if (c.smooth_freq > 1) {
            int sh = c.smooth_freq / 2;
            float* sm = s_smooth.data();
            for (int i = 0; i < NF; i++) {
                float s = 0; int cnt = 0;
                for (int j = i - sh; j <= i + sh; j++) {
                    int idx = j; if (idx < 0) idx = 0; if (idx >= NF) idx = NF - 1;
                    s += gdb[idx]; cnt++;
                }
                sm[i] = s / cnt;
            }
            for (int i = 0; i < NF; i++) gdb[i] = sm[i];
        }

        float* g = s_gain.data();
        for (int i = 0; i < NF; i++) {
            g[i] = fastmath::pow10f_approx(gdb[i] * 0.05f);  
            g[i] = 0.5f * g[i] + 0.5f * prev_gain[i];
            prev_gain[i] = g[i];
            freq_mod[i][0] = freq[i][0] * g[i];
            freq_mod[i][1] = freq[i][1] * g[i];
        }

        fftwf_execute_dft_c2r(plan_inv, freq_mod, time_out);

        const float inv_n = 1.f / N;
        for (int i = 0; i < N; i++) {
            float syn = time_out[i] * inv_n * window[i];
            acc[i]  += syn;
            norm[i] += window2[i];
        }
        for (int i = 0; i < H; i++)
            frame_out[i] = acc[i] / fmaxf(norm[i], 1e-8f);

        std::memmove(acc.data(),  acc.data() + H,  H * sizeof(float));
        std::memset (acc.data() + H, 0, H * sizeof(float));
        std::memmove(norm.data(), norm.data() + H, H * sizeof(float));
        std::memset (norm.data() + H, 0, H * sizeof(float));

        std::swap(floor_cur, prev_floor);
        has_prev_floor = true;
    }
};

class SpectralFloorPostProcessor {
public:
    SpectralFloorPostProcessor(int channels, const SpectralFloorPostFX::Config& cfg)
        : num_channels(channels), c(cfg)
    {
        filters.reserve(num_channels);
        for (int ch = 0; ch < num_channels; ch++)
            filters.push_back(std::make_unique<SpectralFloorPostFX>(cfg));
        per_ch_out.resize(num_channels);
        ch_in.resize(4096);
    }

    int process(const float* interleaved, int n_frames) {
        total_in += n_frames;
        if ((int)ch_in.size() < n_frames) ch_in.resize(n_frames);
        int cap = n_frames + 4 * c.nfft + c.hop;

        int n_out = 0;
        for (int ch = 0; ch < num_channels; ch++) {
            for (int i = 0; i < n_frames; i++)
                ch_in[i] = interleaved[i * num_channels + ch];
            if ((int)per_ch_out[ch].size() < cap) per_ch_out[ch].resize(cap);
            int n = filters[ch]->process(ch_in.data(), n_frames, per_ch_out[ch].data());
            if (ch == 0) n_out = n;
        }

        n_out = capOutput(n_out);
        interleave(n_out);
        total_out += n_out;
        return n_out;
    }

    int flush() {
        int cap = 4 * c.nfft + c.hop;
        int n_out = 0;
        for (int ch = 0; ch < num_channels; ch++) {
            if ((int)per_ch_out[ch].size() < cap) per_ch_out[ch].resize(cap);
            int n = filters[ch]->flush(per_ch_out[ch].data());
            if (ch == 0) n_out = n;
        }
        n_out = capOutput(n_out);
        interleave(n_out);
        total_out += n_out;
        return n_out;
    }

    const float* out_data() const { return out_buf.data(); }

private:
    int num_channels;
    SpectralFloorPostFX::Config c;
    std::vector<std::unique_ptr<SpectralFloorPostFX>> filters;
    std::vector<std::vector<float>> per_ch_out;
    std::vector<float> ch_in;
    std::vector<float> out_buf;
    int64_t total_in  = 0;
    int64_t total_out = 0;

    int capOutput(int n_out) const {
        int64_t max_allowed = total_in - total_out;
        if (n_out > max_allowed) n_out = (int)max_allowed;
        if (n_out < 0) n_out = 0;
        return n_out;
    }

    void interleave(int n_out) {
        out_buf.resize((size_t)n_out * num_channels);
        for (int i = 0; i < n_out; i++)
            for (int ch = 0; ch < num_channels; ch++)
                out_buf[i * num_channels + ch] = per_ch_out[ch][i];
    }
};

static inline bool shouldApplySpectralFloor(int num_channels, float target_kbps) {
    if (num_channels == 2 && target_kbps < 128.0f) return true;
    if (num_channels == 1 && target_kbps <  80.0f) return true;
    return false;
}

#endif // POSTFILTER_H