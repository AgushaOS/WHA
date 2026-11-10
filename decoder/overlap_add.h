#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

struct OverlapAddState {
    std::vector<float> accum;
    std::vector<float> weight;
    size_t             head_offset = 0;
    int                output_pos  = 0;
    int                current_pos = 0;
    int                num_channels = 1;
    size_t             write_batch_frames = 8192;

    void init(int ch, size_t batch) {
        num_channels       = ch;
        write_batch_frames = batch;
        accum.reserve(batch * 4 * ch);
        weight.reserve(batch * 4 * ch);
    }
};

inline void make_window(std::vector<float>& window,
                        std::vector<float>& window_sq,
                        int block_size, int left_overlap, int right_overlap)
{
    window.assign(block_size, 1.0f);
    window_sq.assign(block_size, 1.0f);
    if (left_overlap > 1) {
        for (int i = 0; i < left_overlap; ++i) {
            float t = (float)i / left_overlap;
            float w = t * t * (3.0f - 2.0f * t);
            window[i]    = w;
            window_sq[i] = w * w;
        }
    }
    if (right_overlap > 1) {
        for (int i = 0; i < right_overlap; ++i) {
            float t = (float)i / right_overlap;
            float w = t * t * (3.0f - 2.0f * t);
            window[block_size - 1 - i]    = w;
            window_sq[block_size - 1 - i] = w * w;
        }
    }
}

inline void overlap_add_block(OverlapAddState& ola,
                              const std::vector<std::vector<float>>& recon_chs,
                              int block_size,
                              const std::vector<float>& window,
                              const std::vector<float>& window_sq,
                              int local_start)
{
    const int ch = ola.num_channels;
    const size_t needed = (size_t)(ola.head_offset + block_size) * ch;
    if (ola.accum.size() < needed) {
        ola.accum.resize(needed, 0.0f);
        ola.weight.resize(needed, 0.0f);
    }

    float* accum_ptr  = ola.accum.data();
    float* weight_ptr = ola.weight.data();
    const float* win_ptr    = window.data();
    const float* win_sq_ptr = window_sq.data();

    const float* src_ptrs[8];
    if (ch <= 8) {
        for (int c = 0; c < ch; ++c) src_ptrs[c] = recon_chs[c].data();
        for (int i = 0; i < block_size; ++i) {
            const float wv  = win_ptr[i];
            const float wsq = win_sq_ptr[i];
            const size_t base = (size_t)(local_start + i) * ch;
            for (int c = 0; c < ch; ++c) {
                accum_ptr [base + c] += src_ptrs[c][i] * wv;
                weight_ptr[base + c] += wsq;
            }
        }
    } else {
        for (int i = 0; i < block_size; ++i) {
            const float wv  = win_ptr[i];
            const float wsq = win_sq_ptr[i];
            const size_t base = (size_t)(local_start + i) * ch;
            for (int c = 0; c < ch; ++c) {
                accum_ptr [base + c] += recon_chs[c][i] * wv;
                weight_ptr[base + c] += wsq;
            }
        }
    }
}

inline int extract_ready_samples(OverlapAddState& ola,
                                 std::vector<float>& write_buf)
{
    const int ready = ola.current_pos - ola.output_pos;
    if (ready <= 0) return 0;
    const int ch = ola.num_channels;

    const size_t offset = write_buf.size();
    write_buf.resize(offset + (size_t)ready * ch);
    float* out_ptr = write_buf.data() + offset;

    const float* accum_ptr  = ola.accum.data();
    const float* weight_ptr = ola.weight.data();
    const size_t start_elem = ola.head_offset * ch;

    for (int i = 0; i < ready; ++i) {
        const size_t local_idx = start_elem + (size_t)i * ch;
        for (int c = 0; c < ch; ++c) {
            const float w = weight_ptr[local_idx + c];
            const float v = (w > 1e-9f) ? (accum_ptr[local_idx + c] / w) : 0.0f;
            out_ptr[(size_t)i * ch + c] = v;
        }
    }
    ola.head_offset += ready;
    ola.output_pos   = ola.current_pos;
    return ready;
}

inline void compact_ola_buffer(OverlapAddState& ola) {
    if (ola.head_offset == 0) return;

    const size_t ch = (size_t)ola.num_channels;
    const size_t remove_elements = ola.head_offset * ch;
    const size_t total_elements  = ola.accum.size();
    if (remove_elements >= total_elements) {
        ola.accum.clear();
        ola.weight.clear();
        ola.head_offset = 0;
        return;
    }
    const size_t remaining = total_elements - remove_elements;

    std::memmove(ola.accum.data(),
                 ola.accum.data() + remove_elements,
                 remaining * sizeof(float));
    std::memmove(ola.weight.data(),
                 ola.weight.data() + remove_elements,
                 remaining * sizeof(float));
    ola.accum.resize(remaining);
    ola.weight.resize(remaining);
    ola.head_offset = 0;
}

inline int extract_remaining(OverlapAddState& ola,
                             std::vector<float>& write_buf)
{
    const int ch = ola.num_channels;
    const size_t remaining = ola.accum.size() / ch - ola.head_offset;
    if (remaining == 0) return 0;

    const size_t offset = write_buf.size();
    write_buf.resize(offset + remaining * ch);
    float* out_ptr = write_buf.data() + offset;

    const float* accum_ptr  = ola.accum.data();
    const float* weight_ptr = ola.weight.data();
    const size_t start_elem = ola.head_offset * ch;

    for (size_t i = 0; i < remaining; ++i) {
        const size_t local_idx = start_elem + i * ch;
        for (int c = 0; c < ch; ++c) {
            const float w = weight_ptr[local_idx + c];
            const float v = (w > 1e-9f) ? (accum_ptr[local_idx + c] / w) : 0.0f;
            out_ptr[i * ch + c] = v;
        }
    }
    return (int)remaining;
}