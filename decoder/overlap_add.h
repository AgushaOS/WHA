#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

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
    int ch = ola.num_channels;
    size_t needed = (ola.head_offset + block_size) * ch;
    if (ola.accum.size() < needed) {
        ola.accum.resize(needed, 0.0f);
        ola.weight.resize(needed, 0.0f);
    }

    for (int c = 0; c < ch; ++c) {
        const float* src = recon_chs[c].data();
        for (int i = 0; i < block_size; ++i) {
            int local_idx = (local_start + i) * ch + c;
            ola.accum[local_idx]  += src[i] * window[i];
            ola.weight[local_idx] += window_sq[i];
        }
    }
}

inline int extract_ready_samples(OverlapAddState& ola,
                                 std::vector<float>& write_buf)
{
    int ready = ola.current_pos - ola.output_pos;
    if (ready <= 0) return 0;

    int ch = ola.num_channels;
    size_t offset = write_buf.size();
    write_buf.resize(offset + ready * ch);
    float* out_ptr = write_buf.data() + offset;

    for (int i = 0; i < ready; ++i) {
        int local_idx = ((int)ola.head_offset + i) * ch;
        for (int c = 0; c < ch; ++c) {
            float val = (ola.weight[local_idx + c] > 1e-9f)
                ? (ola.accum[local_idx + c] / ola.weight[local_idx + c])
                : 0.0f;
            out_ptr[i * ch + c] = val;
        }
    }

    ola.head_offset += ready;
    ola.output_pos   = ola.current_pos;
    return ready;
}

inline void compact_ola_buffer(OverlapAddState& ola) {
    size_t compact_threshold =
        (ola.write_batch_frames > 2048) ? ola.write_batch_frames / 2 : 2048;
    if (ola.head_offset > compact_threshold &&
        ola.head_offset * 2 > ola.accum.size() / ola.num_channels)
    {
        size_t remove_elements = ola.head_offset * ola.num_channels;
        if (remove_elements > 0 && remove_elements <= ola.accum.size()) {
            ola.accum.erase(ola.accum.begin(),
                            ola.accum.begin() + remove_elements);
            ola.weight.erase(ola.weight.begin(),
                             ola.weight.begin() + remove_elements);
            ola.head_offset = 0;
        }
    }
}

inline int extract_remaining(OverlapAddState& ola,
                             std::vector<float>& write_buf)
{
    int ch = ola.num_channels;
    size_t remaining = ola.accum.size() / ch - ola.head_offset;
    if (remaining == 0) return 0;

    size_t offset = write_buf.size();
    write_buf.resize(offset + remaining * ch);
    float* out_ptr = write_buf.data() + offset;

    for (size_t i = 0; i < remaining; ++i) {
        size_t local_idx = (ola.head_offset + i) * ch;
        for (int c = 0; c < ch; ++c) {
            float val = (ola.weight[local_idx + c] > 1e-9f)
                ? (ola.accum[local_idx + c] / ola.weight[local_idx + c])
                : 0.0f;
            out_ptr[i * ch + c] = val;
        }
    }
    return (int)remaining;
}