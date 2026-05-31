#ifndef ENTROPY_DECODER_H
#define ENTROPY_DECODER_H

#include <vector>
#include <cstdint>

class BitReaderMSB {
private:
    const uint8_t* data_;
    size_t size_;
    size_t byte_pos_ = 0;
    uint64_t buffer_ = 0;
    int bit_count_ = 0;
public:
    BitReaderMSB(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    inline void ensure_bits(int min_bits) {
        while (bit_count_ < min_bits && byte_pos_ < size_) {
            buffer_ = (buffer_ << 8) | data_[byte_pos_++];
            bit_count_ += 8;
        }
    }
    inline bool read_bit() {
        if (bit_count_ == 0) ensure_bits(1);
        --bit_count_;
        return (buffer_ >> bit_count_) & 1;
    }
    inline uint32_t read_bits(int num_bits) {
        if (num_bits <= 0) return 0;
        ensure_bits(num_bits);
        uint32_t shift = bit_count_ - num_bits;
        uint32_t result = (buffer_ >> shift) & ((1ULL << num_bits) - 1);
        if (shift > 0) buffer_ &= (1ULL << shift) - 1;
        else buffer_ = 0;
        bit_count_ -= num_bits;
        return result;
    }
    int available_bits() const { return ((size_ - byte_pos_) << 3) + bit_count_; }
};

inline int32_t zigzag_decode(uint32_t n) {
    return (n >> 1) ^ -(n & 1);
}

inline std::vector<uint32_t> rice_decode(BitReaderMSB& reader, int count, int k) {
    std::vector<uint32_t> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        uint32_t q = 0;
        while (reader.read_bit()) ++q;
        uint32_t r = reader.read_bits(k);
        out.push_back((q << k) | r);
    }
    return out;
}

inline void apply_inverse_lpc_order1(const std::vector<int32_t>& residual, std::vector<int32_t>& out) {
    size_t n = residual.size();
    out.resize(n);
    if (n == 0) return;
    out[0] = residual[0];
    for (size_t i = 1; i < n; ++i) out[i] = residual[i] + out[i - 1];
}
inline void apply_inverse_lpc_order2(const std::vector<int32_t>& residual, std::vector<int32_t>& out) {
    size_t n = residual.size();
    out.resize(n);
    if (n >= 1) out[0] = residual[0];
    if (n >= 2) out[1] = residual[1] + 2 * out[0];
    for (size_t i = 2; i < n; ++i) out[i] = residual[i] + 2 * out[i - 1] - out[i - 2];
}
inline void apply_inverse_lpc_order3(const std::vector<int32_t>& residual, std::vector<int32_t>& out) {
    size_t n = residual.size();
    out.resize(n);
    if (n >= 1) out[0] = residual[0];
    if (n >= 2) out[1] = residual[1] + 3 * out[0];
    if (n >= 3) out[2] = residual[2] + 3 * out[1] - 3 * out[0];
    for (size_t i = 3; i < n; ++i) out[i] = residual[i] + 3 * out[i - 1] - 3 * out[i - 2] + out[i - 3];
}
inline void apply_inverse_lpc(const std::vector<int32_t>& residual, int order, std::vector<int32_t>& out) {
    switch (order) {
        case 1: apply_inverse_lpc_order1(residual, out); break;
        case 2: apply_inverse_lpc_order2(residual, out); break;
        case 3: apply_inverse_lpc_order3(residual, out); break;
        default: out = residual; break;
    }
}

#endif // ENTROPY_DECODER_H

