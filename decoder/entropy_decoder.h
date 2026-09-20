#ifndef ENTROPY_DECODER_H
#define ENTROPY_DECODER_H

#include <vector>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

class BitReaderMSB {
private:
    const uint8_t* data_;
    size_t size_;
    size_t byte_pos_ = 0;
    uint64_t buffer_ = 0;
    int bit_count_ = 0;

    inline int count_leading_zeros(uint64_t x) const {
        if (x == 0) return 64;
#ifdef _MSC_VER
        unsigned long index;
        _BitScanReverse64(&index, x);
        return 63 - index;
#else
        return __builtin_clzll(x);
#endif
    }

public:
    BitReaderMSB(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    inline void ensure_bits(int min_bits) {
        while (bit_count_ < min_bits && byte_pos_ < size_) {
            if (bit_count_ + 8 > 64) break; 
            buffer_ = (buffer_ << 8) | data_[byte_pos_++];
            bit_count_ += 8;
        }
    }

    inline bool read_bit() {
        if (bit_count_ == 0) ensure_bits(1);
        --bit_count_;
        return (buffer_ >> bit_count_) & 1;
    }

    inline uint64_t read_bits(int num_bits) {
        if (num_bits <= 0) return 0;
        ensure_bits(num_bits);
        uint32_t shift = bit_count_ - num_bits;
        uint64_t mask = (num_bits == 64) ? ~0ULL : ((1ULL << num_bits) - 1);
        uint64_t result = (buffer_ >> shift) & mask;
        
        if (shift > 0) {
            buffer_ &= (1ULL << shift) - 1;
        } else {
            buffer_ = 0;
        }
        bit_count_ -= num_bits;
        return result;
    }

    inline uint32_t read_unary() {
        uint32_t q = 0;
        while (true) {
            ensure_bits(64);
            int avail = bit_count_;
            if (avail == 0) break;
            
            uint64_t inv = ~buffer_;
            if (avail < 64) {
                inv &= (1ULL << avail) - 1; 
            }
            
            int zeros = count_leading_zeros(inv);
            int leading_ones = zeros - (64 - avail);
            
            if (leading_ones == avail) {
                q += avail;
                bit_count_ = 0;
                buffer_ = 0;
            } else {
                q += leading_ones;
                int consume = leading_ones + 1;
                bit_count_ -= consume;
                if (bit_count_ > 0) {
                    buffer_ &= (1ULL << bit_count_) - 1;
                } else {
                    buffer_ = 0;
                }
                break;
            }
        }
        return q;
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
        uint32_t q = reader.read_unary();
        uint32_t r = k > 0 ? reader.read_bits(k) : 0;
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
