#ifndef ENTROPY_ENCODER_H
#define ENTROPY_ENCODER_H

#include <vector>
#include <cstdint>
#include <climits>
#include <cstring>

inline uint32_t zigzag_encode(int32_t n) {
    return (n << 1) ^ (n >> 31);
}

inline std::vector<int32_t> apply_lpc(const std::vector<int32_t>& data, int order) {
    if (order == 0) return data;
    size_t n = data.size();
    std::vector<int32_t> residual(n);
    if (order == 1) {
        residual[0] = data[0];
        for (size_t i = 1; i < n; ++i)
            residual[i] = data[i] - data[i-1];
    } else if (order == 2) {
        if (n >= 1) residual[0] = data[0];
        if (n >= 2) residual[1] = data[1] - 2*data[0];
        for (size_t i = 2; i < n; ++i)
            residual[i] = data[i] - (2*data[i-1] - data[i-2]);
    } else if (order == 3) {
        if (n >= 1) residual[0] = data[0];
        if (n >= 2) residual[1] = data[1] - 3*data[0];
        if (n >= 3) residual[2] = data[2] - (3*data[1] - 3*data[0]);
        for (size_t i = 3; i < n; ++i)
            residual[i] = data[i] - (3*data[i-1] - 3*data[i-2] + data[i-3]);
    }
    return residual;
}

inline std::pair<int, int> rice_optimal_k_and_cost(const std::vector<uint32_t>& data) {
    if (data.empty()) return {0, 0};
    int best_k = 0;
    int best_bits = INT_MAX;
    for (int k = 0; k <= 15; ++k) {
        int bits = 0;
        for (uint32_t x : data) {
            uint32_t q = x >> k;
            bits += q + 1 + k;
        }
        if (bits < best_bits) {
            best_bits = bits;
            best_k = k;
        }
    }
    return {best_k, best_bits};
}

inline int estimate_order_bits(const std::vector<int32_t>& data, int order) {
    if (data.empty()) return 0;
    std::vector<int32_t> residual = apply_lpc(data, order);
    std::vector<uint32_t> uvals(residual.size());
    for (size_t i = 0; i < residual.size(); ++i)
        uvals[i] = zigzag_encode(residual[i]);
    auto [k, bits] = rice_optimal_k_and_cost(uvals);
    return bits + 2 + 4; 
}

inline int select_best_order(const std::vector<int32_t>& data, int band_idx) {
    int max_order = (band_idx < 4) ? 3 : 1;
    int best_order = 0;
    int best_bits = INT_MAX;
    size_t n = data.size();
    for (int order = 0; order <= max_order; ++order) {
        if (order > (int)n) continue;
        int bits = estimate_order_bits(data, order);
        if (bits < best_bits) {
            best_bits = bits;
            best_order = order;
        }
    }
    return best_order;
}

class BitWriterMSB {
private:
    std::vector<uint8_t> data_;
    uint64_t buffer_ = 0;
    int bit_count_ = 0;
public:
    void write_bit(bool bit) {
        buffer_ = (buffer_ << 1) | (bit ? 1 : 0);
        bit_count_++;
        if (bit_count_ == 64) flush64();
    }
    void write_bits(uint32_t value, int num_bits) {
        if (num_bits <= 0) return;
        if (bit_count_ + num_bits <= 64) {
            buffer_ = (buffer_ << num_bits) | (value & ((1ULL << num_bits) - 1));
            bit_count_ += num_bits;
            if (bit_count_ == 64) flush64();
        } else {
            for (int i = num_bits - 1; i >= 0; --i)
                write_bit((value >> i) & 1);
        }
    }
    void flush64() {
        if (bit_count_ >= 64) {
            for (int i = 7; i >= 0; --i)
                data_.push_back((buffer_ >> (i * 8)) & 0xFF);
            buffer_ = 0;
            bit_count_ = 0;
        }
    }
    void flush() {
        while (bit_count_ >= 8) {
            int shift = bit_count_ - 8;
            data_.push_back((buffer_ >> shift) & 0xFF);
            buffer_ &= (1ULL << shift) - 1;
            bit_count_ -= 8;
        }
        if (bit_count_ > 0) {
            data_.push_back((buffer_ << (8 - bit_count_)) & 0xFF);
        }
        buffer_ = 0;
        bit_count_ = 0;
    }
    const std::vector<uint8_t>& data() const { return data_; }
    size_t size() const { return data_.size() * 8 + bit_count_; }
};

inline void rice_encode(BitWriterMSB& writer, const std::vector<uint32_t>& data, int k) {
    for (uint32_t x : data) {
        uint32_t q = x >> k;
        uint32_t r = x & ((1u << k) - 1);
        for (uint32_t i = 0; i < q; ++i) writer.write_bit(1);
        writer.write_bit(0);
        writer.write_bits(r, k);
    }
}

inline int select_best_order(const std::vector<int32_t>& data) {
    return select_best_order(data, 0);
}

inline int rice_optimal_k(const std::vector<uint32_t>& data) {
    return rice_optimal_k_and_cost(data).first;
}

#endif // ENTROPY_ENCODER_H