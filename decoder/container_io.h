#pragma once
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

inline uint8_t read_u8(std::ifstream& f) {
    uint8_t v = 0;
    f.read((char*)&v, 1);
    if (!f) throw std::runtime_error("Unexpected EOF (u8)");
    return v;
}

inline uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read((char*)&v, 4);
    if (!f) throw std::runtime_error("Unexpected EOF (u32)");
    return v;
}

inline float read_f32(std::ifstream& f) {
    float v = 0;
    f.read((char*)&v, 4);
    if (!f) throw std::runtime_error("Unexpected EOF (f32)");
    return v;
}

inline std::vector<uint8_t> read_n(std::ifstream& f, size_t n) {
    std::vector<uint8_t> buf(n);
    f.read((char*)buf.data(), n);
    size_t got = (size_t)f.gcount();
    if (got < n) buf.resize(got);
    return buf;
}