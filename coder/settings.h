#pragma once

struct EncoderSettings {
    bool  enable_ms                       = true;
    int   reservoir_max_factor            = 1024;
    float default_target_kbps             = 128.0f;
    bool  verbose                         = false;
    float transient_ratio_threshold       = 0.15f;
    bool  enable_adaptive_is              = true;
    float adaptive_is_kbps_min            = 128.0f;
    float adaptive_is_kbps_max            = 159.0f;
    int   adaptive_is_base_at_64          = 2;
    int   adaptive_is_base_at_96          = 4;
    int   adaptive_is_base_at_128         = 6;
    int   adaptive_is_base_at_160         = 12;
    int   adaptive_is_base_override       = -1;
    float is_strong_ratio_threshold       = 4.0f;
    float is_equal_ratio_threshold        = 2.0f;
    float is_equal_ratio_threshold_low_bitrate = 3.0f;
};

inline EncoderSettings SETTINGS;