
// Potential new priority function
// inline std::vector<float> compute_channel_priority(
//     const std::vector<std::vector<float>>& coeffs,
//     std::vector<std::vector<float>>& prev_coeffs,
//     const std::vector<int>& coeff_counts,
//     float target_kbps,
//     int sr,
//     int level,
//     bool is_side_channel)
// {
//     const float eps = 1e-12f;

//     int band_count = (int)coeffs.size();
//     if (prev_coeffs.size() != coeffs.size())
//         prev_coeffs = coeffs;

//     static thread_local std::vector<float> priority;
//     priority.resize(band_count);

//     float global_energy = 0.0f;
//     for (const auto& band : coeffs)
//         for (float v : band)
//             global_energy += v * v;

//     global_energy = std::sqrt(global_energy + eps);

//     int total_bands = 1 << level;

//     std::vector<float> weight_original(total_bands, 1.0f);

//     for (int i = 0; i < total_bands; ++i) {
//         int popcnt = __builtin_popcount((unsigned)i);
//         float w = std::exp2((level - 2 * popcnt) * 0.5f);
//         weight_original[i] = w;
//     }

//     std::vector<int> perm(total_bands);
//     for (int i = 0; i < total_bands; ++i)
//         perm[i] = i ^ (i >> 1);

//     std::vector<float> freq_weight(band_count, 1.0f);

//     for (int i = 0; i < total_bands && i < band_count; ++i) {
//         int mapped = perm[i];
//         if (mapped < band_count) {
//             freq_weight[mapped] = weight_original[i] * 0.75f;
//         }
//     }

//     for (int i = 0; i < band_count; ++i) {

//         const auto& cur = coeffs[i];
//         const auto& prev = prev_coeffs[i];

//         float max_v = 0.0f;
//         float l1 = 0.0f;

//         for (float v : cur) {
//             float a = std::fabs(v);
//             max_v = std::max(max_v, a);
//             l1 += a;
//         }

//         float mean = l1 / std::max<size_t>(cur.size(), 1);

//         float peakiness =
//             (max_v - mean) / (max_v + eps);
//         peakiness = std::clamp(peakiness, 0.0f, 1.0f);

//         float abs_energy = 0.0f;
//         for (float v : cur)
//             abs_energy += v * v;

//         float energy_local =
//             std::sqrt(abs_energy + eps) / (global_energy + eps);

//         energy_local = std::clamp(energy_local, 0.0f, 1.0f);

//         float structure =
//             0.65f * peakiness +
//             0.35f * energy_local;

//         float corr = 0.0f;
//         float norm_prev = 0.0f;
//         float norm_cur = 0.0f;

//         size_t N = std::min(cur.size(), prev.size());

//         for (size_t k = 0; k < N; ++k) {
//             corr += cur[k] * prev[k];
//             norm_prev += prev[k] * prev[k];
//             norm_cur += cur[k] * cur[k];
//         }

//         float predict =
//             corr / (std::sqrt(norm_prev * norm_cur) + eps);

//         predict = std::clamp(predict, 0.0f, 1.0f);

//         float l2 = 0.0f;
//         for (float v : cur)
//             l2 += v * v;

//         l2 = std::sqrt(l2 + eps);

//         float sparsity =
//             (l1 * l1) /
//             (cur.size() * l2 * l2 + eps);

//         sparsity = std::clamp(sparsity, 0.0f, 1.0f);

//         float energy_sum = 0.0f;
//         for (float v : cur)
//             energy_sum += v * v;

//         float entropy = 0.0f;

//         for (float v : cur) {
//             float p = (v * v) / (energy_sum + eps);
//             p = std::clamp(p, eps, 1.0f);
//             entropy += -p * std::log(p);
//         }

//         entropy /= std::log((float)cur.size() + eps);

//         float wav_entropy =
//             std::clamp(entropy, 0.0f, 1.0f);

//         float entropy_feature = (1.0f - wav_entropy);

//         float p_val =
//             0.15f * predict +
//             0.25f * sparsity +
//             0.10f * entropy_feature;

//         p_val *= freq_weight[i];
//         p_val = std::tanh(1.3f * p_val);

//         priority[i] = p_val;
//     }

//     static thread_local std::vector<float> prev_priority;

//     if (prev_priority.size() != priority.size())
//         prev_priority = priority;

//     for (int i = 0; i < band_count; ++i)
//         priority[i] =
//             0.70f * priority[i] +
//             0.30f * prev_priority[i];

//     prev_priority = priority;
//     prev_coeffs = coeffs;

//     float sum = 0.0f;
//     for (float v : priority)
//         sum += v;

//     if (sum < eps) {
//         std::fill(priority.begin(), priority.end(), 1.0f / band_count);
//         return priority;
//     }

//     return priority;
// }

inline std::vector<float> compute_channel_priority(
    const std::vector<std::vector<float>>& coeffs,
    std::vector<std::vector<float>>& prev_coeffs,
    const std::vector<int>& coeff_counts,
    float target_kbps,
    int sr,
    int level,
    bool is_side_channel)
{
    const float eps = 1e-12f;

    int band_count = (int)coeffs.size();
    if (prev_coeffs.size() != coeffs.size())
        prev_coeffs = coeffs;

    static thread_local std::vector<float> priority;
    priority.resize(band_count);

    float global_energy = 0.0f;
    for (const auto& band : coeffs)
        for (float v : band)
            global_energy += v * v;

    global_energy = std::sqrt(global_energy + eps);

    int total_bands = 1 << level;

    std::vector<float> weight_original(total_bands, 1.0f);

    for (int i = 0; i < total_bands; ++i) {
        int popcnt = __builtin_popcount((unsigned)i);
        float w = std::exp2((level - 2 * popcnt) * 0.5f);
        weight_original[i] = w;
    }

    std::vector<int> perm(total_bands);
    for (int i = 0; i < total_bands; ++i)
        perm[i] = i ^ (i >> 1);

    std::vector<float> freq_weight(band_count, 1.0f);

    for (int i = 0; i < total_bands && i < band_count; ++i) {
        int mapped = perm[i];
        if (mapped < band_count) {
            freq_weight[mapped] = weight_original[i] * 0.75f;
        }
    }

    for (int i = 0; i < band_count; ++i) {

        const auto& cur = coeffs[i];
        const auto& prev = prev_coeffs[i];

        float max_v = 0.0f;
        float l1 = 0.0f;

        for (float v : cur) {
            float a = std::fabs(v);
            max_v = std::max(max_v, a);
            l1 += a;
        }

        float mean = l1 / std::max<size_t>(cur.size(), 1);

        float peakiness =
            (max_v - mean) / (max_v + eps);
        peakiness = std::clamp(peakiness, 0.0f, 1.0f);

        float abs_energy = 0.0f;
        for (float v : cur)
            abs_energy += v * v;

        float energy_local =
            std::sqrt(abs_energy + eps) / (global_energy + eps);

        energy_local = std::clamp(energy_local, 0.0f, 1.0f);

        float structure =
            0.65f * peakiness +
            0.35f * energy_local;

        float corr = 0.0f;
        float norm_prev = 0.0f;
        float norm_cur = 0.0f;

        size_t N = std::min(cur.size(), prev.size());

        for (size_t k = 0; k < N; ++k) {
            corr += cur[k] * prev[k];
            norm_prev += prev[k] * prev[k];
            norm_cur += cur[k] * cur[k];
        }

        float predict =
            corr / (std::sqrt(norm_prev * norm_cur) + eps);

        predict = std::clamp(predict, 0.0f, 1.0f);

        float l2 = 0.0f;
        for (float v : cur)
            l2 += v * v;

        l2 = std::sqrt(l2 + eps);

        float sparsity =
            (l1 * l1) /
            (cur.size() * l2 * l2 + eps);

        sparsity = std::clamp(sparsity, 0.0f, 1.0f);

        float inter_scale = 0.5f;

        if (i > 0 && i < band_count - 1) {

            auto band_energy = [&](int b) -> float {
                float e = 0.0f;
                for (float v : coeffs[b]) e += v * v;
                return std::sqrt(e + eps);
            };

            float e_prev = band_energy(i - 1);
            float e_next = band_energy(i + 1);
            float e_cur  = std::sqrt(abs_energy + eps);

            float avg_neighbors = 0.5f * (e_prev + e_next);

            inter_scale =
                1.0f - std::fabs(e_cur - avg_neighbors) /
                       (avg_neighbors + eps);

            inter_scale = std::clamp(inter_scale, 0.0f, 1.0f);
        }

        float p_val =
            0.35f * predict +        
            0.55f * sparsity +       
            0.20f * structure +      
            0.20f * inter_scale;     

        p_val *= freq_weight[i];
        p_val = std::tanh(1.3f * p_val);

        priority[i] = p_val;
    }

    static thread_local std::vector<float> prev_priority;

    if (prev_priority.size() != priority.size())
        prev_priority = priority;

    for (int i = 0; i < band_count; ++i)
        priority[i] =
            0.70f * priority[i] +
            0.30f * prev_priority[i];

    prev_priority = priority;
    prev_coeffs = coeffs;

    float sum = 0.0f;
    for (float v : priority)
        sum += v;

    if (sum < eps) {
        std::fill(priority.begin(), priority.end(), 1.0f / band_count);
        return priority;
    }

    return priority;
}

