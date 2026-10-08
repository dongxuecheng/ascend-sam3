#pragma once

// Host-only helpers, also tested without CANN/OpenCV.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sam3::bench
{
struct Summary
{
    double mean = 0, p50 = 0, p95 = 0, minimum = 0, maximum = 0;
};

inline Summary summarize(std::vector<double> values)
{
    if (values.empty()) throw std::invalid_argument("No benchmark samples");
    double sum = 0;
    for (double value : values)
    {
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid timing");
        sum += value;
    }
    std::sort(values.begin(), values.end());
    auto quantile = [&](double q) {
        const double index = q * (values.size() - 1);
        const auto lower = static_cast<std::size_t>(index);
        const auto upper = std::min(lower + 1, values.size() - 1);
        return values[lower] + (values[upper] - values[lower]) * (index - lower);
    };
    return {sum / values.size(), quantile(0.5), quantile(0.95), values.front(), values.back()};
}

struct Difference
{
    std::size_t elements = 0, mismatches = 0, nonfinite = 0;
    double max_abs = 0, abs_sum = 0, squared_error = 0, dot = 0, norm_a = 0, norm_b = 0;

    void add(float actual, float reference, double atol, double rtol)
    {
        ++elements;
        if (!std::isfinite(actual) || !std::isfinite(reference))
        {
            ++nonfinite;
            ++mismatches;
            return;
        }
        const double a = actual, b = reference, error = std::abs(a - b);
        max_abs = std::max(max_abs, error);
        abs_sum += error;
        squared_error += error * error;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        if (error > atol + rtol * std::abs(b)) ++mismatches;
    }
    bool passed() const { return elements != 0 && mismatches == 0; }
    double mae() const { return elements ? abs_sum / elements : 0; }
    double rmse() const { return elements ? std::sqrt(squared_error / elements) : 0; }
    double cosine() const
    {
        if (norm_a == 0 && norm_b == 0) return 1;
        if (norm_a == 0 || norm_b == 0) return 0;
        return std::clamp(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)), -1.0, 1.0);
    }
};
} // namespace sam3::bench
