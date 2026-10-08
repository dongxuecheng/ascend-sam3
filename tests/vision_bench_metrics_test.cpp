#include "common/visionBenchMetrics.hpp"
#include <cassert>
#include <limits>
#include <iostream>

int main()
{
    const auto s = sam3::bench::summarize({4, 1, 3, 2});
    assert(s.mean == 2.5 && s.p50 == 2.5);
    assert(std::abs(s.p95 - 3.85) < 1e-12);
    assert(s.minimum == 1 && s.maximum == 4);
    bool rejected = false;
    try { sam3::bench::summarize({}); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    sam3::bench::Difference exact;
    exact.add(1, 1, 0, 0); exact.add(0, 0, 0, 0);
    assert(exact.passed() && exact.cosine() == 1 && exact.mae() == 0);
    sam3::bench::Difference approximate;
    approximate.add(1.001f, 1, .002, 0);
    assert(approximate.passed());
    approximate.add(0.01f, 0, .002, 100);
    assert(!approximate.passed() && approximate.mismatches == 1);
    sam3::bench::Difference invalid;
    invalid.add(std::numeric_limits<float>::quiet_NaN(), 0, 100, 100);
    invalid.add(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 100, 100);
    assert(!invalid.passed() && invalid.nonfinite == 2 && invalid.mismatches == 2);
    sam3::bench::Difference zero;
    zero.add(0, 0, 0, 0);
    assert(zero.cosine() == 1);
    std::cout << "Vision benchmark metrics tests passed\n";
}
