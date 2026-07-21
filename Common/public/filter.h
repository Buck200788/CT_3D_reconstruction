#pragma once
#define _USE_MATH_DEFINES
#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>

enum class FilterType
{
    RamLak,
    SheppLogan,
    Cosine,
    Hamming,
    Hann,
    None
};

inline FilterType Str2FilterType(const std::string& s)
{
    if (s == "ram-lak")    return FilterType::RamLak;
    if (s == "shepp-logan")return FilterType::SheppLogan;
    if (s == "cosine")     return FilterType::Cosine;
    if (s == "hamming")    return FilterType::Hamming;
    if (s == "hann")       return FilterType::Hann;
    if (s == "none")       return FilterType::None;
    throw std::invalid_argument("invalid filter type");
}

int nextpow2(int x)
{
    if (x <= 1) return 1;
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

class Filter
{
private:
    std::vector<float> filt; // Ë½ÓÐÂË²¨ºËÊý×é
public:
    explicit Filter(int len, FilterType filter, float d = 1.0f);
    const std::vector<float>& GetFilter() const { return filt; }
    const float* Data() const { return filt.data(); }
    size_t Size() const { return filt.size(); }
};