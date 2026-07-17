#pragma once
#include <chrono>
#include <vector>
#include <cmath>

class Timer
{
    std::chrono::high_resolution_clock::time_point m_st;
public:
    void Start() { m_st = std::chrono::high_resolution_clock::now(); }
    double ElapsedMs()
    {
        auto ed = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(ed - m_st).count();
    }
};

inline float CalcMaxError(const std::vector<float>& a, const std::vector<float>& b)
{
    float maxErr = 0.f;
    int N = a.size();
    for (int i = 0; i < N; i++)
        maxErr = std::max(maxErr, fabs(a[i] - b[i]));
    return maxErr;
}