#include "filter.h"
#include <fstream>

#include "filter.h"
#include <cmath>
#include <algorithm>

// Spatial-domain ramp kernel.  CpuFDKRecon performs a spatial convolution,
// therefore GetFilter() must return h[k], not the FFT/frequency response.
Filter::Filter(int len, FilterType filter, float detector_spacing)
{
    const double pi = std::acos(-1.0);
    const int radius = len - 1;
    const int filter_len = 2 * radius + 1;
    filt.assign(filter_len, 0.f);

    const double d = static_cast<double>(detector_spacing);
    const double inv_d2 = 1.0 / (d * d);

    for (int k = -radius; k <= radius; ++k) {
        double h = 0.0;
        if (k == 0) {
            h = 0.25 * inv_d2;
        }
        else if ((std::abs(k) & 1) != 0) {
            h = -inv_d2 / (pi * pi * static_cast<double>(k * k));
        }

        // Ram-Lak is exact here.  Mild spatial truncation windows are supplied
        // only to keep the old API meaningful; start debugging with RamLak.
        const double t = static_cast<double>(std::abs(k)) /
            static_cast<double>(std::max(1, radius));
        double win = 1.0;
        switch (filter) {
        case FilterType::Hann:
            win = 0.5 * (1.0 + std::cos(pi * t));
            break;
        case FilterType::Hamming:
            win = 0.54 + 0.46 * std::cos(pi * t);
            break;
        case FilterType::Cosine:
            win = std::cos(0.5 * pi * t);
            break;
        case FilterType::SheppLogan:
            if (t > 0.0) win = std::sin(0.5 * pi * t) / (0.5 * pi * t);
            break;
        case FilterType::RamLak:
        default:
            break;
        }
        filt[k + radius] = static_cast<float>(h * win);
    }
}

void Filter::Filter_frequency_domain(int len, FilterType filter, float d)
{
    // 使用高精度PI，和FFT内部统一
    const double pi = std::acos(-1.0);
    // 去掉std::max(64, ...) 和Python长度逻辑对齐
    int order = nextpow2(2 * len);
    int halfLen = order / 2 + 1;
    int filter_len = 2 * halfLen - 1;
    //printf("%d %d\n", halfLen, filter_len);

    this->filt.assign(filter_len, 0.f);
    auto& filt = this->filt;
    std::vector<float> half_filt(halfLen, 0.f);
    half_filt[0] = 0.25f;

    // 改用double计算系数，消除单精度PI误差
    for (size_t i = 1; i < halfLen; i += 2) {
        double denom = (pi * i) * (pi * i);
        half_filt[i] = static_cast<float>(-1.0 / denom);
    }

    std::copy(half_filt.begin(), half_filt.end(), filt.begin());
    for (size_t i = halfLen; i < filter_len; ++i)
        filt[i] = half_filt[2*halfLen-1-i];

    std::vector<double> vec_d(filt.begin(), filt.end());
    //printf("FFT输入长度=%zu, filter_len=%d\n", vec_d.size(), filter_len);
    FFT1D fftSolver;
    auto fftRes = fftSolver.fft(vec_d, false);

    for (size_t i = 0; i < halfLen; i++) {
        half_filt[i] = 2.f * static_cast<float>(fftRes[i].real());
        //printf("%f\n", fftRes[i].real());
    }

    // 窗函数部分 pi替换为double版本
    for (size_t i = 1; i < halfLen; i++) {
        double w = 2 * pi * i / order;
        float win = 1.f;
        switch (filter)
        {
        case FilterType::RamLak:
            win = 1.f;
            break;
        case FilterType::SheppLogan:
        {
            double x = w / (2.0 * d);
            win = static_cast<float>(std::sin(x) / (x + 1e-9));
            break;
        }
        case FilterType::Cosine:
        {
            double x = w / (2.0 * d);
            win = static_cast<float>(std::cos(x));
            break;
        }
        case FilterType::Hamming:
            win = 0.54f + 0.46f * static_cast<float>(std::cos(w / d));
            break;
        case FilterType::Hann:
            win = static_cast<float>((1.0 + std::cos(w / d)) * 0.5);
            break;
        default:
            win = 1.f;
        }
        half_filt[i] *= win;
        // 截止判断使用高精度w
        if (w > pi * d)
            half_filt[i] = 0.f;
    }

    std::copy(half_filt.begin(), half_filt.end(), filt.begin());
    for (size_t k = halfLen; k < filter_len; k++)
    {
        filt[k] = half_filt[2 * halfLen - 1 - k];
        //printf("%f\n", filt[k]);
    }
}