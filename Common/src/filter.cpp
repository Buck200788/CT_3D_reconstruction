#include "filter.h"
#include <fstream>

Filter::Filter(int len, FilterType filter, float d)
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
    //return;
    //std::ofstream outs("E:\\test.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(vec_d.data()), sizeof(double) * vec_d.size());
    //std::vector<double>fft_real(fftRes.size(), 0);
    //for (size_t i = 0; i < fftRes.size(); i++) fft_real[i] = fftRes[i].real();
    //outs.write(reinterpret_cast<const char*>(fft_real.data()), sizeof(double) * vec_d.size());
    //outs.close();

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