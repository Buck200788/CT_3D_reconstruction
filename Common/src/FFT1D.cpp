#include "FFT1D.h"
#include <cmath>

FFT1D::FFT1D() : m_PI(std::acos(-1.0))
{
}

void FFT1D::bitReverseSwap(std::vector<cd>& x) const
{
    int n = static_cast<int>(x.size());
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;
        j += bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
}

std::vector<cd> FFT1D::padToPower2(const std::vector<cd>& input) const
{
    std::vector<cd> res = input;
    int sz = 1;
    while (sz < static_cast<int>(res.size()))
        sz <<= 1;
    res.resize(sz, cd(0.0, 0.0));
    return res;
}

void FFT1D::fftCore(std::vector<cd>& data, bool invert) const
{
    int n = static_cast<int>(data.size());
    bitReverseSwap(data);

    for (int len = 2; len <= n; len <<= 1)
    {
        double ang = 2 * m_PI / len * (invert ? -1.0 : 1.0);
        cd wlen(std::cos(ang), std::sin(ang));

        for (int i = 0; i < n; i += len)
        {
            cd w(1.0, 0.0);
            int half = len / 2;
            for (int j = 0; j < half; ++j)
            {
                cd u = data[i + j];
                cd v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    // IFFT¹éÒ»»¯
    if (invert)
    {
        for (auto& val : data)
            val /= n;
    }
}

std::vector<cd> FFT1D::transform(const std::vector<double>& signal, bool invert, bool autoPad)
{
    std::vector<cd> data(signal.begin(), signal.end());
    if (autoPad)
        data = padToPower2(data);
    fftCore(data, invert);
    return data;
}

std::vector<cd> FFT1D::transform(const std::vector<cd>& signal, bool invert, bool autoPad)
{
    std::vector<cd> data = signal;
    if (autoPad)
        data = padToPower2(data);
    fftCore(data, invert);
    return data;
}

std::vector<cd> FFT1D::fft(const std::vector<double>& signal, bool autoPad)
{
    return transform(signal, false, autoPad);
}

std::vector<cd> FFT1D::fft(const std::vector<cd>& signal, bool autoPad)
{
    return transform(signal, false, autoPad);
}

std::vector<cd> FFT1D::ifft(const std::vector<double>& signal, bool autoPad)
{
    return transform(signal, true, autoPad);
}

std::vector<cd> FFT1D::ifft(const std::vector<cd>& signal, bool autoPad)
{
    return transform(signal, true, autoPad);
}