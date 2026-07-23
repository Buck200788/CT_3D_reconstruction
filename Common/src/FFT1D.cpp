#include "FFT1D.h"
#include <cmath>

FFT1D::FFT1D() : m_PI(std::acos(-1.0)) {}

void FFT1D::bitReverseSwap(std::vector<cd>& x) const
{
    int n = static_cast<int>(x.size());
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) std::swap(x[i], x[j]);
    }
}

std::vector<cd> FFT1D::padToPower2(const std::vector<cd>& input) const
{
    std::vector<cd> res = input;
    int sz = 1;
    while (sz < static_cast<int>(res.size())) sz <<= 1;
    res.resize(sz, cd(0.0, 0.0));
    return res;
}

void FFT1D::fftCoreRadix2(std::vector<cd>& data, bool invert) const
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
            for (int j = 0; j < half; j++)
            {
                cd u = data[i + j];
                cd v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert)
        for (auto& val : data) val /= n;
}

// Bluestein Chirp-Z 任意长度FFT核心实现
std::vector<cd> FFT1D::bluesteinFFT(const std::vector<cd>& in, bool invert) const
{
    int N = (int)in.size();
    if (N <= 1) return in;
    int M = 1;
    while (M < 2 * N - 1) M <<= 1;

    std::vector<cd> a(M, cd(0, 0)), b(M, cd(0, 0));
    double sign = invert ? -1.0 : 1.0;
    for (int n = 0; n < N; n++)
    {
        double theta = sign * m_PI * n * n / N;
        cd w(std::cos(theta), std::sin(theta));
        a[n] = in[n] * w;
        b[n] = std::conj(w);
    }
    for (int n = M - N + 1; n < M; n++)
    {
        int k = M - n;
        double theta = sign * m_PI * k * k / N;
        b[n] = std::conj(cd(std::cos(theta), std::sin(theta)));
    }

    // 基2FFT做卷积
    fftCoreRadix2(a, false);
    fftCoreRadix2(b, false);
    for (int i = 0; i < M; i++) a[i] *= b[i];
    fftCoreRadix2(a, true);

    std::vector<cd> out(N);
    for (int k = 0; k < N; k++)
    {
        double theta = sign * m_PI * k * k / N;
        cd w_k(std::cos(theta), std::sin(theta));
        out[k] = a[k] * w_k;
        if (invert) out[k] /= N;
    }
    return out;
}

std::vector<cd> FFT1D::transform(const std::vector<double>& signal, bool invert, bool autoPad)
{
    std::vector<cd> data(signal.begin(), signal.end());
    return transform(data, invert, autoPad);
}

std::vector<cd> FFT1D::transform(const std::vector<cd>& signal, bool invert, bool autoPad)
{
    int N = (int)signal.size();
    // 开启补零：沿用旧基2逻辑
    if (autoPad)
    {
        auto data = padToPower2(signal);
        fftCoreRadix2(data, invert);
        return data;
    }
    // 不补零：走Bluestein任意长度DFT，和numpy完全对齐
    return bluesteinFFT(signal, invert);
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