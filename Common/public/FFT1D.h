#pragma once
#ifndef FFT1D_H
#define FFT1D_H

#include <vector>
#include <complex>

using cd = std::complex<double>;

class FFT1D
{
private:
    const double m_PI;

    // 私有：位逆序置换（原地修改数组）
    void bitReverseSwap(std::vector<cd>& x) const;

    // 私有：补零至最近2的整数次幂
    std::vector<cd> padToPower2(const std::vector<cd>& input) const;

    // 私有核心蝶形运算
    void fftCore(std::vector<cd>& data, bool invert) const;

public:
    FFT1D();

    // 通用变换接口
    std::vector<cd> transform(const std::vector<double>& signal, bool invert = false, bool autoPad = true);
    std::vector<cd> transform(const std::vector<cd>& signal, bool invert = false, bool autoPad = true);

    // 简化对外接口：正FFT
    std::vector<cd> fft(const std::vector<double>& signal, bool autoPad = true);
    std::vector<cd> fft(const std::vector<cd>& signal, bool autoPad = true);

    // 简化对外接口：逆IFFT
    std::vector<cd> ifft(const std::vector<double>& signal, bool autoPad = true);
    std::vector<cd> ifft(const std::vector<cd>& signal, bool autoPad = true);
};

#endif