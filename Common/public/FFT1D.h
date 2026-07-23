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
    // 基2FFT内部实现（保留用于Bluestein内部2次幂FFT）
    void bitReverseSwap(std::vector<cd>& x) const;
    std::vector<cd> padToPower2(const std::vector<cd>& input) const;
    void fftCoreRadix2(std::vector<cd>& data, bool invert) const;
    // Bluestein任意长度FFT
    std::vector<cd> bluesteinFFT(const std::vector<cd>& in, bool invert) const;

public:
    FFT1D();

    // 通用变换：autoPad=false时走Bluestein任意长度FFT
    std::vector<cd> transform(const std::vector<double>& signal, bool invert = false, bool autoPad = false);
    std::vector<cd> transform(const std::vector<cd>& signal, bool invert = false, bool autoPad = false);

    std::vector<cd> fft(const std::vector<double>& signal, bool autoPad = false);
    std::vector<cd> fft(const std::vector<cd>& signal, bool autoPad = false);
    std::vector<cd> ifft(const std::vector<double>& signal, bool autoPad = false);
    std::vector<cd> ifft(const std::vector<cd>& signal, bool autoPad = false);
};
#endif