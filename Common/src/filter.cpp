#include "filter.h"
#include <fstream>

#include "filter.h"
#include <cmath>
#include <algorithm>

// Spatial-domain ramp kernel.  CpuFDKRecon performs a spatial convolution,
// therefore GetFilter() must return h[k], not the FFT/frequency response.
void Filter::Filter_(int len, FilterType filter, float detector_spacing, int type)
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

Filter::Filter(int len, FilterType filter, double spacing, int type, double cutoffRatio)
{
    if (len <= 0)
        throw std::invalid_argument("len must be positive");

    if (!(spacing > 0.0))
        throw std::invalid_argument(
            "spacing must be positive"
        );

    if (!(cutoffRatio > 0.0 &&
        cutoffRatio <= 1.0))
    {
        throw std::invalid_argument(
            "cutoffRatio must be in (0, 1]"
        );
    }

    const double pi = std::acos(-1.0);
    const int order = nextpow2(2 * len);
    const int halfLen = order / 2 + 1;
    // odd number kernel
    const int fftLength =  2 * halfLen - 1;
    filt.assign(fftLength, 0.0f);
    std::vector<float> halfKernel(halfLen, 0.0f);
    halfKernel[0] = static_cast<float>( 0.25 / (spacing * spacing));
    for (int i = 1; i < halfLen; i += 2)
    {
        double value = 0.0;
        if (type == 0)
        {
            const double x = pi * i * spacing;
            value = -1.0 / (x * x);
        }
        else if (type==1)
        {
            const double angle = i * spacing;
            const double s = std::sin(angle);
            if (std::abs(s) < 1.0e-12)
            {
                throw std::runtime_error("Angular filter singularity");
            }
            const double x = pi * s;
            value = -1.0 / (x * x);
        }
        else
        {
            throw std::invalid_argument("type must be 0 for equidistant or 1 for equiangular");
        }

        halfKernel[i] = static_cast<float>(value);
    }

    std::copy( halfKernel.begin(),halfKernel.end(),filt.begin());
    for (int i = halfLen; i < fftLength; ++i)
    {
        filt[i] = halfKernel[2 * halfLen - 1 - i];
    }

    window_filter(filt, filter, cutoffRatio);
}

void Filter::window_filter(std::vector<float>& filt, FilterType filter, double cutoffRatio)
{
    const int fftLength = filt.size();
    const int halfLen = fftLength / 2 + 1;
    const double pi = std::acos(-1.0);

    std::vector<double> spatial(filt.begin(), filt.end());
    FFT1D fftSolver;
    auto spectrum = fftSolver.fft(spatial, false);
    std::vector<float> halfSpectrum(halfLen, 0.0f);
    for (int i = 0; i < halfLen; ++i)
    {
        halfSpectrum[i] = 2.0f * static_cast<float>(spectrum[i].real());
    }
    for (int i = 1; i < halfLen; ++i)
    {
        const double normalizedFreq = 2.0 * i / (cutoffRatio * fftLength);
        if (normalizedFreq > 1.0)
        {
            halfSpectrum[i] = 0.0f;
            continue;
        }
        float win = 1.0f;
        switch (filter)
        {
        case FilterType::RamLak:
            win = 1.0f;
            break;
        case FilterType::SheppLogan:
        {
            const double x = 0.5 * pi * normalizedFreq;
            win = std::abs(x) < 1.0e-12 ? 1.0f : static_cast<float>(std::sin(x) / x);
            break;
        }
        case FilterType::Cosine:
            win = static_cast<float>(std::cos(0.5 * pi * normalizedFreq));
            break;
        case FilterType::Hamming:
            win = static_cast<float>(0.54 + 0.46 * std::cos(pi * normalizedFreq));
            break;
        case FilterType::Hann:
            win = static_cast<float>(0.5 + 0.5 * std::cos(pi * normalizedFreq));
            break;
        default:
            win = 1.0f;
            break;
        }
        halfSpectrum[i] *= win;
    }
    std::copy(halfSpectrum.begin(), halfSpectrum.end(), filt.begin());
    for (int k = halfLen; k < fftLength; ++k)
    {
        filt[k] = halfSpectrum[2 * halfLen - 1 - k];
    }
    std::vector<double> windowedSpectrum(filt.begin(), filt.end());
    auto spatialResult = fftSolver.ifft(windowedSpectrum, false);
    const int centerShift = halfLen;

    for (int i = 0; i < fftLength; ++i)
    {
        const int idx = (i + centerShift) % fftLength;
        filt[i] = static_cast<float>(spatialResult[idx].real());
    }
}