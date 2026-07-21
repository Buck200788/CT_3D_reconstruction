#include "filter.h"

inline Filter::Filter(int len, FilterType filter, float d)
{
    int order = std::max(64, nextpow2(2 * len));
    int halfLen = order / 2 + 1;
    std::vector<float> filt(order, 0.f);

    if (filter == FilterType::None)
    {
        std::fill(filt.begin(), filt.end(), 1.f);
    }
    const float pi = (float)M_PI;
    for (int k = 0; k < halfLen; k++)
    {
        float w = 2.f * pi * k / order;
        float ramp = std::fabs(w); // »ù´¡Ð±ÆÂ·ùÖµ
        float win = 1.f;

        if (w > pi * d)
        {
            filt[k] = 0.f;
            continue;
        }

        switch (filter)
        {
        case FilterType::RamLak:
            win = 1.f;
            break;
        case FilterType::SheppLogan:
        {
            float x = w / (2.f * d);
            win = std::sin(x) / (x + 1e-9f);
            break;
        }
        case FilterType::Cosine:
        {
            float x = w / (2.f * d);
            win = std::cos(x);
            break;
        }
        case FilterType::Hamming:
            win = 0.54f + 0.46f * std::cos(w / d);
            break;
        case FilterType::Hann:
            win = (1.f + std::cos(w / d)) * 0.5f;
            break;
        default:
            win = 1.f;
        }
        filt[k] = ramp * win;
    }

    for (int k = halfLen; k < order; k++)
    {
        filt[k] = filt[2 * halfLen - 1 - k];
    }
}