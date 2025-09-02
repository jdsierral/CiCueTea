#pragma once

#include <Eigen/Core>
//#include <algorithm>

struct CqtConfig {
    void verify()
    {
        //        if (fs / (fMin * (exp2(1.0 / ppo) - 1.0)) > blockSize)
        //            blockSize *= 2;
        blockSize = std::clamp(blockSize, 1UL << 10, 1UL << 16);
        ppo       = std::clamp(ppo, 1.f, 96.f);
        fMin      = std::clamp(fMin, 20.f, 20000.f);
        fMax      = std::clamp(fMax, 20.f, 20000.f);
    }

    void setSampleRate(double newSampleRate)
    {
        fs = newSampleRate;
        verify();
    }

    void setBlockSize(size_t newBlockSize)
    {
        if (fs / (fMin * (exp2(1.0 / ppo) - 1.0)) > newBlockSize) return false;
        size_t minBlockSize = size_t(fs / (fMin * (exp2(1.0 / ppo) - 1.0)));
        blockSize           = std::clamp(newBlockSize, minBlockSize, 65536UL);
        verify();
    }

    void setPpo(float newPpo)
    {
        //        if (fs / (fMin * (exp2(1.0 / newPpo) - 1.0)) > blockSize) return false;
        //        ppo = newPpo;
        float maxPpo = 1.0 / log2(fs / (blockSize * fMin) + 1);
        ppo          = std::clamp(newPpo, 1.f, maxPpo);
        verify();
    }

    void setMinFrequency(float newFMin)
    {
        fMin = std::clamp(newFMin, 20.f, fMax / 2.f);
        verify();
    }

    void setMaxFrequency(float newFMax)
    {
        fMax = std::clamp(newFMax, fMin * 2.f, 20000.f);
        verify();
    }

    void setRefFrequency(float newFRef)
    {
        fRef = std::clamp(newFRef, fMin, fMax);
        verify();
    }

    double fs        = -1;
    size_t blockSize = 1 << 12;
    float  ppo       = 12;
    float  fMin      = 100;
    float  fMax      = 10000;
    float  fRef      = 1000;
};
