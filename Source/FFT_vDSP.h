//
//  FFT_vDSP.h
//  CQTDSP
//
//  Created by Juan Sierra on 4/8/25.
//

/**
 * @file FFT_vDSP.h
 * @brief Provides a pImpl based implemenation wrapping Accelerate vDSP
 * @author Juan Sierra
 * @date 4/8/25
 * @copyright MIT License
 */

#pragma once

#include <Accelerate/Accelerate.h>
#include <Eigen/Core>
#include <assert.h>

using namespace Eigen;

namespace jsa {
class DFTImpl
{
  public:
    DFTImpl(size_t fftSize) :
        workData(fftSize / 2, 4),
        fftOrder(log2(fftSize)),
        fftSize(fftSize),
        inverseFactor(1.0 / double(fftSize))
    {
        assert(exp2(fftOrder) == fftSize);
        setup = vDSP_create_fftsetupD(log2(fftSize), kFFTRadix2);
        workData.setZero();
        assert(setup != nullptr);
    }

    ~DFTImpl()
    {
        if (setup) vDSP_destroy_fftsetupD(setup);
    }

    void dft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex*     inPtr_       = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex*     outPtr_      = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(2).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize);
        vDSP_fft_zipD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Forward);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize);
    }

    void idft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex*     inPtr_       = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex*     outPtr_      = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double*               mulPtr       = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(2).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize);
        vDSP_fft_zipD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Inverse);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize);
        vDSP_vsmulD(mulPtr, singleStride, &inverseFactor, mulPtr, singleStride, 2 * fftSize);
    }

    void rdft(const double* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex*     inPtr_       = reinterpret_cast<DSPDoubleComplex*>(const_cast<double*>(inPtr));
        DSPDoubleComplex*     outPtr_      = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double*               mulPtr       = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(1).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize / 2);
        vDSP_fft_zripD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Forward);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize / 2);
        vDSP_vsmulD(mulPtr, singleStride, &forwardFactor, mulPtr, singleStride, fftSize);
        outPtr_[fftSize / 2].real = outPtr_[0].imag;
        outPtr_[0].imag           = 0.0;
    }

    void irdft(const dcomplex* inPtr, double* outPtr)
    {
        DSPDoubleComplex*     inPtr_       = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex*     outPtr_      = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double*               mulPtr       = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(1).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize / 2);
        splitComplex.imagp[0] = inPtr_[fftSize / 2].real;
        vDSP_fft_zripD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Inverse);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize / 2);
        vDSP_vsmulD(mulPtr, singleStride, &inverseFactor, mulPtr, singleStride, fftSize);
    }

    static const std::string getName()
    {
        return "vDSP";
    }

  private:
    FFTSetupD         setup = nullptr;
    Eigen::ArrayX4d   workData;
    const vDSP_Length fftOrder;
    const size_t      fftSize;
    double            forwardFactor = 0.5;
    double            inverseFactor = NAN;
    const vDSP_Stride singleStride  = 1;
    const vDSP_Stride doubleStride  = 2;
};
} // namespace jsa
