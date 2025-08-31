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
#include <boost/assert.hpp>

using namespace Eigen;

namespace jsa {
class DFTImpl
{
  public:
    DFTImpl(size_t fftSize) :
        workData(fftSize/2, 4),
        fftOrder(log2(fftSize)),
        fftSize(fftSize),
        inverseFactor(1.0/double(fftSize))
    {
        BOOST_ASSERT_MSG(exp2(fftOrder) == fftSize, "FFT Size was not power of 2");
        setup = vDSP_create_fftsetupD(log2(fftSize), kFFTRadix2);
        workData.setZero();
        BOOST_ASSERT_MSG(setup, "Setup not initialized properly");
    }

    ~DFTImpl()
    {
        if (setup) vDSP_destroy_fftsetupD(setup);
    }

    void dft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex* inPtr_ = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_= reinterpret_cast<DSPDoubleComplex*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(2).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize);
        vDSP_fft_zipD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Forward);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize);
    }

    void idft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex* inPtr_ = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_= reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double* mulPtr = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(2).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize);
        vDSP_fft_zipD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Inverse);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize);
        vDSP_vsmulD(mulPtr, singleStride, &inverseFactor, mulPtr, singleStride, 2 * fftSize);
        
    }

    void rdft(const double* inPtr, dcomplex* outPtr)
    {
        DSPDoubleComplex* inPtr_ = reinterpret_cast<DSPDoubleComplex*>(const_cast<double*>(inPtr));
        DSPDoubleComplex* outPtr_= reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double* mulPtr = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(1).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize/2);
        vDSP_fft_zripD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Forward);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize/2);
        vDSP_vsmulD(mulPtr, singleStride, &forwardFactor, mulPtr, singleStride, fftSize);
        outPtr_[fftSize/2].real = outPtr_[0].imag;
        outPtr_[0].imag = 0.0;
    }

    void irdft(const dcomplex* inPtr, double* outPtr)
    {
        DSPDoubleComplex* inPtr_ = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_= reinterpret_cast<DSPDoubleComplex*>(outPtr);
        double* mulPtr = reinterpret_cast<double*>(outPtr);
        DSPDoubleSplitComplex splitComplex = {workData.col(0).data(), workData.col(1).data()};
        vDSP_ctozD(inPtr_, doubleStride, &splitComplex, singleStride, fftSize/2);
        splitComplex.imagp[0] = inPtr_[fftSize/2].real;
        vDSP_fft_zripD(setup, &splitComplex, singleStride, fftOrder, kFFTDirection_Inverse);
        vDSP_ztocD(&splitComplex, singleStride, outPtr_, doubleStride, fftSize/2);
        vDSP_vsmulD(mulPtr, singleStride, &inverseFactor, mulPtr, singleStride, fftSize);
    }
    
    static const std::string getName()
    {
        return "vDSP";
    }

  private:
    FFTSetupD setup = nullptr;
    Eigen::ArrayX4d workData;
    const vDSP_Length fftOrder;
    const size_t fftSize;
    double forwardFactor = 0.5;
    double inverseFactor = NAN;
    const vDSP_Stride singleStride = 1;
    const vDSP_Stride doubleStride = 2;
};
} // namespace jsa


//namespace jsa {
//class DFTImpl
//{
//  public:
//    DFTImpl(size_t fftSize) :
//        fftSize(fftSize)
//    {
//        r2cSetup  = vDSP_DFT_Interleaved_CreateSetupD(r2cSetup, fftSize / 2, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_RealtoComplex);
//        c2rSetup  = vDSP_DFT_Interleaved_CreateSetupD(c2rSetup, fftSize / 2, vDSP_DFT_INVERSE, vDSP_DFT_Interleaved_RealtoComplex);
//        c2cSetup  = vDSP_DFT_Interleaved_CreateSetupD(c2cSetup, fftSize, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex);
//        ic2cSetup = vDSP_DFT_Interleaved_CreateSetupD(ic2cSetup, fftSize, vDSP_DFT_INVERSE, vDSP_DFT_Interleaved_ComplextoComplex);
//        BOOST_ASSERT_MSG(r2cSetup, "Real To Complex not initialized Properly");
//        BOOST_ASSERT_MSG(c2rSetup, "Complex To Real not initialized Properly");
//        BOOST_ASSERT_MSG(c2cSetup, "Complex To Complex not initialized Properly");
//        BOOST_ASSERT_MSG(ic2cSetup, "Inverse Complex To Complex not initialized Properly");
//    }
//
//    ~DFTImpl()
//    {
//        if (r2cSetup) vDSP_DFT_Interleaved_DestroySetupD(r2cSetup);
//        if (c2rSetup) vDSP_DFT_Interleaved_DestroySetupD(c2rSetup);
//        if (c2cSetup) vDSP_DFT_Interleaved_DestroySetupD(c2cSetup);
//        if (ic2cSetup) vDSP_DFT_Interleaved_DestroySetupD(ic2cSetup);
//    }
//
//    void dft(const dcomplex* inPtr, dcomplex* outPtr)
//    {
//        DSPDoubleComplex* inPtr_  = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
//        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
//        vDSP_DFT_Interleaved_ExecuteD(c2cSetup, inPtr_, outPtr_);
//    }
//
//    void idft(const dcomplex* inPtr, dcomplex* outPtr)
//    {
//        DSPDoubleComplex* inPtr_  = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
//        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
//        vDSP_DFT_Interleaved_ExecuteD(ic2cSetup, inPtr_, outPtr_);
//        Map<ArrayXcd>(outPtr, fftSize) *= (1.0 / fftSize);
//    }
//
//    void rdft(const double* inPtr, dcomplex* outPtr)
//    {
//        DSPDoubleComplex* inPtr_  = reinterpret_cast<DSPDoubleComplex*>(const_cast<double*>(inPtr));
//        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
//        vDSP_DFT_Interleaved_ExecuteD(r2cSetup, inPtr_, outPtr_);
//        Map<ArrayXcd>(outPtr, fftSize / 2 + 1) *= (0.5);
//    }
//
//    void irdft(const dcomplex* inPtr, double* outPtr)
//    {
//        DSPDoubleComplex* inPtr_  = reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
//        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
//        vDSP_DFT_Interleaved_ExecuteD(c2rSetup, inPtr_, outPtr_);
//        Map<ArrayXd>(outPtr, fftSize) *= (1.0 / fftSize);
//    }
//
//  private:
//    const size_t fftSize;
//
//    vDSP_DFT_Interleaved_SetupD r2cSetup  = nullptr;
//    vDSP_DFT_Interleaved_SetupD c2rSetup  = nullptr;
//    vDSP_DFT_Interleaved_SetupD c2cSetup  = nullptr;
//    vDSP_DFT_Interleaved_SetupD ic2cSetup = nullptr;
//};
//} // namespace jsa
