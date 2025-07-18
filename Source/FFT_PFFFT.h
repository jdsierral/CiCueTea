//
//  FFT_PFFFT.h
//  CQTDSP
//
//  Created by Juan Sierra on 4/8/25.
//

/**
 * @file FFT_PFFFT.h
 * @brief Provides a pImpl based implemenation wrapping PFFFT
 * @author Juan Sierra
 * @date 4/8/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>
#include <pffft_double.h>

using namespace Eigen;

namespace jsa {
class DFTImpl
{
  public:
    DFTImpl(size_t fftSize) :
        fftSize(fftSize)
    {
        work.resize(2 * fftSize);
        complexSetup = pffftd_new_setup(int(fftSize), PFFFT_COMPLEX);
        realSetup    = pffftd_new_setup(int(fftSize), PFFFT_REAL);
    }

    ~DFTImpl()
    {
        if (complexSetup) pffftd_destroy_setup(complexSetup);
        if (realSetup) pffftd_destroy_setup(realSetup);
    }

    void dft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        double* inPtr_  = reinterpret_cast<double*>(const_cast<dcomplex*>(inPtr));
        double* outPtr_ = reinterpret_cast<double*>(outPtr);
        pffftd_transform_ordered(complexSetup, inPtr_, outPtr_, work.data(), PFFFT_FORWARD);
    }

    void idft(const dcomplex* inPtr, dcomplex* outPtr)
    {
        double* inPtr_  = reinterpret_cast<double*>(const_cast<dcomplex*>(inPtr));
        double* outPtr_ = reinterpret_cast<double*>(outPtr);
        pffftd_transform_ordered(complexSetup, inPtr_, outPtr_, work.data(), PFFFT_BACKWARD);
        Map<ArrayXcd>(outPtr, fftSize) *= (1.0 / fftSize);
    }

    void rdft(const double* inPtr, dcomplex* outPtr)
    {
        double* inPtr_  = reinterpret_cast<double*>(const_cast<double*>(inPtr));
        double* outPtr_ = reinterpret_cast<double*>(outPtr);
        pffftd_transform_ordered(realSetup, inPtr_, outPtr_, work.data(), PFFFT_FORWARD);
    }

    void irdft(const dcomplex* inPtr, double* outPtr)
    {
        double* inPtr_  = reinterpret_cast<double*>(const_cast<dcomplex*>(inPtr));
        double* outPtr_ = reinterpret_cast<double*>(outPtr);
        pffftd_transform_ordered(realSetup, inPtr_, outPtr_, work.data(), PFFFT_BACKWARD);
        Map<ArrayXd>(outPtr, fftSize) *= (1.0 / fftSize);
    }

  private:
    const size_t   fftSize;
    Eigen::ArrayXd work;
    PFFFTD_Setup*  complexSetup = nullptr;
    PFFFTD_Setup*  realSetup    = nullptr;
};
} // namespace jsa
