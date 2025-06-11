//
//  FFT_vDSP.h
//  CQTDSP
//
//  Created by Juan Sierra on 4/8/25.
//

#pragma once

#include <Accelerate/Accelerate.h>
#include <Eigen/Core>

using namespace Eigen;

namespace jsa {
class DFTImpl {
public:
    DFTImpl(size_t fftSize) : fftSize(fftSize) {
        r2cSetup = vDSP_DFT_Interleaved_CreateSetupD(
                                                     r2cSetup, fftSize / 2, vDSP_DFT_FORWARD,
                                                     vDSP_DFT_Interleaved_RealtoComplex);
        c2rSetup = vDSP_DFT_Interleaved_CreateSetupD(
                                                     c2rSetup, fftSize / 2, vDSP_DFT_INVERSE,
                                                     vDSP_DFT_Interleaved_RealtoComplex);
        c2cSetup = vDSP_DFT_Interleaved_CreateSetupD(
                                                     c2cSetup, fftSize, vDSP_DFT_FORWARD,
                                                     vDSP_DFT_Interleaved_ComplextoComplex);
        ic2cSetup = vDSP_DFT_Interleaved_CreateSetupD(
                                                      ic2cSetup, fftSize, vDSP_DFT_INVERSE,
                                                      vDSP_DFT_Interleaved_ComplextoComplex);
    }
    
    ~DFTImpl() {
        if (r2cSetup) vDSP_DFT_Interleaved_DestroySetupD(r2cSetup);
        if (c2rSetup) vDSP_DFT_Interleaved_DestroySetupD(c2rSetup);
        if (c2cSetup) vDSP_DFT_Interleaved_DestroySetupD(c2cSetup);
        if (ic2cSetup) vDSP_DFT_Interleaved_DestroySetupD(ic2cSetup);
    }
    
    void dft(const dcomplex* inPtr, dcomplex* outPtr) {
        DSPDoubleComplex* inPtr_ =
        reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        vDSP_DFT_Interleaved_ExecuteD(c2cSetup, inPtr_, outPtr_);
    }
    
    void idft(const dcomplex *inPtr, dcomplex *outPtr) {
        DSPDoubleComplex* inPtr_ =
        reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        vDSP_DFT_Interleaved_ExecuteD(ic2cSetup, inPtr_, outPtr_);
        Map<ArrayXcd>(outPtr, fftSize) *= (1.0 / fftSize);
    }
    
    void rdft(const double* inPtr, dcomplex* outPtr) {
        DSPDoubleComplex* inPtr_ =
        reinterpret_cast<DSPDoubleComplex*>(const_cast<double*>(inPtr));
        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        vDSP_DFT_Interleaved_ExecuteD(r2cSetup, inPtr_, outPtr_);
        Map<ArrayXcd>(outPtr, fftSize/2+1) *= (0.5);
    }
    
    void irdft(const dcomplex* inPtr, double* outPtr) {
        DSPDoubleComplex* inPtr_ =
        reinterpret_cast<DSPDoubleComplex*>(const_cast<dcomplex*>(inPtr));
        DSPDoubleComplex* outPtr_ = reinterpret_cast<DSPDoubleComplex*>(outPtr);
        vDSP_DFT_Interleaved_ExecuteD(c2rSetup, inPtr_, outPtr_);
        Map<ArrayXd>(outPtr, fftSize) *= (1.0 / fftSize);
    }
    
private:
    const size_t fftSize;
    vDSP_DFT_Interleaved_SetupD r2cSetup = nullptr;
    vDSP_DFT_Interleaved_SetupD c2rSetup = nullptr;
    vDSP_DFT_Interleaved_SetupD c2cSetup = nullptr;
    vDSP_DFT_Interleaved_SetupD ic2cSetup = nullptr;
};
} // namespace jsa
