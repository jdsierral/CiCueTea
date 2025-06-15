//
//  FFT_MKL.h
//  CQTDSP
//
//  Created by Juan Sierra on 6/11/25.
//

#pragma once

#include <Eigen/Core>
#include <mkl.h>

using namespace Eigen;

namespace jsa {
class DFTImpl {
public:
  DFTImpl(size_t fftSize) : fftSize(fftSize) {
    DftiCreateDescriptor(&realSetup, DFTI_DOUBLE, DFTI_REAL, 1, this->fftSize);
    status += DftiSetValue(realSetup, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status += DftiSetValue(realSetup, DFTI_CONJUGATE_EVEN_STORAGE,
                           DFTI_COMPLEX_COMPLEX);
    status += DftiSetValue(realSetup, DFTI_FORWARD_SCALE, 1.0);
    status += DftiSetValue(realSetup, DFTI_BACKWARD_SCALE, 1.0 / fftSize);
    status += DftiCommitDescriptor(realSetup);

    DftiCreateDescriptor(&cplxSetup, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                         this->fftSize);
    status += DftiSetValue(realSetup, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status += DftiSetValue(realSetup, DFTI_CONJUGATE_EVEN_STORAGE,
                           DFTI_COMPLEX_COMPLEX);
    status += DftiSetValue(realSetup, DFTI_FORWARD_SCALE, 1.0);
    status += DftiSetValue(realSetup, DFTI_BACKWARD_SCALE, 1.0 / fftSize);
    status += DftiCommitDescriptor(cplxSetup);
    assert(status == DFTI_NO_ERROR);
  }

  ~DFTImpl() {
    if (realSetup)
      DftiFreeDescriptor(&realSetup);
    if (cplxSetup)
      DftiFreeDescriptor(&cplxSetup);
  }

  void dft(const dcomplex *inPtr, dcomplex *outPtr) {
    MKL_Complex16 *inPtr_ =
        reinterpret_cast<MKL_Complex16 *>(const_cast<dcomplex *>(inPtr));
    MKL_Complex16 *outPtr_ = reinterpret_cast<MKL_Complex16 *>(outPtr);
    DftiComputeForward(cplxSetup, inPtr_, outPtr_);
  }

  void idft(const dcomplex *inPtr, dcomplex *outPtr) {
    MKL_Complex16 *inPtr_ =
        reinterpret_cast<MKL_Complex16 *>(const_cast<dcomplex *>(inPtr));
    MKL_Complex16 *outPtr_ = reinterpret_cast<MKL_Complex16 *>(outPtr);
    DftiComputeBackward(cplxSetup, inPtr_, outPtr_);
  }

  void rdft(const double *inPtr, dcomplex *outPtr) {
    double *inPtr_ = const_cast<double *>(inPtr);
    MKL_Complex16 *outPtr_ = reinterpret_cast<MKL_Complex16 *>(outPtr);
    DftiComputeForward(realSetup, inPtr_, outPtr_);
  }

  void irdft(const dcomplex *inPtr, double *outPtr) {
    MKL_Complex16 *inPtr_ =
        reinterpret_cast<MKL_Complex16 *>(const_cast<dcomplex *>(inPtr));
    DftiComputeBackward(cplxSetup, inPtr_, outPtr);
  }

private:
  const size_t fftSize;
  DFTI_DESCRIPTOR_HANDLE realSetup;
  DFTI_DESCRIPTOR_HANDLE cplxSetup;
  long status = DFTI_NO_ERROR;
};
} // namespace jsa
