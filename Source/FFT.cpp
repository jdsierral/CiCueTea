//
//  FFT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "FFT.hpp"

#include "FFT_FFTW.h"
// #include "FFT_vDSP.h"
// #include "FFT_PFFFT.h"

using namespace jsa;
using namespace Eigen;

DFT::DFT(size_t fftSize) : pImpl(new DFTImpl(fftSize)) {}

void DFT::dft(const ArrayXcd &X, ArrayXcd &Y) {
  pImpl->dft(X.data(), Y.data());
}

void DFT::idft(const ArrayXcd &X, ArrayXcd &Y) {
  pImpl->idft(X.data(), Y.data());
}

void DFT::rdft(const ArrayXd &x, ArrayXcd &X) {
  pImpl->rdft(x.data(), X.data());
}

void DFT::irdft(const ArrayXcd &X, ArrayXd &x) {
  pImpl->irdft(X.data(), x.data());
}

//==========================================================================

void DFT::dft(const ArrayXXcd &X, ArrayXXcd &Y) {
  for (Index k = 0; k < X.cols(); k++) {
    pImpl->dft(X.col(k).data(), Y.col(k).data());
  }
}

void DFT::idft(const ArrayXXcd &X, ArrayXXcd &Y) {
  for (Index k = 0; k < X.cols(); k++) {
    pImpl->idft(X.col(k).data(), Y.col(k).data());
  }
}

void DFT::rdft(const ArrayXXd &x, ArrayXXcd &X) {
  for (Index k = 0; k < x.cols(); k++) {
    pImpl->rdft(x.col(k).data(), X.col(k).data());
  }
}

void DFT::irdft(const ArrayXXcd &X, ArrayXXd &x) {
  for (Index k = 0; k < X.cols(); k++) {
    pImpl->irdft(X.col(k).data(), x.col(k).data());
  }
}
