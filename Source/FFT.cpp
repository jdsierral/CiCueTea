//
//  FFT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "FFT.hpp"

#include "FFT_FFTW.h"



using namespace jsa;
using namespace arma;


DFT::DFT() :
pImpl(new DFTImpl())
{
}

void DFT::init(size_t fftSize) {
    pImpl->init(fftSize);
}

void DFT::dft(const cx_vec& X, cx_vec& Y)
{
    pImpl->dft(X.memptr(), Y.memptr());
}

void DFT::idft(const cx_vec& X, cx_vec& Y)
{
    pImpl->idft(X.memptr(), Y.memptr());
}

void DFT::rdft(const vec& x, cx_vec& X)
{
    pImpl->rdft(x.memptr(), X.memptr());
}

void DFT::irdft(const cx_vec& X, vec& x)
{
    pImpl->irdft(X.memptr(), x.memptr());
}

//==========================================================================

void DFT::dft(const cx_mat& X, cx_mat& Y)
{
    for (uword k = 0; k < X.n_cols; k++) {
        pImpl->dft(X.colptr(k), Y.colptr(k));
    }
}

void DFT::idft(const cx_mat& X, cx_mat& Y)
{
    for (uword k = 0; k < X.n_cols; k++) {
        pImpl->idft(X.colptr(k), Y.colptr(k));
    }
}

void DFT::rdft(const mat& x, cx_mat& X)
{
    for (uword k = 0; k < x.n_cols; k++) {
        pImpl->rdft(x.colptr(k), X.colptr(k));
    }
}

void DFT::irdft(const cx_mat& X, mat& x)
{
    for (uword k = 0; k < X.n_cols; k++) {
        pImpl->irdft(X.colptr(k), x.colptr(k));
    }
}

