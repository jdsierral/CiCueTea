//
//  FFT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "FFT.hpp"

#include <fftw3.h>

namespace jsa {
class FFTWImpl : public DFTImpl
{
public:
    FFTWImpl(size_t fftSize)
    {
        this->fftSize = int(fftSize);
        unsigned int flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
        r2cPlan = fftw_plan_dft_r2c_1d(int(fftSize), nullptr, nullptr, flags);
        c2rPlan = fftw_plan_dft_c2r_1d(int(fftSize), nullptr, nullptr, flags);
        c2cPlan = fftw_plan_dft_1d(int(fftSize), nullptr, nullptr, FFTW_FORWARD, flags);
        ic2cPlan =fftw_plan_dft_1d(int(fftSize), nullptr, nullptr, FFTW_BACKWARD, flags);
    }
    ~FFTWImpl(){
        if (r2cPlan) fftw_destroy_plan(r2cPlan);
        if (c2rPlan) fftw_destroy_plan(c2rPlan);
        if (c2cPlan) fftw_destroy_plan(c2cPlan);
        if (ic2cPlan) fftw_destroy_plan(ic2cPlan);
    }

    void dft(const std::complex<double>* inPtr, std::complex<double>* outPtr) override {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(inPtr));
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft(c2cPlan, inPtr_, outPtr_);
    }
    
    void idft(const std::complex<double>* inPtr, std::complex<double>* outPtr) override {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(inPtr));
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft(ic2cPlan, inPtr_, outPtr_);
    }
    
    void rdft(const double* inPtr, std::complex<double>* outPtr) override {
        double* inPtr_ = const_cast<double*>(inPtr);
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft_r2c(r2cPlan, inPtr_, outPtr_);
    }
    
    void irdft(const std::complex<double>* inPtr, double* outPtr) override {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(inPtr));
        fftw_execute_dft_c2r(r2cPlan, inPtr_, outPtr);
    }

private:
    int fftSize;
    fftw_plan r2cPlan = nullptr;
    fftw_plan c2rPlan = nullptr;
    fftw_plan c2cPlan = nullptr;
    fftw_plan ic2cPlan = nullptr;
};
}

using namespace jsa;
using namespace arma;



void DFT::init(size_t fftSize) {
    pImpl.reset(new FFTWImpl(fftSize));
}

void DFT::dft(const cx_vec& X, cx_vec& Y) {pImpl->dft(X.memptr(), Y.memptr());}
void DFT::idft(const cx_vec& X, cx_vec& Y) {pImpl->idft(X.memptr(), Y.memptr());}
void DFT::rdft(const vec& x, cx_vec& X) {pImpl->rdft(x.memptr(), X.memptr());}
void DFT::irdft(const cx_vec& X, vec& x) {pImpl->irdft(X.memptr(), x.memptr());}

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
    for (uword k = 0; k < X.n_cols; k++) {
        pImpl->rdft(x.colptr(k), X.colptr(k));
    }
}

void DFT::irdft(const cx_mat& X, mat& x)
{
    for (uword k = 0; k < X.n_cols; k++) {
        pImpl->irdft(X.colptr(k), x.colptr(k));
    }
}

