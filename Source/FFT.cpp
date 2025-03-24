//
//  FFT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "FFT.hpp"

#include <fftw3.h>

using namespace Eigen;

namespace jsa {
class DFTImpl {
public:
    DFTImpl(){}
    
    ~DFTImpl(){ cleanup(); }
    
    void init(size_t fftSize) {
        cleanup();
        this->fftSize = fftSize;
        unsigned int flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
        r2cPlan = fftw_plan_dft_r2c_1d(int(fftSize), nullptr, nullptr, flags);
        c2rPlan = fftw_plan_dft_c2r_1d(int(fftSize), nullptr, nullptr, flags);
        c2cPlan = fftw_plan_dft_1d(int(fftSize), nullptr, nullptr, FFTW_FORWARD, flags);
        ic2cPlan =fftw_plan_dft_1d(int(fftSize), nullptr, nullptr, FFTW_BACKWARD, flags);
    }
    
    void dft(const dcomplex* inPtr, dcomplex* outPtr) {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<dcomplex*>(inPtr));
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft(c2cPlan, inPtr_, outPtr_);
    }
    
    void idft(const dcomplex* inPtr, dcomplex* outPtr) {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<dcomplex*>(inPtr));
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft(ic2cPlan, inPtr_, outPtr_);
        Map<ArrayXcd>(outPtr, fftSize) *= (1.0 / fftSize);
    }
    
    void rdft(const double* inPtr, dcomplex* outPtr) {
        double* inPtr_ = const_cast<double*>(inPtr);
        fftw_complex* outPtr_ = reinterpret_cast<fftw_complex*>(outPtr);
        fftw_execute_dft_r2c(r2cPlan, inPtr_, outPtr_);
    }
    
    void irdft(const dcomplex* inPtr, double* outPtr) {
        fftw_complex* inPtr_ = reinterpret_cast<fftw_complex*>(const_cast<dcomplex*>(inPtr));
        fftw_execute_dft_c2r(c2rPlan, inPtr_, outPtr);
        Map<ArrayXd>(outPtr, fftSize) *= (1.0 / fftSize);
    }
    
private:
    void cleanup() {
        if (r2cPlan) fftw_destroy_plan(r2cPlan);
        if (c2rPlan) fftw_destroy_plan(c2rPlan);
        if (c2cPlan) fftw_destroy_plan(c2cPlan);
        if (ic2cPlan) fftw_destroy_plan(ic2cPlan);
    }
    
    size_t fftSize = 0;
    fftw_plan r2cPlan = nullptr;
    fftw_plan c2rPlan = nullptr;
    fftw_plan c2cPlan = nullptr;
    fftw_plan ic2cPlan = nullptr;
};
}

using namespace jsa;
using namespace Eigen;


DFT::DFT() :
pImpl(new DFTImpl())
{
}

void DFT::init(size_t fftSize) {
    pImpl->init(fftSize);
}

void DFT::dft(const ArrayXcd& X, ArrayXcd& Y)
{
    pImpl->dft(X.data(), Y.data());
}

void DFT::idft(const ArrayXcd& X, ArrayXcd& Y)
{
    pImpl->idft(X.data(), Y.data());
}

void DFT::rdft(const ArrayXd& x, ArrayXcd& X)
{
    pImpl->rdft(x.data(), X.data());
}

void DFT::irdft(const ArrayXcd& X, ArrayXd& x)
{
    pImpl->irdft(X.data(), x.data());
}

//==========================================================================

void DFT::dft(const ArrayXXcd& X, ArrayXXcd& Y)
{
    for (Index k = 0; k < X.cols(); k++) {
        pImpl->dft(X.col(k).data(), Y.col(k).data());
    }
}

void DFT::idft(const ArrayXXcd& X, ArrayXXcd& Y)
{
    for (Index k = 0; k < X.cols(); k++) {
        pImpl->idft(X.col(k).data(), Y.col(k).data());
    }
}

void DFT::rdft(const ArrayXXd& x, ArrayXXcd& X)
{
    for (Index k = 0; k < x.cols(); k++) {
        pImpl->rdft(x.col(k).data(), X.col(k).data());
    }
}

void DFT::irdft(const ArrayXXcd& X, ArrayXXd& x)
{
    for (Index k = 0; k < X.cols(); k++) {
        pImpl->irdft(X.col(k).data(), x.col(k).data());
    }
}

