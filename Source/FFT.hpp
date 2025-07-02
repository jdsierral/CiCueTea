//
//  FFT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include <armadillo>

namespace jsa {

class DFTImpl {
public:
    virtual void dft(const std::complex<double>* inPtr, std::complex<double>* outPtr) = 0;
    virtual void idft(const std::complex<double>* inPtr, std::complex<double>* outPtr) = 0;
    virtual void rdft(const double* inPtr, std::complex<double>* outPtr) = 0;
    virtual void irdft(const std::complex<double>* inPtr, double* outPtr) = 0;
};

class DFT {
public:
    DFT(){}
    void init(size_t fftSize);
    void dft(const arma::cx_vec& X, arma::cx_vec& Y);
    void idft(const arma::cx_vec& X, arma::cx_vec& Y);
    void rdft(const arma::vec& x, arma::cx_vec& X);
    void irdft(const arma::cx_vec& X, arma::vec& x);
    
    void dft(const arma::cx_vec& X, arma::subview_col<arma::cx_double>& Y);
    void idft(const arma::cx_vec& X, arma::subview_col<arma::cx_double>& Y);
    void rdft(const arma::vec& x, arma::subview_col<arma::cx_double>& X);
    void irdft(const arma::cx_vec& X, arma::subview_col<double>& x);
    
    
    void dft(const arma::cx_mat& X, arma::cx_mat& Y);
    void idft(const arma::cx_mat& X, arma::cx_mat& Y);
    void rdft(const arma::mat& x, arma::cx_mat& X);
    void irdft(const arma::cx_mat& X, arma::mat& x);
    
private:
    std::unique_ptr<DFTImpl> pImpl;
};

}
