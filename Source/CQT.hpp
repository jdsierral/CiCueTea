//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include "FFT.hpp"

#include <armadillo>


namespace jsa {

using CqtCoefs = arma::field<arma::cx_vec>;

class NsgfCqtCommon {
public:
    virtual void init(double sampleRate, size_t nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency);
    
    double fs = -1;
    size_t nSamps;
    size_t nFreqs;
    size_t nBands;
    double ppo;
    double fMin;
    double fMax;
    double fRef;
    
    arma::vec bax;
    arma::vec fax;
    arma::vec d;
    
    arma::cx_vec X;
    arma::cx_vec Y;
        
    DFT dft;
};

template <typename T>
class NsgfCqtBase : public NsgfCqtCommon {
public:
    virtual void forward(const arma::vec& x, T& Xcq) = 0;
    virtual void inverse(const T& Xcq, arma::vec& x) = 0;
};

class NsgfCqtFull : public NsgfCqtBase<arma::cx_mat> {
public:
    void init(double sampleRate, size_t nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const  arma::vec &  x , arma::cx_mat& Xcq) override;
    void inverse(const arma::cx_mat& Xcq, arma::vec& x ) override;
    
    arma::mat g;
    arma::mat gDual;
    
    arma::cx_mat Yi;
    arma::cx_mat Xi;
    
};

class NsgfCqtSparse : public NsgfCqtBase<CqtCoefs> {
public:
    using CqtFrame = arma::field<arma::vec>;
    using SpanList = arma::field<arma::span>;
    
    void init(double sampleRate, size_t nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const arma::vec&  x, CqtCoefs& Xcq) override;
    void inverse(const CqtCoefs& Xcq, arma::vec& x ) override;
    CqtCoefs getCoefs() const;
    
    arma::uvec padIdxs(arma::uvec ii);
    
    SpanList idx;
    CqtFrame g;
    CqtFrame gDual;
    CqtCoefs phase;
    arma::vec scale;

    CqtCoefs Yi;
    CqtCoefs Xi;
    CqtCoefs Zi;
    std::vector<DFT> dfts;
    inline static const double th = 1e-6;
};

}
