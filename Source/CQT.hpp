//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include "FFT.hpp"

#include <armadillo>
#include "VectorOps.h"

namespace jsa {

class NsgfCqtCommon {
public:
    virtual void init(double sampleRate, arma::uword nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency);
    
    double fs = -1;
    arma::uword nSamps;
    arma::uword nFreqs;
    arma::uword nBands;
    double ppo;
    double fMin;
    double fMax;
    double fRef;
    
    arma::vec bax;
    arma::vec fax;
    arma::vec d;
    arma::cx_vec Xdft;
        
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
    void init(double sampleRate, arma::uword numSamples, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const  arma::vec &  x , arma::cx_mat& Xcq) override;
    void inverse(const arma::cx_mat& Xcq, arma::vec& x ) override;
    
    arma::mat g;
    arma::mat gDual;
    arma::cx_mat Xmat;
};

struct CqtCoefs {
public:
    CqtCoefs(const arma::ivec& sz) :
        sizes(sz),
        nBands(sz.size())
    {
        data.resize(nBands);
        offsets.resize(nBands);
        offsets = cumsum(sizes) - sizes(0);
        
    }
    
    arma::cx_vec operator()(arma::uword n) {
        return arma::cx_vec(data.memptr() + offsets(n), sizes(n));
    }
    
private:
    arma::cx_vec data;
    arma::ivec offsets;
    arma::ivec sizes;
    arma::uword nBands;
};


class NsgfCqtSparse : public NsgfCqtBase<std::vector<arma::cx_vec>> {
public:
    struct Idx {
        arma::uword i0 = 0;
        arma::uword len = 0;
    };
    using Coefs = std::vector<arma::cx_vec>;
    using Frame = std::vector<arma::vec>;
    using SpanList = std::vector<Idx>;
    
    void init(double sampleRate, arma::uword nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const arma::vec&  x, Coefs& Xcq) override;
    void inverse(const Coefs& Xcq, arma::vec& x ) override;
    
    Frame getFrame() const;
    Coefs getCoefs() const;
    Coefs getValidCoefs() const;
    
    Idx getIdx(const arma::vec& ii);
    
    SpanList idx;
    Frame g;
    Frame gDual;
    Coefs phase;
    arma::vec scale;

    Coefs Xcoefs;
    std::vector<DFT> dfts;
    inline static const double th = 1e-6;
};

}
