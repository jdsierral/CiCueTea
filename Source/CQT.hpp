//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include "FFT.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include "VectorOps.h"

namespace jsa {

class NsgfCqtCommon {
public:
    virtual void init(double sampleRate, Eigen::Index nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency);
    
    double fs = -1;
    Eigen::Index nSamps;
    Eigen::Index nFreqs;
    Eigen::Index nBands;
    double ppo;
    double fMin;
    double fMax;
    double fRef;
    
    Eigen::ArrayXd bax;
    Eigen::ArrayXd fax;
    Eigen::ArrayXd d;
    Eigen::ArrayXcd Xdft;
        
    DFT dft;
};

template <typename T>
class NsgfCqtBase : public NsgfCqtCommon {
public:
    virtual void forward(const Eigen::ArrayXd& x, T& Xcq) = 0;
    virtual void inverse(const T& Xcq, Eigen::ArrayXd& x) = 0;
};

class NsgfCqtFull : public NsgfCqtBase<Eigen::ArrayXXcd> {
public:
    void init(double sampleRate, Eigen::Index numSamples, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const  Eigen::ArrayXd &  x , Eigen::ArrayXXcd& Xcq) override;
    void inverse(const Eigen::ArrayXXcd& Xcq, Eigen::ArrayXd& x ) override;
    
    Eigen::ArrayXXd g;
    Eigen::ArrayXXd gDual;
    Eigen::ArrayXXcd Xmat;
};

struct CqtCoefs {
public:
    CqtCoefs(const Eigen::ArrayXi& sz) :
        sizes(sz),
        nBands(sz.size())
    {
        data.resize(nBands);
        offsets.resize(nBands);
        offsets = cumsum(sizes) - sizes(0);
    }
    
    Eigen::Map<const Eigen::ArrayXcd> operator()(Eigen::Index n) {
        return Eigen::Map<const Eigen::ArrayXcd> ( data.data() + offsets(n), sizes(n));
    }
    
private:
    Eigen::ArrayXcd data;
    Eigen::ArrayXi offsets;
    Eigen::ArrayXi sizes;
    Eigen::Index nBands;
};


class NsgfCqtSparse : public NsgfCqtBase<std::vector<Eigen::ArrayXcd>> {
public:
    struct Idx {
        Eigen::Index i0 = 0;
        Eigen::Index len = 0;
    };
    using Coefs = std::vector<Eigen::ArrayXcd>;
    using Frame = std::vector<Eigen::ArrayXd>;
    using SpanList = std::vector<Idx>;
    
    void init(double sampleRate, Eigen::Index nSamps, double ppo, double minFrequency,
              double maxFrequency, double refFrequency) override;
    void forward(const Eigen::ArrayXd&  x, Coefs& Xcq) override;
    void inverse(const Coefs& Xcq, Eigen::ArrayXd& x ) override;
    Coefs getCoefs() const;
    
    Idx getIdx(const Eigen::ArrayXd& ii);
    
    SpanList idx;
    Frame g;
    Frame gDual;
    Coefs phase;
    Eigen::ArrayXd scale;

    Coefs Xcoefs;
    std::vector<DFT> dfts;
    inline static const double th = 1e-6;
};

}
