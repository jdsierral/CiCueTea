//
//  OverlapAddProcessor.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <armadillo>

#include "Splicer.hpp"
#include "Slicer.hpp"
#include "RingBuffer.hpp"
#include "CQT.hpp"
#include "Benchtools.h"
#include "VectorOps.h"

namespace jsa {

class cqtFullProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef);
    double processSample(double sample);
    virtual void processBlock(arma::cx_mat& block) = 0;
    
protected:
    std::shared_ptr<NsgfCqtFull> cqt;
    
private:
    arma::vec xi;
    arma::vec win;
    arma::cx_mat Xcq;
    Slicer slicer;
    Splicer splicer;
    double fs = -1;
};

//==========================================================================

class slidingCQTFullProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef);
    double processSample(double sample);
    virtual void processBlock(arma::cx_mat& block) = 0;
    
protected:
    std::shared_ptr<NsgfCqtFull> cqt;
    
private:
    arma::vec xi;
    DoubleBuffer<arma::cx_mat> Xcq;
    DoubleBuffer<arma::cx_mat> Zcq;
    arma::cx_mat Ycq;
    arma::vec win;
    Slicer slicer;
    Splicer splicer;
    double fs = -1;
};

//==========================================================================

class cqtSparseProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef);
    double processSample(double sample);
    virtual void processBlock(NsgfCqtSparse::Coefs& block) = 0;
    
protected:
    std::shared_ptr<NsgfCqtSparse> cqt;
    
private:
    arma::vec xi;
    arma::vec win;
    NsgfCqtSparse::Coefs Xcq;
    Slicer slicer;
    Splicer splicer;
    double fs = -1;
};

//==========================================================================

class slidingCqtSparseProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef);
    double processSample(double sample);
    virtual void processBlock(NsgfCqtSparse::Coefs& block) = 0;
    
protected:
    std::shared_ptr<NsgfCqtSparse> cqt;
    
private:
    arma::vec xi;
    DoubleBuffer<NsgfCqtSparse::Coefs> Xcq;
    DoubleBuffer<NsgfCqtSparse::Coefs> Zcq;
    NsgfCqtSparse::Coefs Ycq;
    arma::vec win;
    NsgfCqtSparse::Frame Win;
    Slicer slicer;
    Splicer splicer;
    double fs = -1;
};

}
