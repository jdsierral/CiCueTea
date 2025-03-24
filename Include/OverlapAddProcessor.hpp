//
//  OverlapAddProcessor.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <Eigen/Core>

#include "Splicer.hpp"
#include "Slicer.hpp"
#include "RingBuffer.hpp"
#include "CQT.hpp"
#include "Benchtools.h"
#include "VectorOps.h"

namespace jsa {

class NsgfCqtProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef);
    double processSample(double sample);
    virtual void processBlock(Eigen::ArrayXXcd& block) = 0;
    
protected:
    NsgfCqtFull cqt;
    
private:
    Eigen::ArrayXd xi;
    Eigen::ArrayXXcd Xcq;
    Eigen::ArrayXd win;
    Slicer slicer;
    Splicer splicer;
};

class SliCQOlaProcessor {
public:
    void init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef);
    double processSample(double sample);
    virtual void processBlock(Eigen::ArrayXXcd& block) = 0;
    
protected:
    NsgfCqtFull cqt;
    
private:
    Eigen::ArrayXd xi;
    Eigen::ArrayXd yi;
    DoubleBuffer<Eigen::ArrayXXcd> Xcq;
    DoubleBuffer<Eigen::ArrayXXcd> Zcq;
    Eigen::ArrayXXcd Ycq;
    Eigen::ArrayXd win;
    Slicer slicer;
    Splicer splicer;
};

}
