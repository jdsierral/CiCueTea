//
//  OverlapAddProcessor.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include "OverlapAddProcessor.hpp"

#include "Benchtools.h"

using namespace Eigen;
using namespace jsa;

void NsgfCqtProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi = ArrayXd::Zero(blockSize);
    Xcq = ArrayXXcd::Zero(blockSize, cqt.nBands);
}

double NsgfCqtProcessor::processSample(double sample) {
    RealTimeChecker ck;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        xi *= win;
        cqt.forward(xi, Xcq);
        processBlock(Xcq);
        cqt.inverse(Xcq, xi);
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}

void SliCQOlaProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    Index nBands = cqt.nBands;
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    yi.resize(blockSize);
    ArrayXXcd coefs = ArrayXXcd::Zero(blockSize, nBands);
    ArrayXXcd validCoefs = ArrayXXcd::Zero(blockSize/2, nBands);
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
}

double SliCQOlaProcessor::processSample(double sample)
{
    RealTimeChecker ck;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        Index sz = xi.size();
        xi *= win;
        Eigen::ArrayXXcd& Xi = Xcq.next();
        cqt.forward(xi, Xi);
        Xi.colwise() *= win;
        Eigen::ArrayXXcd& Zi = Zcq.next();
        Zi = Xi.topRows(sz/2) + Xcq.last().bottomRows(sz/2);
        
        processBlock(Zi);
        
        Ycq.topRows(sz/2) = Zcq.last();
        Ycq.bottomRows(sz/2) = Zi;
        Ycq.colwise() *= win;
        cqt.inverse(Ycq, yi);
        yi *= win;
        splicer.pushBlock(yi);
    }
    return sample;
}
