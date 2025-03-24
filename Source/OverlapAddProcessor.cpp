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
    Index nBands = cqt.nBands;
    win.resize(blockSize);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    Xcq.resize(blockSize, nBands);
    xi = ArrayXd::Zero(blockSize);
    Xcq = ArrayXXcd::Zero(blockSize, nBands);
    this->fs = fs;
    assert(blockSize == cqt.nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == Xcq.rows());
}

double NsgfCqtProcessor::processSample(double sample) {
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = slicer.getBlockSize();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt.nSamps);
        assert(sz == Xcq.rows());
        xi *= win;
        cqt.forward(xi, Xcq);
        processBlock(Xcq);
        cqt.inverse(Xcq, xi);
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}


//==========================================================================
//==========================================================================

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
    this->fs = fs;
    
    assert(blockSize == cqt.nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == yi.size());
    assert(blockSize == Xcq.current().rows());
    assert(blockSize == Xcq.last().rows());
    assert(blockSize == Zcq.current().rows() * 2);
    assert(blockSize == Zcq.last().rows() * 2);
    assert(blockSize == Ycq.rows());
}

double SliCQOlaProcessor::processSample(double sample)
{
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = xi.size();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt.nSamps);
        xi *= win;
        Eigen::ArrayXXcd& Xi = Xcq.next();
        assert(sz == Xi.rows());
        cqt.forward(xi, Xi);
        Xi.colwise() *= win;
        Eigen::ArrayXXcd& Zi = Zcq.next();
        assert(sz == 2 * Zi.rows());
        Zi = Xi.topRows(sz/2) + Xcq.last().bottomRows(sz/2);
        
        processBlock(Zi);
        
        assert(sz == 2 * Zi.rows());
        Ycq.topRows(sz/2) = Zcq.last();
        Ycq.bottomRows(sz/2) = Zi;
        assert(sz == Ycq.rows());
        Ycq.colwise() *= win;
        assert(sz == yi.size());
        cqt.inverse(Ycq, yi);
        yi *= win;
        splicer.pushBlock(yi);
    }
    return sample;
}

//==========================================================================
//==========================================================================

void NsgfCqtSparseProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    win.resize(blockSize);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    xi = ArrayXd::Zero(blockSize);
    Xcq = cqt.getCoefs();
    this->fs = fs;
    assert(blockSize == cqt.nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
}

double NsgfCqtSparseProcessor::processSample(double sample) {
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = slicer.getBlockSize();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt.nSamps);
        xi *= win;
        cqt.forward(xi, Xcq);
        processBlock(Xcq);
        cqt.inverse(Xcq, xi);
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}

//==========================================================================
//==========================================================================


void SliCQSparseProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    yi.resize(blockSize);
    Win = cqt.getFrame();
    auto coefs = cqt.getCoefs();
    auto validCoefs = cqt.getValidCoefs();
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    this->fs = fs;
    
    assert(blockSize == cqt.nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == yi.size());
//    assert(blockSize == Xcq.current().rows());
//    assert(blockSize == Xcq.last().rows());
//    assert(blockSize == Zcq.current().rows() * 2);
//    assert(blockSize == Zcq.last().rows() * 2);
//    assert(blockSize == Ycq.rows());
}

double SliCQSparseProcessor::processSample(double sample)
{
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = xi.size();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt.nSamps);
        xi *= win;
        NsgfCqtSparse::Coefs& Xi = Xcq.next();
        cqt.forward(xi, Xi);
        
        for (Index n = 0; n < cqt.nBands; n++) {
            assert(Xi[n].size() == Win[n].size());
            Xi[n] *= Win[n];
        }
        
        NsgfCqtSparse::Coefs& Zi = Zcq.next();
        
        for (Index n = 0; n < cqt.nBands; n++) {
            Index ol = Xi[n].size()/2;
            Zi[n] = Xi[n].head(ol) + Xcq.last()[n].tail(ol);
        }
        
        processBlock(Zi);
        
        for (Index n = 0; n < cqt.nBands; n++) {
            Index ol = Ycq[n].size()/2;
            Ycq[n].head(ol) = Zcq.last()[n];
            Ycq[n].tail(ol) = Zi[n];
        }
        
        for (Index n = 0; n < cqt.nBands; n++) {
            assert(Ycq[n].size() == Win[n].size());
            Ycq[n] *= Win[n];
        }
        
        assert(sz == yi.size());
        cqt.inverse(Ycq, yi);
        yi *= win;
        splicer.pushBlock(yi);
    }
    return sample;
}
