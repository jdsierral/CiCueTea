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

void cqtFullProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtFull>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    Index nBands = cqt->nBands;
    win.resize(blockSize);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    Xcq.resize(blockSize, nBands);
    xi.setZero();
    Xcq.setZero();
    this->fs = fs;
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == Xcq.rows());
}

double cqtFullProcessor::processSample(double sample) {
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = slicer.getBlockSize();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt->nSamps);
        assert(sz == Xcq.rows());
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            curCqt->forward(xi, Xcq);
            processBlock(Xcq);
            curCqt->inverse(Xcq, xi);
        }
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}


//==========================================================================
//==========================================================================

void slidingCQTFullProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtFull>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    Index nBands = cqt->nBands;
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    ArrayXXcd coefs = ArrayXXcd::Zero(blockSize, nBands);
    ArrayXXcd validCoefs = ArrayXXcd::Zero(blockSize/2, nBands);
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    this->fs = fs;
    
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == Xcq.current().rows());
    assert(blockSize == Xcq.last().rows());
    assert(blockSize == Zcq.current().rows() * 2);
    assert(blockSize == Zcq.last().rows() * 2);
    assert(blockSize == Ycq.rows());
}

double slidingCQTFullProcessor::processSample(double sample)
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
        assert(sz == cqt->nSamps);
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            Eigen::ArrayXXcd& Xi = Xcq.next();
            assert(sz == Xi.rows());
            curCqt->forward(xi, Xi);
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
            assert(sz == xi.size());
            curCqt->inverse(Ycq, xi);
        }
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}

//==========================================================================
//==========================================================================

void cqtSparseProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtSparse>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    win.resize(blockSize);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    xi.setZero();
    Xcq = cqt->getCoefs();
    this->fs = fs;
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
}

double cqtSparseProcessor::processSample(double sample) {
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const Index sz = slicer.getBlockSize();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt->nSamps);
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            curCqt->forward(xi, Xcq);
            processBlock(Xcq);
            curCqt->inverse(Xcq, xi);
        }
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}

//==========================================================================
//==========================================================================


void slidingCqtSparseProcessor::init(double fs, double blockSize, double ppo, double fMax, double fMin, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtSparse>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    win = hann(blockSize).sqrt();
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    Win = cqt->getFrame();
    
    for (Index n = 0; n < cqt->nBands; n++) {
        Index sz = Win[n].size();
        Win[n] = hann(sz).sqrt();
    }
    
    auto coefs = cqt->getCoefs();
    auto validCoefs = cqt->getValidCoefs();
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    this->fs = fs;
    
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
//    assert(blockSize == Xcq.current().rows());
//    assert(blockSize == Xcq.last().rows());
//    assert(blockSize == Zcq.current().rows() * 2);
//    assert(blockSize == Zcq.last().rows() * 2);
//    assert(blockSize == Ycq.rows());
}

double slidingCqtSparseProcessor::processSample(double sample)
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
        assert(sz == cqt->nSamps);
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            NsgfCqtSparse::Coefs& Xi = Xcq.next();
            curCqt->forward(xi, Xi);
            Index nBands = curCqt->nBands;
            
            for (Index n = 0; n < nBands; n++) {
                assert(Xi[n].size() == Win[n].size());
                Xi[n] *= Win[n];
            }
            
            NsgfCqtSparse::Coefs& Zi = Zcq.next();
            
            for (Index n = 0; n < nBands; n++) {
                Index ol = Xi[n].size()/2;
                Zi[n] = Xi[n].head(ol) + Xcq.last()[n].tail(ol);
            }
            
            processBlock(Zi);
            
            for (Index n = 0; n < nBands; n++) {
                Index ol = Ycq[n].size()/2;
                Ycq[n].head(ol) = Zcq.last()[n];
                Ycq[n].tail(ol) = Zi[n];
            }
            
            for (Index n = 0; n < nBands; n++) {
                assert(Ycq[n].size() == Win[n].size());
                Ycq[n] *= Win[n];
            }
            
            assert(sz == xi.size());
            curCqt->inverse(Ycq, xi);
        }
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}
