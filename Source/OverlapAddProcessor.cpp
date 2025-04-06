//
//  OverlapAddProcessor.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include "OverlapAddProcessor.hpp"

#include "Benchtools.h"

using namespace arma;
using namespace jsa;

void cqtFullProcessor::init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtFull>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    uword nBands = cqt->nBands;
    win.resize(blockSize);
    win = sqrt(hann(blockSize));
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    Xcq.resize(blockSize, nBands);
    xi.zeros();
    Xcq.zeros();
    this->fs = fs;
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == Xcq.n_rows);
}

double cqtFullProcessor::processSample(double sample) {
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        assert(xi.size() == xi.size());
        assert(xi.size() == win.size());
        assert(xi.size() == cqt->nSamps);
        assert(xi.size() == Xcq.n_rows);
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

void slidingCQTFullProcessor::init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtFull>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    uword nBands = cqt->nBands;
    win = sqrt(hann(blockSize));
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    cx_mat coefs = zeros<cx_mat>(blockSize, nBands);
    cx_mat validCoefs = zeros<cx_mat>(blockSize/2, nBands);
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    this->fs = fs;
    
    assert(blockSize == cqt->nSamps);
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
    assert(blockSize == Xcq.current().n_rows);
    assert(blockSize == Xcq.last().n_rows);
    assert(blockSize == Zcq.current().n_rows * 2);
    assert(blockSize == Zcq.last().n_rows * 2);
    assert(blockSize == Ycq.n_rows);
}

double slidingCQTFullProcessor::processSample(double sample)
{
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        const uword sz = xi.size();
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt->nSamps);
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            cx_mat& Xi = Xcq.next();
            assert(sz == Xi.n_rows);
            curCqt->forward(xi, Xi);
            Xi.each_col([&](cx_vec& col) { col = col % win; });
//            Xi = Xi.each_col() % win;
            cx_mat& Zi = Zcq.next();
            assert(sz == 2 * Zi.n_rows);
            Zi = Xi.head_rows(sz/2) + Xcq.last().tail_rows(sz/2);
            
            processBlock(Zi);
            
            assert(sz == 2 * Zi.n_rows);
            Ycq.head_rows(sz/2) = Zcq.last();
            Ycq.tail_rows(sz/2) = Zi;
            assert(sz == Ycq.n_rows);
            Ycq.each_col([&](cx_vec& col) { col = col % win; });
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

void cqtSparseProcessor::init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtSparse>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    win.resize(blockSize);
    win = sqrt(hann(blockSize));
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    xi.zeros();
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
        assert(xi.size() == xi.size());
        assert(xi.size() == win.size());
        assert(xi.size() == cqt->nSamps);
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


void slidingCqtSparseProcessor::init(double fs, double blockSize, double ppo, double fMin, double fMax, double fRef)
{
    auto newCqt = std::make_shared<NsgfCqtSparse>();
    newCqt->init(fs, blockSize, ppo, fMin, fMax, fRef);
    std::atomic_store(&cqt, newCqt);
    win = sqrt(hann(blockSize));
    slicer.setSize(blockSize, blockSize/2);
    splicer.setSize(blockSize, blockSize/2);
    xi.resize(blockSize);
    Win = cqt->getFrame();
    
    for (uword n = 0; n < cqt->nBands; n++) {
        uword sz = Win[n].size();
        Win[n] = sqrt(hann(sz));
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
//    assert(blockSize == Xcq.current().n_rows);
//    assert(blockSize == Xcq.last().n_rows);
//    assert(blockSize == Zcq.current().n_rows * 2);
//    assert(blockSize == Zcq.last().n_rows * 2);
//    assert(blockSize == Ycq.n_rows);
}

double slidingCqtSparseProcessor::processSample(double sample)
{
    RealTimeChecker ck;
    
    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        assert(xi.size() == xi.size());
        assert(xi.size() == win.size());
        assert(xi.size() == cqt->nSamps);
        xi *= win;
        auto curCqt = std::atomic_load(&cqt);
        if (curCqt) {
            NsgfCqtSparse::Coefs& Xi = Xcq.next();
            curCqt->forward(xi, Xi);
            uword nBands = curCqt->nBands;
            
            for (uword n = 0; n < nBands; n++) {
                assert(Xi[n].size() == Win[n].size());
                Xi[n] = Xi[n] % Win[n];
            }
            
            NsgfCqtSparse::Coefs& Zi = Zcq.next();
            
            for (uword n = 0; n < nBands; n++) {
                uword ol = Xi[n].size()/2;
                Zi[n] = Xi[n].head(ol) + Xcq.last()[n].tail(ol);
            }
            
            processBlock(Zi);
            
            for (uword n = 0; n < nBands; n++) {
                uword ol = Ycq[n].size()/2;
                Ycq[n].head(ol) = Zcq.last()[n];
                Ycq[n].tail(ol) = Zi[n];
            }
            
            for (uword n = 0; n < nBands; n++) {
                assert(Ycq[n].size() == Win[n].size());
                Ycq[n] = Ycq[n] % Win[n];
            }
            
            curCqt->inverse(Ycq, xi);
        }
        xi *= win;
        splicer.pushBlock(xi);
    }
    return sample;
}
