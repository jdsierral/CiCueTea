//
//  OverlapAddProcessor.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include "CQTProcessor.hpp"

#include "RTChecker.h"
#include "SignalUtils.h"

using namespace Eigen;
using namespace jsa;

CqtDenseProcessor::CqtDenseProcessor(double sampleRate, Index numSamples,
                                   double fraction, double minFrequency,
                                   double maxFrequency, double refFrequency) :
    cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency, refFrequency),
    xi(cqt.getBlockSize()),
    win(cqt.getBlockSize()),
    Xcq(cqt.getBlockSize(), cqt.getNumBands()),
    slicer(cqt.getBlockSize(), cqt.getBlockSize() / 2),
    splicer(cqt.getBlockSize(), cqt.getBlockSize() / 2)
{
    win = hann(cqt.getBlockSize()).sqrt();
    xi.setZero();
    Xcq.setZero();
    fs = cqt.getSampleRate();
    assert(cqt.getBlockSize() == win.size());
    assert(cqt.getBlockSize() == slicer.getBlockSize());
    assert(cqt.getBlockSize() == splicer.getBlockSize());
    assert(cqt.getBlockSize() == xi.size());
    assert(cqt.getBlockSize() == Xcq.rows());
}

double CqtDenseProcessor::processSample(double sample)
{
    RealTimeChecker ck;

    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        assert(xi.size() == win.size());
        assert(xi.size() == cqt.getBlockSize());
        assert(xi.size() == Xcq.rows());
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

CqtSparseProcessor::CqtSparseProcessor(double sampleRate, Index numSamples,
                                       double fraction, double minFrequency,
                                       double maxFrequency, double refFrequency) :
    cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency, refFrequency),
    xi(cqt.getBlockSize()),
    win(cqt.getBlockSize()),
    Xcq(cqt.getCoefs()),
    slicer(cqt.getBlockSize(), cqt.getBlockSize() / 2),
    splicer(cqt.getBlockSize(), cqt.getBlockSize() / 2)
{
    win = hann(cqt.getBlockSize()).sqrt();
    xi.setZero();
    fs = cqt.getSampleRate();
    assert(cqt.getBlockSize() == win.size());
    assert(cqt.getBlockSize() == slicer.getBlockSize());
    assert(cqt.getBlockSize() == splicer.getBlockSize());
    assert(cqt.getBlockSize() == xi.size());
}

double CqtSparseProcessor::processSample(double sample)
{
    RealTimeChecker ck;

    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        xi = slicer.getBlock();
        assert(xi.size() == xi.size());
        assert(xi.size() == win.size());
        assert(xi.size() == cqt.getNumSamps());
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

SlidingCQTDenseProcessor::SlidingCQTDenseProcessor(double sampleRate, Index numSamples,
                                                 double fraction, double minFrequency,
                                                 double maxFrequency, double refFrequency) :
    cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency, refFrequency),
    xi(cqt.getBlockSize()),
    win(cqt.getBlockSize()),
    slicer(cqt.getBlockSize(), cqt.getBlockSize() / 2),
    splicer(cqt.getBlockSize(), cqt.getBlockSize() / 2)

{
    Index nBands         = cqt.getNumBands();
    Index blockSize      = cqt.getBlockSize();
    win                  = hann(blockSize).sqrt();
    ArrayXXcd coefs      = ArrayXXcd::Zero(blockSize, nBands);
    ArrayXXcd validCoefs = ArrayXXcd::Zero(blockSize / 2, nBands);
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    fs  = cqt.getSampleRate();

    assert(blockSize == cqt.getNumSamps());
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

double SlidingCQTDenseProcessor::processSample(double sample)
{
    RealTimeChecker ck;

    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        Eigen::ArrayXXcd& Xi   = Xcq.current();
        Eigen::ArrayXXcd& Xim1 = Xcq.last();
        Eigen::ArrayXXcd& Zi   = Zcq.current();
        Eigen::ArrayXXcd& Zim1 = Zcq.last();
        Eigen::ArrayXXcd& Yi   = Ycq;

        Index sz = xi.size();
        Index ol = sz / 2;

        assert(sz == Xi.rows());
        assert(sz == 2 * Zi.rows());
        assert(sz == Ycq.rows());
        assert(sz == xi.size());
        assert(sz == win.size());
        assert(sz == cqt.getNumSamps());

        xi = slicer.getBlock();
        xi *= win; // xi *= win; xi /= 2;

        cqt.forward(xi, Xi);
        Xi.colwise() *= win;
        Zi = Xi.topRows(ol) + Xim1.bottomRows(ol);

        processBlock(Zi);

        Yi.topRows(ol)    = Zim1;
        Yi.bottomRows(ol) = Zi;

        Yi.colwise() *= win;

        cqt.inverse(Yi, xi);
        xi *= win;

        splicer.pushBlock(xi);
        Xcq.advance();
        Zcq.advance();
    }
    return sample;
}

//==========================================================================
//==========================================================================

SlidingCqtSparseProcessor::SlidingCqtSparseProcessor(double sampleRate, Index numSamples,
                                                     double fraction, double minFrequency,
                                                     double maxFrequency, double refFrequency) :
    cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency, refFrequency),
    xi(cqt.getBlockSize()),
    win(cqt.getBlockSize()),
    slicer(cqt.getBlockSize(), cqt.getBlockSize() / 2),
    splicer(cqt.getBlockSize(), cqt.getBlockSize() / 2)
{
    Index nBands    = cqt.getNumBands();
    Index blockSize = cqt.getBlockSize();
    xi.setZero();
    win = hann(blockSize).sqrt();
    Win = cqt.getFrame();

    for (Index n = 0; n < nBands; n++) {
        Index sz = Win[n].size();
        Win[n]   = hann(sz).sqrt();
    }

    auto coefs      = cqt.getCoefs();
    auto validCoefs = cqt.getValidCoefs();
    Xcq.fill(coefs);
    Zcq.fill(validCoefs);
    Ycq = coefs;
    fs  = cqt.getSampleRate();

    assert(blockSize == cqt.getBlockSize());
    assert(blockSize == win.size());
    assert(blockSize == slicer.getBlockSize());
    assert(blockSize == splicer.getBlockSize());
    assert(blockSize == xi.size());
}

double SlidingCqtSparseProcessor::processSample(double sample)
{
    RealTimeChecker ck;

    if (fs < 0) return 0;
    slicer.pushSample(sample);
    sample = splicer.getSample();
    if (slicer.hasBlock()) {
        NsgfCqtSparse::Coefs& Xi     = Xcq.current();
        NsgfCqtSparse::Coefs& Xim1   = Xcq.last();
        NsgfCqtSparse::Coefs& Zi     = Zcq.current();
        NsgfCqtSparse::Coefs& Zim1   = Zcq.last();
        NsgfCqtSparse::Coefs& Yi     = Ycq;
        Index                 nBands = cqt.getNumBands();
        assert(xi.size() == cqt.getBlockSize());
        xi = slicer.getBlock();

        xi *= win;

        cqt.forward(xi, Xi);

        for (Index k = 0; k < nBands; k++) {
            Xi[k] *= Win[k];
        }

        for (Index k = 0; k < nBands; k++) {
            Index ol = cqt.getLength(k) / 2;
            Zi[k]    = Xim1[k].tail(ol) + Xi[k].head(ol);
        }

        processBlock(Zi);

        for (Index k = 0; k < nBands; k++) {
            Index ol       = cqt.getLength(k) / 2;
            Yi[k].head(ol) = Zim1[k];
            Yi[k].tail(ol) = Zi[k];
        }
        for (Index k = 0; k < nBands; k++) {
            Yi[k] *= Win[k];
        }

        cqt.inverse(Yi, xi);
        xi *= win;
        splicer.pushBlock(xi);
        Xcq.advance();
        Zcq.advance();
    }
    return sample;
}
