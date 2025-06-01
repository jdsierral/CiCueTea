//
//  OverlapAddProcessor.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include "CQTProcessor.hpp"

#include "Benchtools.h"
#include "SignalUtils.h"

using namespace Eigen;
using namespace jsa;

CqtFullProcessor::CqtFullProcessor(double sampleRate, double numSamples,
                                   double fraction, double minFrequency,
                                   double maxFrequency, double refFrequency)
    : cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
          refFrequency) {
  Index nBands = cqt.getNumBands();
  Index blockSize = cqt.getBlockSize();
  win.resize(blockSize);
  win = hann(blockSize).sqrt();
  slicer.setSize(blockSize, blockSize / 2);
  splicer.setSize(blockSize, blockSize / 2);
  xi.resize(blockSize);
  Xcq.resize(blockSize, nBands);
  xi.setZero();
  Xcq.setZero();
  fs = cqt.getSampleRate();
  assert(blockSize == cqt.getNumSamps());
  assert(blockSize == win.size());
  assert(blockSize == slicer.getBlockSize());
  assert(blockSize == splicer.getBlockSize());
  assert(blockSize == xi.size());
  assert(blockSize == Xcq.rows());
}

double CqtFullProcessor::processSample(double sample) {
  RealTimeChecker ck;

  if (fs < 0)
    return 0;
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

SlidingCQTFullProcessor::SlidingCQTFullProcessor(
    double sampleRate, double numSamples, double fraction, double minFrequency,
    double maxFrequency, double refFrequency)
    : cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
          refFrequency) {
  Index nBands = cqt.getNumBands();
  Index blockSize = cqt.getBlockSize();
  win = hann(blockSize).sqrt();
  slicer.setSize(blockSize, blockSize / 2);
  splicer.setSize(blockSize, blockSize / 2);
  xi.resize(blockSize);
  ArrayXXcd coefs = ArrayXXcd::Zero(blockSize, nBands);
  ArrayXXcd validCoefs = ArrayXXcd::Zero(blockSize / 2, nBands);
  Xcq.fill(coefs);
  Zcq.fill(validCoefs);
  Ycq = coefs;
  fs = cqt.getSampleRate();

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

double SlidingCQTFullProcessor::processSample(double sample) {
  RealTimeChecker ck;

  if (fs < 0)
    return 0;
  slicer.pushSample(sample);
  sample = splicer.getSample();
  if (slicer.hasBlock()) {
    xi = slicer.getBlock();
    const Index sz = xi.size();
    assert(sz == xi.size());
    assert(sz == win.size());
    assert(sz == cqt.getNumSamps());
    xi *= win;

    Eigen::ArrayXXcd &Xi = Xcq.next();
    assert(sz == Xi.rows());
    cqt.forward(xi, Xi);
    Xi.colwise() *= win;
    Eigen::ArrayXXcd &Zi = Zcq.next();
    assert(sz == 2 * Zi.rows());
    Zi = Xi.topRows(sz / 2) + Xcq.last().bottomRows(sz / 2);

    processBlock(Zi);

    assert(sz == 2 * Zi.rows());
    Ycq.topRows(sz / 2) = Zcq.last();
    Ycq.bottomRows(sz / 2) = Zi;
    assert(sz == Ycq.rows());
    Ycq.colwise() *= win;
    assert(sz == xi.size());
    cqt.inverse(Ycq, xi);

    xi *= win;
    splicer.pushBlock(xi);
  }
  return sample;
}

//==========================================================================
//==========================================================================

CqtSparseProcessor::CqtSparseProcessor(double sampleRate, double numSamples,
                                       double fraction, double minFrequency,
                                       double maxFrequency, double refFrequency)
    : cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
          refFrequency) {
  Index nBands = cqt.getNumBands();
  Index blockSize = cqt.getBlockSize();

  win.resize(blockSize);
  win = hann(blockSize).sqrt();
  slicer.setSize(blockSize, blockSize / 2);
  splicer.setSize(blockSize, blockSize / 2);
  xi.resize(blockSize);
  xi.setZero();
  Xcq = cqt.getCoefs();
  fs = cqt.getSampleRate();
  assert(blockSize == cqt.getNumSamps());
  assert(blockSize == win.size());
  assert(blockSize == slicer.getBlockSize());
  assert(blockSize == splicer.getBlockSize());
  assert(blockSize == xi.size());
}

double CqtSparseProcessor::processSample(double sample) {
  RealTimeChecker ck;

  if (fs < 0)
    return 0;
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

SlidingCqtSparseProcessor::SlidingCqtSparseProcessor(
    double sampleRate, double numSamples, double fraction, double minFrequency,
    double maxFrequency, double refFrequency)
    : cqt(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
          refFrequency) {
  Index nBands = cqt.getNumBands();
  Index blockSize = cqt.getBlockSize();
  win = hann(blockSize).sqrt();
  slicer.setSize(blockSize, blockSize / 2);
  splicer.setSize(blockSize, blockSize / 2);
  xi.resize(blockSize);
  Win = cqt.getFrame();

  for (Index n = 0; n < nBands; n++) {
    Index sz = Win[n].size();
    Win[n] = hann(sz).sqrt();
  }

  auto coefs = cqt.getCoefs();
  auto validCoefs = cqt.getValidCoefs();
  Xcq.fill(coefs);
  Zcq.fill(validCoefs);
  Ycq = coefs;
  fs = cqt.getSampleRate();

  assert(blockSize == cqt.getBlockSize());
  assert(blockSize == win.size());
  assert(blockSize == slicer.getBlockSize());
  assert(blockSize == splicer.getBlockSize());
  assert(blockSize == xi.size());
}

double SlidingCqtSparseProcessor::processSample(double sample) {
  RealTimeChecker ck;

  if (fs < 0)
    return 0;
  slicer.pushSample(sample);
  sample = splicer.getSample();
  if (slicer.hasBlock()) {
    xi = slicer.getBlock();
    assert(xi.size() == cqt.getBlockSize());
    assert(xi.size() == cqt.getBlockSize());
    xi *= win;
    NsgfCqtSparse::Coefs &Xi = Xcq.next();
    cqt.forward(xi, Xi);
    Index nBands = cqt.getNumBands();

    for (Index n = 0; n < nBands; n++) {
      assert(Xi[n].size() == Win[n].size());
      Xi[n] *= Win[n];
    }

    NsgfCqtSparse::Coefs &Zi = Zcq.next();

    for (Index n = 0; n < nBands; n++) {
      Index ol = Xi[n].size() / 2;
      Zi[n] = Xi[n].head(ol) + Xcq.last()[n].tail(ol);
    }

    processBlock(Zi);

    for (Index n = 0; n < nBands; n++) {
      Index ol = Ycq[n].size() / 2;
      Ycq[n].head(ol) = Zcq.last()[n];
      Ycq[n].tail(ol) = Zi[n];
    }

    for (Index n = 0; n < nBands; n++) {
      assert(Ycq[n].size() == Win[n].size());
      Ycq[n] *= Win[n];
    }

    cqt.inverse(Ycq, xi);
    xi *= win;
    splicer.pushBlock(xi);
  }
  return sample;
}
