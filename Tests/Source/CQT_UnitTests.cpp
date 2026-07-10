//
//  CQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/9/25.
//
//  Core correctness tests for the NSGF-CQT engine, organized in two layers:
//    1. Frame-theory identities: the painless-frame condition Σₖ gₖ·g̃ₖ = 1,
//       which is the mathematical guarantee of invertibility.
//    2. Transform round trips: forward → inverse reconstruction at
//       numerical precision, for the dense, sparse, and block-streamed cases.
//

#include <boost/test/unit_test.hpp>
#include <cmath>

#include <Eigen/Core>

#include <CQT.hpp>
#include <MathUtils.h>
#include <SignalUtils.h>

#include "EmptyCQTProc.h"
#include "TestSignals.h"

using namespace Eigen;
using namespace std;
using namespace jsa;
using namespace jsa::test;

// Layer 1 — painless-frame identity (dense): the frame and its canonical dual
// must satisfy Σₖ gₖ(f)·g̃ₖ(f) = 1 at every bin up to Nyquist. This holds by
// construction (g̃ = g/d with d = Σ g²), so the tolerance is numerical only.
BOOST_AUTO_TEST_CASE(CQTTestDense1)
{
    double fs     = 48000;
    int    nSamps = 1 << 10;
    double frac   = 1;
    double fMin   = 100;
    double fMax   = 10000;
    double fRef   = 1500;

    NsgfCqtDense cqt(fs, nSamps, frac, fMin, fMax, fRef);

    ArrayXd ggDual = (cqt.getFrame() * cqt.getDualFrame())
                         .rowwise()
                         .sum()
                         .head(cqt.getNumFreqs() / 2 + 1);

    BOOST_CHECK(rms(ggDual - 1) < 1e-10);
}

// Layer 2 — dense round trip: a sinusoid at the reference frequency must
// survive forward → inverse at numerical precision (measured ≈ 3e-16).
BOOST_AUTO_TEST_CASE(CQTTestDense2)
{
    double fs     = 48000;
    size_t nSamps = 1 << 10;
    double frac   = 1;
    double fMin   = 100;
    double fMax   = 10000;
    double fRef   = 1500;

    NsgfCqtDense cqt(fs, nSamps, frac, fMin, fMax, fRef);
    ArrayXd      t = regspace(int(nSamps)) / fs;
    ArrayXd      x = (2 * M_PI * fRef * t).sin();
    ArrayXd      y(nSamps);
    ArrayXXcd    Xcq(cqt.getNumSamps(), cqt.getNumBands());

    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);

    ArrayXd dif = x - y;
    BOOST_CHECK(rms(dif) < 1e-10);
}

// Layer 1 — painless-frame identity (sparse): as in the dense case, but each
// atom lives on its truncated band span, so the identity is assembled
// span-by-span. Exact by construction (d is computed after thresholding).
BOOST_AUTO_TEST_CASE(CQTTestSparse1)
{
    double fs     = 48000;
    int    nSamps = 1 << 10;
    double frac   = 1;
    double fMin   = 100;
    double fMax   = 10000;
    double fRef   = 1500;

    NsgfCqtSparse cqt(fs, nSamps, frac, fMin, fMax, fRef);

    ArrayXd buf = ArrayXd::Zero(cqt.getNumFreqs());
    for (int k = 0; k < cqt.getNumBands(); k++) {
        auto  s   = cqt.getBandSpan(k);
        Index i0  = s.i0;
        Index len = s.len;
        buf.segment(i0, len) += (cqt.getAtom(k) * cqt.getDualAtom(k));
    }
    ArrayXd ggDual = buf.head(cqt.getNumFreqs() / 2 + 1);
    BOOST_CHECK(rms(ggDual - 1) < 1e-10);
}

// Layer 2 — sparse round trip, at both a coarse (1 band/octave) and a fine
// (12 bands/octave) resolution, since band truncation behaves differently
// as the Gaussians narrow.
BOOST_AUTO_TEST_CASE(CQTTestSparse2)
{
    double fs     = 48000;
    int    nSamps = 1 << 16;
    double fMin   = 100;
    double fMax   = 10000;
    double fRef   = 1500;

    for (double frac : {1.0, 1.0 / 12.0}) {
        NsgfCqtSparse cqt(fs, nSamps, frac, fMin, fMax, fRef);

        ArrayXd t = regspace(int(nSamps)) / fs;
        ArrayXd x = (2 * M_PI * fRef * t + M_PI_4 / 2).sin();
        ArrayXd y(nSamps);
        auto    Xcq = cqt.getCoefs();

        cqt.forward(x, Xcq);
        cqt.inverse(Xcq, y);

        ArrayXd dif = x - y;
        BOOST_CHECK_MESSAGE(rms(dif) < 1e-10,
                            "frac = " << frac << ", rms = " << rms(dif));
    }
}

// Construction contract: an invalid configuration must never crash or throw —
// it constructs an inert object that reports !isValid() and outputs silence.
// This supports host lifecycles (DAWs) that construct with a placeholder
// sample rate before the real one is known.
BOOST_AUTO_TEST_CASE(CQTTestValidity)
{
    double fs = 48000;
    Index  N  = 1 << 10;

    BOOST_CHECK(NsgfCqtDense(fs, N, 1, 100, 10000, 1500).isValid());
    BOOST_CHECK(!NsgfCqtDense(0, N, 1, 100, 10000, 1500).isValid());     // placeholder fs
    BOOST_CHECK(!NsgfCqtDense(-48e3, N, 1, 100, 10000, 1500).isValid()); // negative fs
    BOOST_CHECK(!NsgfCqtDense(fs, N - 1, 1, 100, 10000, 1500).isValid()); // non-pow2 block
    BOOST_CHECK(!NsgfCqtDense(fs, N, 1, 0, 10000, 1500).isValid());      // fMin = 0
    BOOST_CHECK(!NsgfCqtDense(fs, N, 1, 100, 30000, 1500).isValid());    // fMax over Nyquist
    BOOST_CHECK(!NsgfCqtDense(fs, N, 1, 10000, 6000, 8000).isValid());  // inverted range

    // Sub-octave ranges are legitimate — feasibility is measured on the
    // constructed frame, not assumed from the parameter extent.
    BOOST_CHECK(NsgfCqtDense(fs, N, 1, 6000, 10000, 8000).isValid());

    // Fuzzy boundary — structurally fine, but Q is too high for the block
    // (12 bands/octave from 100 Hz in 256 samples): no parameter rule fires;
    // the *measured* frame-health check does.
    BOOST_CHECK(!NsgfCqtDense(fs, 1 << 8, 1.0 / 12, 100, 10000, 1500).isValid());
    BOOST_CHECK(!NsgfCqtSparse(fs, 1 << 8, 1.0 / 12, 100, 10000, 1500).isValid());
    BOOST_CHECK(NsgfCqtSparse(fs, 1 << 16, 1.0 / 12, 100, 10000, 1500).isValid());

    // fRef only anchors the band grid — it may lie outside [fMin, fMax].
    // ceil() rounds the band counts outward (they go negative on one side),
    // so bax(0) <= fMin and bax(end) >= fMax hold regardless; the measured
    // health checks judge the result like any other configuration.
    {
        NsgfCqtDense anchored(fs, N, 1, 2000, 20000, 440);
        BOOST_CHECK(anchored.isValid());
        ArrayXd   xa = ArrayXd::Random(N);
        ArrayXd   ya(N);
        ArrayXXcd Xa(anchored.getNumSamps(), anchored.getNumBands());
        anchored.forward(xa, Xa);
        anchored.inverse(Xa, ya);
        BOOST_CHECK_MESSAGE(rms(xa - ya) < 1e-10, "rms = " << rms(xa - ya));

        NsgfCqtSparse anchoredSp(fs, N, 1, 2000, 20000, 440);
        BOOST_CHECK(anchoredSp.isValid());
        auto Xs = anchoredSp.getCoefs();
        anchoredSp.forward(xa, Xs);
        anchoredSp.inverse(Xs, ya);
        BOOST_CHECK_MESSAGE(rms(xa - ya) < 1e-10, "rms = " << rms(xa - ya));
    }

    // Inert dense transform: outputs are zeroed, nothing crashes.
    NsgfCqtDense bad(0, N, 1, 100, 10000, 1500);
    ArrayXd      x   = ArrayXd::Random(N);
    ArrayXd      y   = ArrayXd::Random(N);
    ArrayXXcd    Xcq = ArrayXXcd::Random(N, 8);
    bad.forward(x, Xcq);
    BOOST_CHECK(Xcq.abs().maxCoeff() == 0);
    bad.inverse(Xcq, y);
    BOOST_CHECK(y.abs().maxCoeff() == 0);

    // Inert sparse transform.
    NsgfCqtSparse badSparse(0, N, 1, 100, 10000, 1500);
    BOOST_CHECK(!badSparse.isValid());
    NsgfCqtSparse::Coefs coefs = badSparse.getCoefs(); // empty, no assert
    badSparse.forward(x, coefs);
    badSparse.inverse(coefs, y);
    BOOST_CHECK(y.abs().maxCoeff() == 0);

    // Inert processors: silence out, no crash, for all four variants.
    CqtDense     p1(0, N, 1, 100, 10000, 1500);
    CqtSparse    p2(0, N, 1, 100, 10000, 1500);
    SliCqtDense  p3(0, N, 1, 100, 10000, 1500);
    SliCqtSparse p4(0, N, 1, 100, 10000, 1500);
    BOOST_CHECK(!p1.isValid());
    BOOST_CHECK(!p2.isValid());
    BOOST_CHECK(!p3.isValid());
    BOOST_CHECK(!p4.isValid());
    double acc = 0;
    for (Index n = 0; n < 4 * N; n++) {
        acc += std::abs(p1.processSample(1.0)) + std::abs(p2.processSample(1.0)) +
               std::abs(p3.processSample(1.0)) + std::abs(p4.processSample(1.0));
    }
    BOOST_CHECK(acc == 0);
}

// Layer 2 — block-streamed round trip: a chirp is processed in 50%-overlapped
// √Hann WOLA blocks through forward → inverse; the overlap-add of the block
// round trips must reassemble the signal. This is the transform-level version
// of what the processor classes automate (see SlidingCQT_UnitTests.cpp).
BOOST_AUTO_TEST_CASE(SlidingCQT)
{
    double fs     = 48000;
    Index  nSamps = (1 << 16);
    double frac   = 1;
    double fMin   = 100;
    double fMax   = 10000;
    double fRef   = 1500;

    Index   blockSize = 1 << 12;
    Index   hopSize   = blockSize / 2;
    Index   nBlocks   = (nSamps - blockSize) / hopSize;
    ArrayXd win       = hann(blockSize).sqrt();

    NsgfCqtDense cqt(fs, blockSize, frac, fMin, fMax, fRef);

    ArrayXd t      = regspace(nSamps) / fs;
    double  fScale = 1.2;
    ArrayXd x      = logChirp(t, fMin * fScale, fMax / fScale);
    x.head(blockSize).fill(0);
    x.tail(blockSize).fill(0);
    x *= hann(nSamps);
    ArrayXd y = ArrayXd::Zero(nSamps);

    ArrayXd   xi  = ArrayXd::Zero(blockSize);
    ArrayXXcd Xcq = ArrayXXcd::Zero(blockSize, cqt.getNumBands());
    ArrayXd   yi  = ArrayXd::Zero(blockSize);
    ArrayXXcd Ycq = ArrayXXcd::Zero(blockSize, cqt.getNumBands());

    for (Index i = 0; i < nBlocks; i++) {
        Index i0 = i * hopSize;
        xi       = x.segment(i0, blockSize) * win;

        cqt.forward(xi, Xcq);
        Ycq = Xcq.eval();
        cqt.inverse(Ycq, yi);
        y.segment(i0, blockSize) += yi * win;
    }

    ArrayXd d = y - x;

    BOOST_CHECK(rms(d) < 1e-10);
}
