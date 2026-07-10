//
//  Slicing_UnitTests.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//
//  Tests for the slicing/splicing machinery that turns the block transform
//  into a stream processor: the Slicer chops a sample stream into overlapped
//  blocks, the Splicer overlap-adds processed blocks back into a stream.
//

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include <CQT.hpp>
#include <SignalUtils.h>
#include <Slicer.hpp>
#include <Splicer.hpp>

#include "VectorOps.h"

using namespace Eigen;
using namespace std;
using namespace jsa;

// Slicer contract, verified on a ramp input. The slicer streams from time
// zero (its buffer starts zeroed), so the first blockSize/hopSize − 1 blocks
// contain zero-padded warmup. The contract is:
//   - a block is emitted every hopSize samples, and its last element is
//     always the newest sample;
//   - after warmup, blocks hold consecutive samples (step exactly 1) and
//     consecutive block starts are hopSize apart.
BOOST_AUTO_TEST_CASE(Slicing1)
{
    Index  N         = (1 << 8);
    Index  blockSize = 8;
    Index  hopSize   = 4;
    Slicer slicer(blockSize, hopSize);

    ArrayXd x = ArrayXd::LinSpaced(N, 0, N - 1);

    Index  nWarmup   = blockSize / hopSize - 1;
    Index  nBlocks   = 0;
    double prevFirst = 0;
    for (Index n = 0; n < N; n++) {
        slicer.pushSample(x(n));
        if (slicer.hasBlock()) {
            ArrayXd block = slicer.getBlock();
            BOOST_CHECK(block(blockSize - 1) == x(n));
            if (nBlocks >= nWarmup) {
                ArrayXd step = block.tail(blockSize - 1) - block.head(blockSize - 1);
                BOOST_CHECK((step - 1).abs().maxCoeff() < 1e-12);
            }
            if (nBlocks > nWarmup) {
                BOOST_CHECK(block(0) - prevFirst == double(hopSize));
            }
            prevFirst = block(0);
            nBlocks++;
        }
    }
    BOOST_CHECK(nBlocks == N / hopSize);
}

// Slicer → Splicer identity (no transform in between): with a √Hann analysis
// and synthesis window at 50% overlap the WOLA condition holds, so the output
// must equal the input delayed by one block.
BOOST_AUTO_TEST_CASE(Slicing2)
{

    Index blockSize = 1 << 10;
    Index hopSize   = blockSize / 2;

    Slicer  slicer(blockSize, hopSize);
    Splicer splicer(blockSize, hopSize);

    Index   N = (1 << 16);
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    ArrayXd window = hann(blockSize).sqrt();

    for (Index n = 0; n < N; n++) {
        double sample = x(n);
        slicer.pushSample(sample);
        sample = splicer.getSample();

        if (slicer.hasBlock()) {
            ArrayXd block = slicer.getBlock();
            block *= window;
            block *= window;
            splicer.pushBlock(block);
        }

        y(n) = sample;
    }

    ArrayXd d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-6);
}

// Full sliced chain: Slicer → CQT forward → inverse → Splicer, with a Hann
// analysis window absorbed into the slice. The identity manipulation must
// reconstruct the stream (delayed by one block).
BOOST_AUTO_TEST_CASE(CQTSlicing1)
{
    double fs   = 48000;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;

    Index blockSize = 1 << 10;
    Index hopSize   = blockSize / 2;

    Index   N = (1 << 16);
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    ArrayXd w = hann(blockSize);

    Slicer       slicer(blockSize, hopSize);
    Splicer      composer(blockSize, hopSize);
    NsgfCqtDense cqt(fs, blockSize, frac, fMin, fMax, fRef);

    Index     nBands = cqt.getNumBands();
    ArrayXXcd Xcq(blockSize, nBands);
    ArrayXXcd Ycq(blockSize, nBands);
    Xcq.setZero();
    Ycq.setZero();

    ArrayXd xi(blockSize);
    ArrayXd yi(blockSize);

    for (Index n = 0; n < N; n++) {
        double sample = x(n);
        slicer.pushSample(sample);
        y(n) = composer.getSample();

        if (slicer.hasBlock()) {
            xi = slicer.getBlock();
            xi *= w;
            cqt.forward(xi, Xcq);
            Ycq = Xcq;
            cqt.inverse(Ycq, yi);
            composer.pushBlock(yi);
        }
    }

    ArrayXd d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-6);
}
