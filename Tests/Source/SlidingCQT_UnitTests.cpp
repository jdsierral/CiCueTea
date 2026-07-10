//
//  SlidingCQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/23/25.
//
//  Layer 3 — processor-class API tests. Each CQT{Dense,Sparse} /
//  SliCqt{Dense,Sparse} processor is driven sample-by-sample through an
//  identity manipulation (EmptyCQTProc.h) and must reconstruct its input
//  delayed by exactly getLatency() samples — so these tests verify the
//  round trip and the latency report at the same time.
//

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "EmptyCQTProc.h"
#include "VectorOps.h"

using namespace Eigen;
using namespace std;
using namespace jsa;

// Block-processor, dense: exact reconstruction (painless frame, no sliding).
BOOST_AUTO_TEST_CASE(OlaProc1)
{
    double fs        = 48000;
    Index  N         = 1 << 10;
    Index  blockSize = 1 << 5;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    CqtDense ola(fs, blockSize, 1, 1e2, 1e4, 1e3);

    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }

    Index   latency = ola.getLatency();
    ArrayXd d       = x.head(N - latency) - y.tail(N - latency);
    BOOST_CHECK_MESSAGE(rms(d) < 1e-10, "rms = " << rms(d));
}

// Sliding processor, dense: reconstruction through the sliCQ-style
// overlapped path.
BOOST_AUTO_TEST_CASE(OlaProc2)
{
    double fs        = 48000;
    Index  N         = 1 << 20;
    Index  blockSize = 1 << 16;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    SliCqtDense ola(fs, blockSize, 1, 1e2, 1e4, 1e3);

    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }

    Index   latency = ola.getLatency();
    ArrayXd d       = x.head(N - latency) - y.tail(N - latency);
    BOOST_CHECK_MESSAGE(rms(d) < 1e-3, "rms = " << rms(d));
}

// Block-processor, sparse: exact reconstruction.
BOOST_AUTO_TEST_CASE(OlaProc3)
{
    double fs        = 48000;
    Index  N         = 1 << 16;
    Index  blockSize = 1 << 10;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    CqtSparse ola(fs, blockSize, 1, 1e2, 1e4, 1e3);

    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }

    Index   latency = ola.getLatency();
    ArrayXd d       = x.head(N - latency) - y.tail(N - latency);
    BOOST_CHECK_MESSAGE(rms(d) < 1e-10, "rms = " << rms(d));
}

// Sliding processor, sparse, at 3 bands/octave.
BOOST_AUTO_TEST_CASE(OlaProc4)
{
    double fs        = 48000;
    Index  N         = 1 << 18;
    Index  blockSize = 1 << 16;
    double frac      = 1.0 / 3.0;
    double fMax      = 1e4;
    double fMin      = 1e2;
    double fRef      = 1e3;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    SliCqtSparse ola(fs, blockSize, frac, fMin, fMax, fRef);

    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }

    Index   latency = ola.getLatency();
    ArrayXd d       = x.head(N - latency) - y.tail(N - latency);
    BOOST_CHECK_MESSAGE(rms(d) < 1e-3, "rms = " << rms(d));
}
