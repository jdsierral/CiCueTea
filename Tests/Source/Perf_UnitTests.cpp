//
//  Perf_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 4/6/25.
//

#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <matplot/matplot.h>
#include <numbers>

#include <Benchtools.h>
#include <CQT.hpp>
#include <CQTProcessor.hpp>
#include <Slicer.hpp>
#include <Splicer.hpp>
#include <VectorOps.h>

#define NUM_SAMPLES 1 << 20
#define BLOCK_SIZE 1 << 16
#define LIB_NAME "EIGEN"
#define SAMPLE_RATE 48000
#define POINTS_PER_OCTAVE 1 / 12
#define MIN_FREQUENCY 1e2
#define MAX_FREQUENCY 1e4
#define REF_FREQUENCY 1e3

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

#include "EmptyCQTProc.h"

BOOST_AUTO_TEST_CASE(perf1)
{
    double fs          = SAMPLE_RATE;
    double fMin        = MIN_FREQUENCY;
    double fMax        = MAX_FREQUENCY;
    double fRef        = REF_FREQUENCY;
    double frac        = 1.0 / POINTS_PER_OCTAVE;
    Index  N           = NUM_SAMPLES;
    Index  blockSize   = BLOCK_SIZE;
    Index  overlapSize = blockSize / 2;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    cqtFull ola(fs, blockSize, frac, fMin, fMax, fRef);

    cout << LIB_NAME << " PERF 1" << endl;
    {
        Timer t;
        for (Index n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }

    ArrayXd d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(perf2)
{
    double fs          = SAMPLE_RATE;
    double fMin        = MIN_FREQUENCY;
    double fMax        = MAX_FREQUENCY;
    double fRef        = REF_FREQUENCY;
    double frac        = 1.0 / POINTS_PER_OCTAVE;
    Index  N           = NUM_SAMPLES;
    Index  blockSize   = BLOCK_SIZE;
    Index  overlapSize = blockSize / 2;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    sliCQTFull ola(fs, blockSize, frac, fMin, fMax, fRef);

    cout << LIB_NAME << " PERF 2" << endl;
    {
        Timer t;
        for (Index n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }

    x = x.head(N - blockSize - blockSize / 2);
    y = y.tail(N - blockSize - blockSize / 2);

    ArrayXd d = x - y;
    BOOST_CHECK(rms(d) < 1e-3);
}

BOOST_AUTO_TEST_CASE(perf3)
{
    double fs          = SAMPLE_RATE;
    double fMin        = MIN_FREQUENCY;
    double fMax        = MAX_FREQUENCY;
    double fRef        = REF_FREQUENCY;
    double frac        = 1.0 / POINTS_PER_OCTAVE;
    Index  N           = NUM_SAMPLES;
    Index  blockSize   = BLOCK_SIZE;
    Index  overlapSize = blockSize / 2;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    cqtSparse ola(fs, blockSize, frac, fMin, fMax, fRef);

    cout << LIB_NAME << " PERF 3" << endl;
    {
        Timer t;
        for (Index n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }

    ArrayXd d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(perf4)
{
    double fs          = SAMPLE_RATE;
    double fMin        = MIN_FREQUENCY;
    double fMax        = MAX_FREQUENCY;
    double fRef        = REF_FREQUENCY;
    double frac        = 1.0 / POINTS_PER_OCTAVE;
    Index  N           = NUM_SAMPLES;
    Index  blockSize   = BLOCK_SIZE;
    Index  overlapSize = blockSize / 2;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    sliCQTSparse ola(fs, blockSize, frac, fMin, fMax, fRef);

    cout << LIB_NAME << " PERF 4" << endl;
    {
        Timer t;
        for (Index n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }

    x = x.head(N - blockSize - blockSize / 2);
    y = y.tail(N - blockSize - blockSize / 2);

    ArrayXd d = x - y;
    cout << rms(d) << endl;
    BOOST_CHECK(rms(d) < 1e-3);
}
