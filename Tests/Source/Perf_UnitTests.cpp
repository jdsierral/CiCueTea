//
//  Perf_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 4/6/25.
//

#pragma once

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <OverlapAddProcessor.hpp>
#include <Splicer.hpp>
#include <Slicer.hpp>
#include <OverlapAddProcessor.hpp>
#include <VectorOps.h>

#define NUM_SAMPLES 1<<20
#define BLOCK_SIZE 1<<16
#define LIB_NAME "ARMADILLO"
#define SAMPLE_RATE 48000
#define POINTS_PER_OCTAVE 12
#define MIN_FREQUENCY 1e2
#define MAX_FREQUENCY 1e4
#define REF_FREQUENCY 1e3

using namespace arma;
using namespace std;
using namespace jsa;

class cqtFull : public jsa::cqtFullProcessor
{
public:
    void processBlock(arma::cx_mat& block) override {}
};

class sliCQTFull : public jsa::slidingCQTFullProcessor
{
public:
    void processBlock(arma::cx_mat& block) override {};
};

class cqtSparse : public jsa::cqtSparseProcessor
{
public:
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {}
};

class sliCQTSparse : public jsa::slidingCqtSparseProcessor
{
public:
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {};
};

BOOST_AUTO_TEST_CASE(perf1) {
    double fs = SAMPLE_RATE;
    double fMin = MIN_FREQUENCY;
    double fMax = MAX_FREQUENCY;
    double fRef = REF_FREQUENCY;
    double ppo  = POINTS_PER_OCTAVE;
    uword N = NUM_SAMPLES;
    uword blockSize = BLOCK_SIZE;
    uword overlapSize = blockSize/2;
    
    vec x = arma::randn(N);
    vec y = arma::zeros(N);
    
    cqtFull ola;
    ola.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    cout << LIB_NAME << " PERF 1" << endl;
    {
        Timer t;
        for (uword n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }
    x.save(csv_name("x.csv"));
    y.save(csv_name("y.csv"));
    vec d = x.head(N-blockSize) - y.tail(N-blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(perf2) {
    double fs = SAMPLE_RATE;
    double fMin = MIN_FREQUENCY;
    double fMax = MAX_FREQUENCY;
    double fRef = REF_FREQUENCY;
    double ppo = POINTS_PER_OCTAVE;
    uword N = NUM_SAMPLES;
    uword blockSize = BLOCK_SIZE;
    uword overlapSize = blockSize/2;
    
    vec x = arma::randn(N);
    vec y = arma::zeros(N);
    
    sliCQTFull ola;
    ola.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    cout << LIB_NAME << " PERF 2" << endl;
    {
        Timer t;
        for (uword n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }
    
    x = x.head(N-blockSize-blockSize/2);
    y = y.tail(N-blockSize-blockSize/2);
    
    vec d = x - y;
    BOOST_CHECK(rms(d) < 1e-3);
}

BOOST_AUTO_TEST_CASE(perf3) {
    double fs = SAMPLE_RATE;
    double fMin = MIN_FREQUENCY;
    double fMax = MAX_FREQUENCY;
    double fRef = REF_FREQUENCY;
    double ppo = POINTS_PER_OCTAVE;
    uword N = NUM_SAMPLES;
    uword blockSize = BLOCK_SIZE;
    uword overlapSize = blockSize/2;
    
    vec x = arma::randn(N);
    vec y = arma::zeros(N);
    
    cqtSparse ola;
    ola.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    cout << LIB_NAME << " PERF 3" << endl;
    {
        Timer t;
        for (uword n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }
    
    vec d = x.head(N-blockSize) - y.tail(N-blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(perf4) {
    double fs = SAMPLE_RATE;
    double fMin = MIN_FREQUENCY;
    double fMax = MAX_FREQUENCY;
    double fRef = REF_FREQUENCY;
    double ppo = POINTS_PER_OCTAVE;
    uword N = NUM_SAMPLES;
    uword blockSize = BLOCK_SIZE;
    uword overlapSize = blockSize/2;
    
    vec x = arma::randn(N);
    vec y = arma::zeros(N);
    
    sliCQTSparse ola;
    ola.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    cout << LIB_NAME << " PERF 1" << endl;
    {
        Timer t;
        for (uword n = 0; n < N; n++) {
            y(n) = ola.processSample(x(n));
        }
    }
        
    x = x.head(N-blockSize-blockSize/2);
    y = y.tail(N-blockSize-blockSize/2);
    
    vec d = x - y;
    BOOST_CHECK(rms(d) < 1e-3);
}
