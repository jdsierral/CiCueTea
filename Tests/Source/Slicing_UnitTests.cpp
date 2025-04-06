//
//  Slicing_UnitTests.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <OverlapAddProcessor.hpp>
#include <Splicer.hpp>
#include <Slicer.hpp>
#include <VectorOps.h>

using namespace arma;
using namespace std;
using namespace jsa;

BOOST_AUTO_TEST_CASE(Slicing1) {
    Slicer slicer;
    
    uword N = (1<<8);
    uword blockSize = 8;
    uword hopSize = 4;
    
    slicer.setSize(blockSize, hopSize);
    vec x = regspace(0, N-1);
    
    for (uword n = 0; n < N; n++) {
        slicer.pushSample(x(n));
        if (slicer.hasBlock()) {
//            std::cout << slicer.getBlock().transpose() << std::endl;
        }
    }
}

BOOST_AUTO_TEST_CASE(Slicing2) {
    Slicer slicer;
    Splicer splicer;
    
    uword blockSize = 1<<10;
    uword hopSize = blockSize/2;
    
    slicer.setSize(blockSize, hopSize);
    splicer.setSize(blockSize, hopSize);
    
    uword N = (1<<16);
    vec x = randn(N);
    vec y = zeros(N);
    
    vec window = sqrt(hann(blockSize));
    
    for (uword n = 0; n < N; n++) {
        double sample = x(n);
        slicer.pushSample(sample);
        sample = splicer.getSample();
        
        if (slicer.hasBlock()) {
            vec block = slicer.getBlock();
            block *= window;
            block *= window;
            splicer.pushBlock(block);
        }
        
        y(n) = sample;
    }
    
    vec d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-6);
}

BOOST_AUTO_TEST_CASE(CQTSlicing1) {
    NsgfCqtFull cqt;
    Slicer slicer;
    Splicer composer;
    
    double fs = 48000;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    uword blockSize = 1<<10;
    uword hopSize = blockSize/2;
    
    uword N = (1<<16);
    vec x = randn(N);
    vec y = zeros(N);
    vec w = hann(blockSize);
    
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    slicer.setSize(blockSize, hopSize);
    composer.setSize(blockSize, hopSize);
    
    uword nBands = cqt.nBands;
    cx_mat Xcq(blockSize, nBands);
    cx_mat Ycq(blockSize, nBands);
    Xcq.zeros();
    Ycq.zeros();
    
    vec xi(blockSize);
    vec yi(blockSize);
    
    for (uword n = 0; n < N; n++) {
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
    
    vec d = x.head(N - blockSize) - y.tail(N - blockSize);
    BOOST_CHECK(rms(d) < 1e-6);
}


BOOST_AUTO_TEST_CASE(CQTSlicing2) {
    return;
    NsgfCqtFull cqt;
    Slicer slicer;
    Splicer composer;
    
    double fs = 48000;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    uword blockSize = 1<<16;
    uword hopSize = blockSize/2;
    uword overlapSize = blockSize - hopSize;
    
    uword N = (1<<20);
    vec t = regspace(0, N-1) / fs;
//    ArrayXd x = ArrayXd::Random(N);
    vec x = logChirp(t, fMin, fMax);
    vec y = zeros(N);
    vec w = hann(blockSize);
    x.head(blockSize).zeros();
    x.tail(blockSize).zeros();
    
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    slicer.setSize(blockSize, hopSize);
    composer.setSize(blockSize, hopSize);
    
    uword nBands = cqt.nBands;
    cx_mat Xm1(blockSize, nBands);
    cx_mat X_i(blockSize, nBands);
    cx_mat Y_i(blockSize, nBands);
    cx_mat Z_i(overlapSize, nBands);
    cx_mat Zm1(overlapSize, nBands);
    
    Xm1.zeros();
    X_i.zeros();
    Z_i.zeros();
    Zm1.zeros();
    Y_i.zeros();
    
    vec xi(blockSize);
    vec yi(blockSize);
    
    for (uword n = 0; n < N; n++) {
        double sample = x(n);
        slicer.pushSample(sample);
        y(n) = composer.getSample();
        
        if (slicer.hasBlock()) {
            xi = slicer.getBlock();
            xi *= w;
            Xm1 = X_i;
            cqt.forward(xi, X_i);
            Zm1 = Z_i;
            Z_i = Xm1.tail_rows(overlapSize) + X_i.head_rows(overlapSize);
            
            Y_i.head_rows(overlapSize) = Zm1;
            Y_i.tail_rows(overlapSize) = Z_i;
            
            cqt.inverse(Y_i, yi);
            yi *= w;
            composer.pushBlock(yi);
        }
    }
    y = y.tail(N - blockSize - overlapSize);
    x = x.head(N - blockSize - overlapSize);
    vec d = x - y;
    cout << rms(d) << endl;
    BOOST_CHECK(rms(d) < 1e-3);
}
