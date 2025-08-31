//
//  Slicing_UnitTests.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <matplot/matplot.h>
#include <numbers>

#include <CQT.hpp>
#include <CQTProcessor.hpp>
#include <SignalUtils.h>
#include <Slicer.hpp>
#include <Splicer.hpp>

#include "VectorOps.h"

using namespace Eigen;
using namespace std;
using namespace jsa;

BOOST_AUTO_TEST_CASE(Slicing1)
{

    Index N         = (1 << 8);
    Index blockSize = 8;
    Index hopSize   = 4;
    Slicer slicer(blockSize, hopSize);
    
    ArrayXd x = ArrayXd::LinSpaced(N, 0, N - 1);

    for (Index n = 0; n < N; n++) {
        slicer.pushSample(x(n));
        if (slicer.hasBlock()) {
            //            std::cout << slicer.getBlock().transpose() << std::endl;
        }
    }
}

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

    Slicer  slicer(blockSize, hopSize);
    Splicer composer(blockSize, hopSize);
    NsgfCqtFull cqt(fs, blockSize, frac, fMin, fMax, fRef);

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

BOOST_AUTO_TEST_CASE(CQTSlicing2)
{
    return;
    
    double fs   = 48000;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;

    Index blockSize   = 1 << 16;
    Index hopSize     = blockSize / 2;
    Index overlapSize = blockSize - hopSize;

    Index   N = (1 << 20);
    ArrayXd t = regspace(N) / fs;
    //    ArrayXd x = ArrayXd::Random(N);
    ArrayXd x = logChirp(t, fMin, fMax);
    ArrayXd y = ArrayXd::Zero(N);
    ArrayXd w = hann(blockSize);
    x.head(blockSize).setZero();
    x.tail(blockSize).setZero();

    Slicer  slicer(blockSize, hopSize);
    Splicer composer(blockSize, hopSize);
    NsgfCqtFull cqt(fs, blockSize, frac, fMin, fMax, fRef);

    Index     nBands = cqt.getNumBands();
    ArrayXXcd Xm1(blockSize, nBands);
    ArrayXXcd X_i(blockSize, nBands);
    ArrayXXcd Y_i(blockSize, nBands);
    ArrayXXcd Z_i(overlapSize, nBands);
    ArrayXXcd Zm1(overlapSize, nBands);

    Xm1.setZero();
    X_i.setZero();
    Z_i.setZero();
    Zm1.setZero();
    Y_i.setZero();

    ArrayXd xi(blockSize);
    ArrayXd yi(blockSize);

    for (Index n = 0; n < N; n++) {
        double sample = x(n);
        slicer.pushSample(sample);
        y(n) = composer.getSample();

        if (slicer.hasBlock()) {
            xi = slicer.getBlock();
            xi *= w;
            Xm1 = X_i;
            cqt.forward(xi, X_i);
            Zm1 = Z_i;
            Z_i = Xm1.bottomRows(overlapSize) + X_i.topRows(overlapSize);

            Y_i.topRows(overlapSize)    = Zm1;
            Y_i.bottomRows(overlapSize) = Z_i;

            cqt.inverse(Y_i, yi);
            yi *= w;
            composer.pushBlock(yi);
        }
    }
    y         = y.tail(N - blockSize - overlapSize);
    x         = x.head(N - blockSize - overlapSize);
    ArrayXd d = x - y;
    eig2armaVec(x).save(arma::csv_name("x.csv"));
    eig2armaVec(y).save(arma::csv_name("y.csv"));
    cout << rms(d) << endl;
    BOOST_CHECK(rms(d) < 1e-3);
}
