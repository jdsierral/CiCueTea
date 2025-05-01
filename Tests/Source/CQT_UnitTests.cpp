//
//  CQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/9/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <VectorOps.h>
#include <MathUtils.h>
#include <SignalUtils.h>


using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

namespace plt = matplot;

BOOST_AUTO_TEST_CASE(CQTTestFull1) {
    double fs = 48000;
    int nSamps = 1<<10;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    NsgfCqtFull cqt(fs, nSamps, frac, fMin, fMax, fRef);
    
    if (false) {
        cout << "nBands: " << cqt.getNumBands() << endl;
        cout << "nFreqs: " << cqt.getNumFreqs() << endl;
        cout << "nSamps: " << cqt.getNumSamps() << endl;
        
        eig2armaVec(cqt.getBandAxis()).print();
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.getNumBands(); k++) {
            ArrayXd xi = cqt.getFrequencyAxis();
            ArrayXd yi = cqt.getFrame().col(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
        plt::semilogx(cqt.getFrequencyAxis(), cqt.getDiagonalization());

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.getNumBands(); k++) {
            ArrayXd xi = cqt.getFrequencyAxis();
            ArrayXd yi = cqt.getDualFrame().col(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
    
    ArrayXd ggDual = (cqt.getFrame() * cqt.getDualFrame()).rowwise().sum().head(cqt.getNumFreqs()/2+1);
    
    BOOST_CHECK( rms(ggDual - 1) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestFull2) {
    double fs = 48000;
    size_t nSamps = 1<<10;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    NsgfCqtFull cqt(fs, nSamps, frac, fMin, fMax, fRef);
    ArrayXd t = regspace(int(nSamps)) / fs;
    ArrayXd x = (2 * M_PI * fRef * t).sin();
    ArrayXd y(nSamps);
    ArrayXXcd Xcq(cqt.getNumSamps(), cqt.getNumBands());
    
    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);
    
    if (false) {
//        eig2armaMat(Xcq).save(csv_name("Xcq.csv"));
//        eig2armaMat(cqt.Xmat).save(csv_name("Xmat.csv"));
//        eig2armaVec(cqt.Xdft).save(csv_name("Xdft.csv"));
        ArrayXd Sxx = 10 * (Xcq.abs2().colwise().sum()).log10();
        plt::figure();
        plt::semilogx(cqt.getBandAxis(), Sxx);
        plt::show();
        plt::figure();
        plt::imagesc(toStdVector(20 * Xcq.transpose().abs().log10()));
        plt::colorbar();
        plt::show();
        plt::figure();
        plt::subplot(2, 1, 0);
        plt::plot(t, x);
        plt::subplot(2, 1, 1);
        plt::plot(t, y);
        plt::show();
        
        cout << "x: " << rms(x) << endl;
        cout << "y: " << rms(y) << endl;
    }

    ArrayXd dif = x - y;
    BOOST_CHECK( rms(dif) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestSparse1) {
    double fs = 48000;
    int nSamps = 1<<10;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    NsgfCqtSparse cqt(fs, nSamps, frac, fMin, fMax, fRef);
    
    if (false) {
        cout << "nBands: " << cqt.getNumBands() << endl;
        cout << "nFreqs: " << cqt.getNumFreqs() << endl;
        cout << "nSamps: " << cqt.getNumSamps() << endl;
        eig2armaVec(cqt.getBandAxis()).print();
        
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.getNumBands(); k++) {
            auto s = cqt.getBandSpan(k);
            ArrayXd xi = cqt.getFrequencyAxis(k);
            ArrayXd yi = cqt.getAtom(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
//        plt::semilogx(cqt.getFrequencyAxis(), cqt.getDiagonalization());

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.getNumBands(); k++) {
            ArrayXd xi = cqt.getFrequencyAxis(k);
            ArrayXd yi = cqt.getDualAtom(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
        
    ArrayXd buf = ArrayXd::Zero(cqt.getNumFreqs());
    for (int k = 0; k < cqt.getNumBands(); k++) {
        auto s = cqt.getBandSpan(k);
        Index i0 = s.i0;
        Index len = s.len;
        buf.segment(i0, len) += (cqt.getAtom(k) * cqt.getDualAtom(k));
    }
    buf = buf.head(cqt.getNumFreqs()/2+1);
    double sum = (buf - 1).sum();
    BOOST_CHECK( abs(sum) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestSparse2) {
    double fs = 48000;
    int nSamps = 1<<16;
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    NsgfCqtSparse cqt(fs, nSamps, frac, fMin, fMax, fRef);
    
    ArrayXd t = regspace(int(nSamps)) / fs;
    ArrayXd x = (2 * M_PI * fRef * t + M_PI_4/2).sin();
    ArrayXd y(nSamps);
    auto Xcq = cqt.getCoefs();
    
    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);
    
    if (false) {
//        eig2armaMat(Xcq).save(csv_name("Xcq.csv"));
//        eig2armaMat(cqt.Xmat).save(csv_name("Xmat.csv"));
//        eig2armaVec(cqt.Xdft).save(csv_name("Xdft.csv"));
//        ArrayXd Sxx = 10 * (Xcq.abs2().colwise().sum()).log10();
//        plt::figure();
//        plt::semilogx(cqt.bax, Sxx);
//        plt::show();
//        plt::figure();
//        plt::imagesc(toStdVector(20 * Xcq.transpose().abs().log10()));
//        plt::colorbar();
//        plt::show();
        
        plt::figure();
        plt::subplot(2, 1, 0);
        plt::plot(t, x);
        plt::subplot(2, 1, 1);
        plt::plot(t, y);
        plt::show();
        
        cout << "x: " << rms(x) << endl;
        cout << "y: " << rms(y) << endl;
        for (int k = 0; k < cqt.getNumBands(); k++) {
            cout << k << ": " << Xcq[k].size() << endl;
        }
    }
    
    ArrayXd dif = x - y;
    BOOST_CHECK( rms(dif) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(SlidingCQT) {
    double fs = 48000;
    Index nSamps = (1<<16);
    double frac = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    Index blockSize = 1<<12;
    Index hopSize = blockSize / 2;
    Index overlapSize = blockSize - hopSize;
    Index nBlocks = (nSamps - blockSize) / hopSize;
    ArrayXd win = hann(blockSize).sqrt();
    
    NsgfCqtFull cqt(fs, blockSize, frac, fMin, fMax, fRef);
    
    ArrayXd t = regspace(nSamps) / fs;
//    ArrayXd x = (2 * M_PI * fRef * t + M_PI_4/2).sin();
    double fScale = 1.2;
    ArrayXd x = logChirp(t, fMin * fScale, fMax / fScale);
    x.head(blockSize).fill(0);
    x.tail(blockSize).fill(0);
    x *= hann(nSamps);
    ArrayXd y = ArrayXd::Zero(nSamps);
    
    ArrayXd xi = ArrayXd::Zero(blockSize);
    ArrayXd xim1 = ArrayXd::Zero(blockSize);
    ArrayXXcd Xcq = ArrayXXcd::Zero(blockSize, cqt.getNumBands());
    ArrayXXcd Xcqm1 = ArrayXXcd::Zero(blockSize, cqt.getNumBands());
    ArrayXd yi = ArrayXd::Zero(blockSize);
    ArrayXXcd Ycq = ArrayXXcd::Zero(blockSize, cqt.getNumBands());
    ArrayXXcd XVal = ArrayXXcd::Zero(blockSize/2, cqt.getNumBands());
    ArrayXXcd YVal = ArrayXXcd::Zero(blockSize/2, cqt.getNumBands());
    
    ArrayXXcd X = ArrayXXcd::Zero(nSamps, cqt.getNumBands()); X.fill(0);

    for (Index i = 0; i < nBlocks; i++) {
        Index i0 = i * hopSize;
        xi = x.segment(i0, blockSize) * win;
        
        cqt.forward(xi, Xcq);
//        XVal = Xcq.topRows(overlapSize) + Xcqm1.bottomRows(overlapSize);
        
        for (int k = 0; k < cqt.getNumBands(); k++) {
//            Ycq.col(k).segment(i0, overlapSize) = Xcqm1.col(k) * win.tail(overlapSize);
//            Ycq.col(k).segment(i0, overlapSize) = XVal.col(k) * win.head(overlapSize);
            X.col(k).segment(i0, blockSize) += Xcq.col(k) * win;
        }
        Ycq = Xcq.eval();
        cqt.inverse(Ycq, yi);
        y.segment(i0, blockSize) += yi * win;
        
        
        
        
        Xcqm1 = Xcq.eval();
    }
    
    cout << "Norm: " << X.matrix().norm() << endl;
    
    if (false) {
        //    eig2armaVec(x).save(csv_name("xvec.csv"));
        //    eig2armaMat(X).save(csv_name("Xmat.csv"));
    }
        
    ArrayXd d = y - x;
    
    BOOST_CHECK(rms(d) < 1e-10);
}

