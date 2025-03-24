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

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

namespace plt = matplot;

BOOST_AUTO_TEST_CASE(DFTTest1) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    ArrayXd x = ArrayXd::Ones(fftSize);
    ArrayXcd X (fftSize/2+1);
    dft.rdft(x, X);
    
    if (false) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(X[0] == dcomplex(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest2) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    ArrayXd x = ArrayXd::Ones(fftSize);
    ArrayXcd X (fftSize/2+1);
    ArrayXd y = ArrayXd::Zero(fftSize);
    dft.rdft(x, X);
    dft.irdft(X, y);
    
    if (false) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(y[0] == 1);
}

BOOST_AUTO_TEST_CASE(DFTTest3) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y (fftSize);
    dft.dft(X, Y);
    
    if (false) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(Y[0] == dcomplex(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest4) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y (fftSize);
    dft.idft(X, Y);
    
    if (false) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(Y[0] == dcomplex(1));
}

BOOST_AUTO_TEST_CASE(CQTTestFull1) {
    NsgfCqtFull cqt;
    double fs = 48000;
    int nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    if (false) {
        cout << "nBands: " << cqt.nBands << endl;
        cout << "nFreqs: " << cqt.nFreqs << endl;
        cout << "nSamps: " << cqt.nSamps << endl;
        
        eig2armaVec(cqt.bax).print();
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.nBands; k++) {
            ArrayXd xi = cqt.fax;
            ArrayXd yi = cqt.g.col(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
        plt::semilogx(cqt.fax, cqt.d);

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.nBands; k++) {
            ArrayXd xi = cqt.fax;
            ArrayXd yi = cqt.gDual.col(k);
            plt::semilogx(cqt.fax, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
    
    ArrayXd ggDual = (cqt.g * cqt.gDual).rowwise().sum().head(cqt.nFreqs/2+1);
    
    BOOST_CHECK( rms(ggDual - 1) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestFull2) {
    NsgfCqtFull cqt;
    double fs = 48000;
    size_t nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    ArrayXd t = regspace(int(nSamps)) / fs;
    ArrayXd x = (2 * M_PI * fRef * t).sin();
    ArrayXd y(nSamps);
    ArrayXXcd Xcq(cqt.nSamps, cqt.nBands);
    
    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);
    
    if (false) {
//        eig2armaMat(Xcq).save(csv_name("Xcq.csv"));
//        eig2armaMat(cqt.Xmat).save(csv_name("Xmat.csv"));
//        eig2armaVec(cqt.Xdft).save(csv_name("Xdft.csv"));
        ArrayXd Sxx = 10 * (Xcq.abs2().colwise().sum()).log10();
        plt::figure();
        plt::semilogx(cqt.bax, Sxx);
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
    NsgfCqtSparse cqt;
    double fs = 48000;
    int nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    
    if (false) {
        cout << "nBands: " << cqt.nBands << endl;
        cout << "nFreqs: " << cqt.nFreqs << endl;
        cout << "nSamps: " << cqt.nSamps << endl;
        eig2armaVec(cqt.bax).print();
        
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.nBands; k++) {
            ArrayXd xi = cqt.fax.segment(cqt.idx[k].i0, cqt.idx[k].len);
            ArrayXd yi = cqt.g[k];
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
        plt::semilogx(cqt.fax, cqt.d);

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.nBands; k++) {
            ArrayXd xi = cqt.fax.segment(cqt.idx[k].i0, cqt.idx[k].len);
            ArrayXd yi = cqt.gDual[k];
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
        
    ArrayXd buf = ArrayXd::Zero(cqt.nFreqs);
    for (int k = 0; k < cqt.nBands; k++) {
        Index i0 = cqt.idx[k].i0;
        Index len = cqt.idx[k].len;
        buf.segment(i0, len) += (cqt.g[k] * cqt.gDual[k]);
    }
    buf = buf.head(cqt.nFreqs/2+1);
    double sum = (buf - 1).sum();
    BOOST_CHECK( abs(sum) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestSparse2) {
    NsgfCqtSparse cqt;
    double fs = 48000;
    int nSamps = 1<<16;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    
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
        for (int k = 0; k < cqt.nBands; k++) {
            cout << k << ": " << Xcq[k].size() << endl;
        }
    }
    
    ArrayXd dif = x - y;
    BOOST_CHECK( rms(dif) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(SlidingCQT) {
    NsgfCqtFull cqt;
    double fs = 48000;
    Index nSamps = (1<<16);
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    Index blockSize = 1<<12;
    Index hopSize = blockSize / 2;
    Index overlapSize = blockSize - hopSize;
    Index nBlocks = (nSamps - blockSize) / hopSize;
    ArrayXd win = hann(blockSize).sqrt();
    
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
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
    ArrayXXcd Xcq = ArrayXXcd::Zero(blockSize, cqt.nBands);
    ArrayXXcd Xcqm1 = ArrayXXcd::Zero(blockSize, cqt.nBands);
    ArrayXd yi = ArrayXd::Zero(blockSize);
    ArrayXXcd Ycq = ArrayXXcd::Zero(blockSize, cqt.nBands);
    ArrayXXcd XVal = ArrayXXcd::Zero(blockSize/2, cqt.nBands);
    ArrayXXcd YVal = ArrayXXcd::Zero(blockSize/2, cqt.nBands);
    
    ArrayXXcd X = ArrayXXcd::Zero(nSamps, cqt.nBands); X.fill(0);

    for (Index i = 0; i < nBlocks; i++) {
        Index i0 = i * hopSize;
        xi = x.segment(i0, blockSize) * win;
        
        cqt.forward(xi, Xcq);
//        XVal = Xcq.topRows(overlapSize) + Xcqm1.bottomRows(overlapSize);
        
        for (int k = 0; k < cqt.nBands; k++) {
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

