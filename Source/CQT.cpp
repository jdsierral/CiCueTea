//
//  CQT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "CQT.hpp"

#include <iostream>

#include "Benchtools.h"

//#define REALTIME_CHECKS
//
//#ifdef REALTIME_CHECKS
//  #define ENTERING_REAL_TIME_CRITICAL_CODE Eigen::internal::set_is_malloc_allowed(false);
//  #define EXITING_REAL_TIME_CRITICAL_CODE Eigen::internal::set_is_malloc_allowed(true);
//#else
//  #define ENTERING_REAL_TIME_CRITICAL_CODE
//  #define EXITING_REAL_TIME_CRITICAL_CODE
//#endif

using namespace jsa;
using namespace Eigen;

inline double square(double x) { return x * x; }
inline ArrayXd regspace(Index num) {
    return ArrayXd::LinSpaced(num, 0, num-1);
}

inline ArrayXd regspace(Index low, Index high) {
    return ArrayXd::LinSpaced(high-low+1, low, high);
}

constexpr uint32_t nextPow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

void NsgfCqtCommon::init(double sampleRate, Index numSamples, double pointsPerOctave,
                         double minFrequency, double maxFrequency, double refFrequency)
{
    fs = sampleRate;
    nSamps = numSamples;
    ppo = pointsPerOctave;
    fMin = minFrequency;
    fMax = maxFrequency;
    fRef = refFrequency;
    
    // Initialize Stuff
    Xdft.resize(nSamps);
    dft.init(nSamps);
    
    Index nBandsUp = Index(ceil(ppo * log2(fMax / fRef)));
    Index nBandsDown = Index(ceil(ppo * log2(fRef / fMin)));
    nBands = nBandsDown + nBandsUp + 1;
    bax = fRef * (log(2) * regspace(-nBandsDown, nBandsUp) / ppo).exp();
    
    nFreqs = nSamps;
    fax = ArrayXd::LinSpaced(nFreqs, 0, nFreqs-1) * fs / double(nFreqs);
}

//==========================================================================
//==========================================================================
//==========================================================================

void NsgfCqtFull::init(double sampleRate, Index numSamples, double ppo,
                       double minFrequency, double maxFrequency, double refFrequency)
{
    NsgfCqtCommon::init(sampleRate, numSamples, ppo,
                        minFrequency, maxFrequency, refFrequency);
    

    Xmat.resize(nSamps, nBands);
    
    double c = log(16) / (square(1.0 / ppo));
    ArrayXXd outerRatio = fax.rowwise().replicate(bax.size()) / bax.transpose().colwise().replicate(fax.size());
    g = (-c * (outerRatio).log2().square()).exp();
    
    Index end = nBands - 1;
    g.col( 0 ) = (fax < bax( 0 )).select(1, g.col( 0 ));
    g.col(end) = (fax > bax(end)).select(1, g.col(end));
    g = g.sqrt();
    d = g.square().rowwise().sum();
    gDual = g.colwise() / d;
    
    g.bottomRows(nFreqs/2-1).fill(0);
    gDual.bottomRows(nFreqs/2-1).fill(0);
}

void NsgfCqtFull::forward(const ArrayXd& x, ArrayXXcd& Xcq) {
    RealTimeChecker ck;
    assert(Xcq.cols() == Index(nBands));
    assert(Xcq.rows() == Index(nSamps));
    dft.rdft(x, Xdft);
    for (Index k = 0; k < nBands; k++) {
        Xmat.col(k) = 2 * g.col(k) * Xdft;
    }
    dft.idft(Xmat, Xcq);
}

void NsgfCqtFull::inverse(const ArrayXXcd& Xcq, ArrayXd& x) {
    RealTimeChecker ck;
    dft.dft(Xcq, Xmat);
    Xdft = (Xmat * gDual).rowwise().sum() / 2;
    dft.irdft(Xdft, x);
}


//==========================================================================
//==========================================================================
//==========================================================================

NsgfCqtSparse::Idx findIdx(const ArrayXd& x, double th) {
    Index i0 = 0;
    Index len = 0;
    for (Index i = 0; i < x.size(); i++) {
        if (x(i) > th) {
            i0 = i;
            break;
        }
    }
    
    for (Index i = i0; i < x.size(); i++) {
        len++;
        if (x(i) < th) { break; }
    }
    return {i0, len};
}

void NsgfCqtSparse::init(double sampleRate, Index numSamples, double ppo,
                         double minFrequency, double maxFrequency, double refFrequency)
{
    NsgfCqtCommon::init(sampleRate, numSamples, ppo,
                        minFrequency, maxFrequency, refFrequency);
    
    // Initialize Stuff
    idx.resize(nBands);
    g.resize(nBands);
    gDual.resize(nBands);
    phase.resize(nBands);
    scale.resize(nBands);
    Xdft.resize(nSamps);
    dfts.resize(nBands);
    
    double c = log(16) / (square(1.0 / ppo));
    ArrayXXd outerRatio = fax.rowwise().replicate(bax.size()) / bax.transpose().colwise().replicate(fax.size());
    ArrayXXd g_ = (-c * (outerRatio).log2().square()).exp();
    
    Index end = nBands - 1;
    g_.col( 0 ) = (fax < bax( 0 )).select(1, g_.col( 0 ));
    g_.col(end) = (fax > bax(end)).select(1, g_.col(end));
    g_ = g_.sqrt();
    d = g_.square().rowwise().sum();
    ArrayXXd gDual_ = g_.colwise() / d;
    
    g_.bottomRows(nFreqs/2-1).fill(0);
    gDual_.bottomRows(nFreqs/2-1).fill(0);
    
    using namespace std::complex_literals;
    
    for (Index k = 0; k < nBands; k++) {
        idx[k] = getIdx(g_.col(k));
        Index i0 = idx[k].i0;
        Index nCoefs = idx[k].len;
        scale(k) = double(nCoefs);
        ArrayXd n = ArrayXd::LinSpaced(nCoefs, 0, nCoefs-1);
        phase[k] = exp(1i * 2.0 * M_PI * double(i0) * n / double(nCoefs));
        g[k] = g_.col(k).segment(i0, nCoefs).eval();
        gDual[k] = gDual_.col(k).segment(i0, nCoefs).eval();
        dfts[k].init(nCoefs);
    }
    
    Xcoefs = getCoefs();
    
    dft.init(nSamps);
}

void NsgfCqtSparse::forward(const ArrayXd& x, Coefs& Xcq) {
    RealTimeChecker ck;
    assert(Index(Xcq.size()) == nBands);
    Xdft.fill(0);
    dft.rdft(x, Xdft);
    for (Index k = 0; k < nBands; k++) {
        Xcoefs[k] = 2.0 * g[k] * phase[k] * Xdft.segment(idx[k].i0, idx[k].len);
        dfts[k].idft(Xcoefs[k], Xcq[k]);
    }
}

void NsgfCqtSparse::inverse(const Coefs& Xcq, ArrayXd& x) {
    RealTimeChecker ck;
    assert(Index(Xcq.size()) == nBands);
    Xdft.fill(0);
    for (Index k = 0; k < nBands; k++) {
        dfts[k].dft(Xcq[k], Xcoefs[k]);
        Xdft.segment(idx[k].i0, idx[k].len) += 0.5 * gDual[k] * phase[k].conjugate() * Xcoefs[k];
    }
    dft.irdft(Xdft, x);
}

NsgfCqtSparse::Idx NsgfCqtSparse::getIdx(const ArrayXd& x) {
    Index i0 = 0;
    Index i1 = x.size();
    for (Index i = 0; i < x.size(); i++) {
        if (x(i) < th) continue;
        i0 = i;
        break;
    }
    
    for (Index i = x.size()-1; i >= 0; i--) {
        if (x(i) < th) continue;
        i1 = i;
        break;
    }
    
    Index len = i1 - i0 + 1;
    if (len < 4) len = 4;
    len = Index(nextPow2((unsigned int)(len)));
    return {i0, len};
}

NsgfCqtSparse::Coefs NsgfCqtSparse::getCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Coefs coefs(nBands);
    for (Index k = 0; k < nBands; k++) {
        coefs[k].resize(g[k].size());
    }
    return coefs;
}
