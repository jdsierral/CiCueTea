//
//  CQT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "CQT.hpp"

#include <iostream>

#include "Benchtools.h"

using namespace jsa;
using namespace arma;

void NsgfCqtCommon::init(double sampleRate, uword numSamples, double pointsPerOctave,
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
    Xdft.zeros();
    
    sword nBandsUp = sword(ceil(ppo * log2(fMax / fRef)));
    sword nBandsDown = sword(ceil(ppo * log2(fRef / fMin)));
    nBands = nBandsDown + nBandsUp + 1;
    bax = fRef * exp(log(2) * regspace(-nBandsDown, nBandsUp) / ppo);
    
    nFreqs = nSamps;
    fax = regspace(0, nFreqs-1) * fs / double(nFreqs);
}

//==========================================================================
//==========================================================================
//==========================================================================

void NsgfCqtFull::init(double sampleRate, uword numSamples, double ppo,
                       double minFrequency, double maxFrequency, double refFrequency)
{
    NsgfCqtCommon::init(sampleRate, numSamples, ppo,
                        minFrequency, maxFrequency, refFrequency);
    

    Xmat.resize(nSamps, nBands);
    
    double c = log(16) / (square(1.0 / ppo));
    mat outerRatio = fax * (1.0 / bax.t());
    g = exp(-c * square(log2(outerRatio)));
    
    uword end = nBands - 1;
    for (uword k = 0; k < fax.size(); k++) {
        if (fax(k) <  bax(0) ) g(k,  0 ) = 1;
        if (fax(k) > bax(end)) g(k, end) = 1;
    }
    g = sqrt(g);
    d = sum(arma::square(g), 1);
    gDual = g.each_col() / d;
    
    g.tail_rows(nFreqs/2-1).zeros();
    gDual.tail_rows(nFreqs/2-1).zeros();
    
    Xmat.zeros();
}

void NsgfCqtFull::forward(const vec& x, cx_mat& Xcq) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    assert(Xcq.n_cols == uword(nBands));
    assert(Xcq.n_rows == uword(nSamps));
    dft.rdft(x, Xdft);
    for (uword k = 0; k < nBands; k++) {
        Xmat.col(k) = 2 * (g.col(k) % Xdft);
    }
    dft.idft(Xmat, Xcq);
}

void NsgfCqtFull::inverse(const cx_mat& Xcq, vec& x) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    dft.dft(Xcq, Xmat);
    Xdft.zeros();
    for (uword k = 0; k < nBands; k++) {
        Xdft += Xmat.col(k) % gDual.col(k) / 2;
    }
    dft.irdft(Xdft, x);
}


//==========================================================================
//==========================================================================
//==========================================================================

NsgfCqtSparse::Idx findIdx(const vec& x, double th) {
    uword i0 = 0;
    uword len = 0;
    for (uword i = 0; i < x.size(); i++) {
        if (x(i) > th) {
            i0 = i;
            break;
        }
    }
    
    for (uword i = i0; i < x.size(); i++) {
        len++;
        if (x(i) < th) { break; }
    }
    return {i0, len};
}

void NsgfCqtSparse::init(double sampleRate, uword numSamples, double ppo,
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
    mat outerRatio = fax * (1.0 / bax.t());
    mat g_ = exp(-c * square(log2(outerRatio)));
    
    uword end = nBands - 1;
    g_.elem(fax < bax(1)).ones();
    g_.elem(fax > bax(end)).ones();
    g_ = sqrt(g_);
    d = sum(arma::square(g_), 1);
    mat gDual_ = g_.each_col() / d;
    
    g_.tail_rows(nFreqs/2-1).zeros();
    gDual_.tail_rows(nFreqs/2-1).zeros();
    
    using namespace std::complex_literals;
    
    for (uword k = 0; k < nBands; k++) {
        idx[k] = getIdx(g_.col(k));
        uword i0 = idx[k].i0;
        uword nCoefs = idx[k].len;
        scale(k) = double(nCoefs);
        vec n = regspace(0, nCoefs-1);
        phase[k] = exp(1i * datum::tau * double(i0) * n / double(nCoefs));
        g[k] = vec(g_.colptr(k) + i0, nCoefs);
        gDual[k] = vec(g_.colptr(k) + i0, nCoefs);
        dfts[k].init(nCoefs);
    }
    
    Xcoefs = getCoefs();
    
    dft.init(nSamps);
}

void NsgfCqtSparse::forward(const vec& x, Coefs& Xcq) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    assert(uword(Xcq.size()) == nBands);
    Xdft.fill(0);
    dft.rdft(x, Xdft);
    for (uword k = 0; k < nBands; k++) {
        Xcoefs[k] = 2.0 * g[k] % phase[k] % Xdft.subvec(idx[k].i0, idx[k].i0 + idx[k].len - 1);
        dfts[k].idft(Xcoefs[k], Xcq[k]);
    }
}

void NsgfCqtSparse::inverse(const Coefs& Xcq, vec& x) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    assert(uword(Xcq.size()) == nBands);
    Xdft.fill(0);
    for (uword k = 0; k < nBands; k++) {
        dfts[k].dft(Xcq[k], Xcoefs[k]);
        Xdft.subvec(idx[k].i0, idx[k].i0 + idx[k].len - 1) += 0.5 * gDual[k] * conj(phase[k]) * Xcoefs[k];
    }
    dft.irdft(Xdft, x);
}

NsgfCqtSparse::Idx NsgfCqtSparse::getIdx(const vec& x) {
    uword i0 = 0;
    uword i1 = x.size();
    for (uword i = 0; i < x.size(); i++) {
        if (x(i) < th) continue;
        i0 = i;
        break;
    }
    
    for (uword i = x.size()-1; i >= 0; i--) {
        if (x(i) < th) continue;
        i1 = i;
        break;
    }
    
    uword len = i1 - i0 + 1;
    if (len < 4) len = 4;
    len = uword(nextPow2((unsigned int)(len)));
    return {i0, len};
}

NsgfCqtSparse::Frame NsgfCqtSparse::getFrame() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Frame frame(nBands);
    for (uword k = 0; k < nBands; k++) {
        uword sz = g[k].size();
        frame[k].resize(sz);
    }
    return frame;
}

NsgfCqtSparse::Coefs NsgfCqtSparse::getCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Coefs coefs(nBands);
    for (uword k = 0; k < nBands; k++) {
        uword sz = g[k].size();
        coefs[k].resize(sz);
        coefs[k].zeros();
    }
    return coefs;
}

NsgfCqtSparse::Coefs NsgfCqtSparse::getValidCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Coefs coefs(nBands);
    for (uword k = 0; k < nBands; k++) {
        uword sz = g[k].size();
        assert(sz % 2 == 0);
        coefs[k].resize(sz/2);
        coefs[k].zeros();
    }
    return coefs;
}
