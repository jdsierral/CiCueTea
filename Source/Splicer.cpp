//
//  Splicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Splicer.hpp"

#include "VectorOps.h"
#include "Benchtools.h"

using namespace jsa;
using namespace arma;

void Splicer::setSize(uword newBlockSize, uword newHopSize) {
    blockSize = newBlockSize;
    hopSize = newHopSize;
    overlapSize = blockSize - hopSize;
    bufferSize = nextPow2(blockSize + 1);
    buffer.resize(bufferSize);
    buffer.zeros();
    wp = 0;
    rp = constrain(bufferSize-hopSize, bufferSize);
}

void Splicer::pushBlock(const vec& block) {
    RealTimeChecker rt;
    for (uword n = 0, m = wp; n < block.size(); n++, m++) {
        m = constrain(m, bufferSize);
        buffer(m) = (n < overlapSize) * buffer(m) + block(n);
    }
    wp += hopSize;
    wp = constrain(wp, bufferSize);
}

double Splicer::getSample() {
    RealTimeChecker rt;
    double sample = buffer(rp);
    rp++;
    rp = constrain(rp, bufferSize);
    return sample;
}

