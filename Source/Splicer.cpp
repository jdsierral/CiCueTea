//
//  Splicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Splicer.hpp"

#include "Benchtools.h"
#include "MathUtils.h"

using namespace jsa;
using namespace Eigen;

void Splicer::setSize(Index newBlockSize, Index newHopSize) {
    blockSize = newBlockSize;
    hopSize = newHopSize;
    overlapSize = blockSize - hopSize;
    bufferSize = nextPow2(blockSize + 1);
    buffer.resize(bufferSize);
    buffer.setZero();
    wp = 0;
    rp = constrain(bufferSize-hopSize, bufferSize);
}

void Splicer::pushBlock(const ArrayXd& block) {
    RealTimeChecker rt;
    for (Index n = 0, m = wp; n < block.size(); n++, m++) {
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

