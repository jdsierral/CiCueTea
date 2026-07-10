//
//  Splicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Splicer.hpp"

#include <algorithm>

#include "MathUtils.h"
#include "RTChecker.h"

using namespace jsa;
using namespace Eigen;

Splicer::Splicer(Index newBlockSize, Index newHopSize)
{
    blockSize   = std::max<Index>(newBlockSize, 1);
    hopSize     = std::clamp<Index>(newHopSize, 1, blockSize);
    overlapSize = blockSize - hopSize;
    bufferSize  = nextPow2(size_t(blockSize + 1));
    buffer.resize(bufferSize);
    buffer.setZero();
    wp = 0;
    rp = constrain(bufferSize - hopSize, bufferSize);
}

void Splicer::pushBlock(const ArrayXd& block)
{
    RealTimeChecker rt;
    for (Index n = 0, m = wp; n < block.size(); n++, m++) {
        m         = constrain(m, bufferSize);
        // Overlap region accumulates into what is already there; past it,
        // the (n < overlapSize) mask zeroes the stale buffer content so the
        // new block overwrites it.
        buffer(m) = (n < overlapSize) * buffer(m) + block(n);
    }
    wp += hopSize;
    wp = constrain(wp, bufferSize);
}

double Splicer::getSample()
{
    RealTimeChecker rt;
    double          sample = buffer(rp);
    rp++;
    rp = constrain(rp, bufferSize);
    return sample;
}
