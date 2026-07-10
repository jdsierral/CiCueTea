//
//  Slicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Slicer.hpp"

#include <algorithm>

#include "MathUtils.h"
#include "RTChecker.h"

using namespace jsa;
using namespace Eigen;

Slicer::Slicer(Index newBlockSize, Index newHopSize)
{
    blockSize   = std::max<Index>(newBlockSize, 1);
    hopSize     = std::clamp<Index>(newHopSize, 1, blockSize);
    overlapSize = blockSize - hopSize;
    bufferSize  = nextPow2(size_t(blockSize + 1));
    buffer.resize(2 * bufferSize);
    buffer.setZero();
    wp = 0;
    rp = constrain(bufferSize - overlapSize, bufferSize);
}

void Slicer::pushSample(double sample)
{
    RealTimeChecker rt;
    wp         = constrain(wp, bufferSize);
    buffer(wp) = buffer(wp + bufferSize) = sample;
    wp++;
}

bool Slicer::hasBlock()
{
    RealTimeChecker rt;
    return (wp % hopSize) == 0;
}

Map<const ArrayXd> Slicer::getBlock()
{
    RealTimeChecker rt;
    rp           = constrain(rp, bufferSize);
    auto segment = buffer.segment(rp, blockSize);
    rp += hopSize;
    return Map<const ArrayXd>(segment.data(), segment.size());
}
