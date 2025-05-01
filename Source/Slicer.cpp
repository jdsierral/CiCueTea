//
//  Slicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Slicer.hpp"

#include "VectorOps.h"
#include "Benchtools.h"
#include "MathUtils.h"

using namespace jsa;

void Slicer::setSize(Eigen::Index newBlockSize, Eigen::Index newHopSize) {
    blockSize = newBlockSize;
    hopSize = newHopSize;
    overlapSize = blockSize - hopSize;
    bufferSize = nextPow2(blockSize + 1);
    buffer.resize(2 * bufferSize);
    buffer.setZero();
    wp = 0;
    rp = constrain(bufferSize - overlapSize, bufferSize);
}

void Slicer::pushSample(double sample) {
    RealTimeChecker rt;
    wp = constrain(wp, bufferSize);
    buffer(wp) = buffer(wp + bufferSize) = sample;
    wp++;
}

bool Slicer::hasBlock() {
    RealTimeChecker rt;
    return (wp % hopSize) == 0;
}

Eigen::Map<const Eigen::ArrayXd> Slicer::getBlock() {
    RealTimeChecker rt;
    rp = constrain(rp, bufferSize);
    auto segment = buffer.segment(rp, blockSize);
    rp += hopSize;
    return Eigen::Map<const Eigen::ArrayXd>(segment.data(), segment.size());
}
