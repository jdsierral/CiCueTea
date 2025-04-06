//
//  Slicer.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#include "Slicer.hpp"

#include "VectorOps.h"
#include "Benchtools.h"

using namespace jsa;
using namespace arma;

void Slicer::setSize(uword newBlockSize, uword newHopSize) {
    blockSize = newBlockSize;
    hopSize = newHopSize;
    overlapSize = blockSize - hopSize;
    bufferSize = nextPow2(blockSize + 1);
    buffer.resize(2 * bufferSize);
    buffer.zeros();
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

const vec Slicer::getBlock() {
    RealTimeChecker rt;
    rp = constrain(rp, bufferSize);
    vec segment(buffer.memptr() + rp, blockSize, false, false);
    rp += hopSize;
    return segment;
}
