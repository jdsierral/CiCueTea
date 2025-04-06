//
//  Slicer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <armadillo>

namespace jsa {

class Slicer {
public:
    void setSize(arma::uword newBlockSize, arma::uword newHopSize);
    void pushSample(double sample);
    bool hasBlock();
    const arma::vec getBlock();
    
    arma::uword getBlockSize() const { return blockSize; }
    arma::uword getOverlapSize() const { return overlapSize; }
    arma::uword getHopSize() const { return hopSize; }
    arma::uword getBufferSize() const { return bufferSize; }
    
private:
    arma::vec buffer;
    arma::uword bufferSize;
    arma::uword blockSize;
    arma::uword overlapSize;
    arma::uword hopSize;
    size_t wp = 0;
    size_t rp = 0;
};

}
