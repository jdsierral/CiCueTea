//
//  Splicer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <Eigen/Core>

namespace jsa {

class Splicer {
public:
    void setSize(Eigen::Index newBlockSize, Eigen::Index newHopSize);
    void pushBlock(const Eigen::ArrayXd& block);
    double getSample();

private:
    Eigen::ArrayXd buffer;
    Eigen::Index bufferSize;
    Eigen::Index blockSize;
    Eigen::Index overlapSize;
    Eigen::Index hopSize;
    size_t wp = 0;
    size_t rp = 0;
};

}
