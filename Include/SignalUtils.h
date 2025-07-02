//
//  SignalUtils.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

#pragma once

#include <Eigen/Core>

#include "MathUtils.h"

namespace jsa {

inline Eigen::ArrayXd hann(Eigen::Index N) {
    assert(N > 0);
    Eigen::ArrayXd n = regspace(N);
    Eigen::ArrayXd win = (M_PI * n / N).sin().square();
    return win;
}

}
