//
//  SignalUtils.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

/**
 * @file SignalUtils.h
 * @brief Provides an implementation of a Slicer of continuous data
 * @author Juan Sierra
 * @date 3/23/25
 * @copyright MIT License
 */

#pragma once

#include <numbers>
#include <Eigen/Core>

#include "MathUtils.h"

namespace jsa {

inline Eigen::ArrayXd hann(Eigen::Index N)
{
    assert(N > 0);
    Eigen::ArrayXd n   = regspace(N);
    Eigen::ArrayXd win = (std::numbers::pi * n / N).sin().square();
    return win;
}

} // namespace jsa
