//
//  SignalUtils.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

/**
 * @file SignalUtils.h
 * @brief Signal utilities: window functions.
 * @author Juan Sierra
 * @date 3/23/25
 * @copyright MIT License
 */

#pragma once

#include <cassert>
#include <numbers>

#include <Eigen/Core>

#include "MathUtils.h"

namespace jsa::cicuetea {

/**
 * @brief Periodic Hann window of length N: sin²(πn/N).
 *
 * Periodic (DFT-even) form, so sqrt-Hann analysis/synthesis pairs at 50%
 * overlap satisfy the WOLA condition exactly.
 */
inline Eigen::ArrayXd hann(Eigen::Index N)
{
    assert(N > 0);
    Eigen::ArrayXd n   = regspace(N);
    Eigen::ArrayXd win = (std::numbers::pi * n / N).sin().square();
    return win;
}

} // namespace jsa::cicuetea
