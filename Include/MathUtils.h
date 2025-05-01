//
//  MathUtils.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

#pragma once

#include <Eigen/Core>

inline constexpr uint32_t nextPow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline size_t constrain(size_t idx, size_t size) {
    return idx % size;
}

inline double square(double x) { return x * x; }


inline Eigen::ArrayXd regspace(Eigen::Index num) {
    return Eigen::ArrayXd::LinSpaced(num, 0, num-1);
}

inline Eigen::ArrayXd regspace(Eigen::Index low, Eigen::Index high) {
    return Eigen::ArrayXd::LinSpaced(high-low+1, low, high);
}

inline Eigen::ArrayXd logspace(double start, double end, Eigen::Index num) {
    return Eigen::ArrayXd::LinSpaced(num, log(start), log(end)).exp();
}
