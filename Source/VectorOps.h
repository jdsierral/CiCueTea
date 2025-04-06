//
//  VectorOps.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <cassert>
#include <armadillo>

namespace jsa {

inline double square(double x) { return x * x; }

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

template <typename T> inline T nextPow2(T n) { return exp2(ceil(log2(n))); }

//

inline double rms(const arma::vec& x) {
    return std::sqrt(arma::mean(arma::square(x)));
}

//inline Eigen::ArrayXd regspace(Eigen::Index num) {
//    return Eigen::ArrayXd::LinSpaced(num, 0, num-1);
//}
//
//inline Eigen::ArrayXd regspace(Eigen::Index low, Eigen::Index high) {
//    return Eigen::ArrayXd::LinSpaced(high-low+1, low, high);
//}

inline arma::vec hann(arma::uword N) {
    assert(N > 0);
    arma::vec n = arma::regspace(0, N-1);
    arma::vec win = arma::square(sin(arma::datum::pi * n / N));
    return win;
}

inline arma::vec logChirp(const arma::vec& t, double f0, double f1) {
    double t1 = t(t.size() - 1);
    arma::vec r = t;
    r.for_each([&](double& val){ val = pow(f1/f0, val/t1); });
//    Eigen::ArrayXd r = t.unaryExpr([&](double val){ return pow(f1/f0, val/t1); });
    arma::vec phase = (t1 / std::log(f1/f0)*f0)*(r - 1.0);
//    Eigen::ArrayXd phase = (t1/std::log(f1/f0)*f0)*(r - 1.0);
    return cos(phase * 2 * arma::datum::pi);  // Chirp signal
}

inline std::vector<std::vector<double>> toStdVector(const arma::mat& arr) {
    std::vector<std::vector<double>> vec(arr.n_rows, std::vector<double>(arr.n_cols));

    for (arma::uword i = 0; i < arr.n_rows; ++i) {
        for (arma::uword j = 0; j < arr.n_cols; ++j) {
            vec[i][j] = arr(i, j);  // Row-major filling
        }
    }
    return vec;
}

}
