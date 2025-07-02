//
//  VectorOps.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <Eigen/Core>
#include <armadillo>

namespace jsa {

template <typename T> inline T nextPow2(T n) { return exp2(ceil(log2(n))); }

inline arma::vec eig2armaVec(Eigen::ArrayXd x) {
  return arma::vec(x.data(), x.size());
}

inline arma::cx_vec eig2armaVec(Eigen::ArrayXcd x) {
  return arma::cx_vec(x.data(), x.size());
}

inline arma::mat eig2armaMat(Eigen::ArrayXXd x) {
  return arma::mat(x.data(), x.rows(), x.cols());
}

inline arma::cx_mat eig2armaMat(Eigen::ArrayXXcd x) {
  return arma::cx_mat(x.data(), x.rows(), x.cols());
}

inline void saveCoefs(const std::vector<Eigen::ArrayXcd>& X, std::string name) {
    for (size_t i = 0; i < X.size(); i++) {
        std::string name_i = name + "_" + std::to_string(i+1) + ".csv";
        eig2armaVec(X[i]).save(arma::csv_name(name_i));
    }
}

inline double rms(const Eigen::ArrayXd &x) {
  return std::sqrt(x.square().mean());
}

inline Eigen::ArrayXd logChirp(const Eigen::ArrayXd &t, double f0, double f1) {
  double t1 = t(t.size() - 1);
  Eigen::ArrayXd r =
      t.unaryExpr([&](double val) { return pow(f1 / f0, val / t1); });
  Eigen::ArrayXd phase = (t1 / std::log(f1 / f0) * f0) * (r - 1.0);
  return (phase * 2 * M_PI).cos(); // Chirp signal
}

inline std::vector<std::vector<double>>
toStdVector(const Eigen::ArrayXXd &arr) {
  std::vector<std::vector<double>> vec(arr.rows(),
                                       std::vector<double>(arr.cols()));

  for (Eigen::Index i = 0; i < arr.rows(); ++i) {
    for (Eigen::Index j = 0; j < arr.cols(); ++j) {
      vec[i][j] = arr(i, j); // Row-major filling
    }
  }
  return vec;
}

template <typename Derived>
inline Derived cumsum(const Eigen::DenseBase<Derived> &input) {
  Derived result(input.size());
  if (input.size() == 0)
    return result;

  result(0) = input(0);
  for (int i = 1; i < input.size(); ++i) {
    result(i) = result(i - 1) + input(i);
  }
  return result;
}

} // namespace jsa
