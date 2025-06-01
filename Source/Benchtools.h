//
//  Benchtools.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/22/25.
//

#pragma once

#include <iostream>
#include <Eigen/Core>

#define REALTIME_CHECKS

#ifdef REALTIME_CHECKS
#define ENTERING_REAL_TIME_CRITICAL_CODE                                       \
  Eigen::internal::set_is_malloc_allowed(false);
#define EXITING_REAL_TIME_CRITICAL_CODE                                        \
  Eigen::internal::set_is_malloc_allowed(true);
#else
#define ENTERING_REAL_TIME_CRITICAL_CODE
#define EXITING_REAL_TIME_CRITICAL_CODE
#endif

namespace jsa {

class RealTimeChecker {
public:
  RealTimeChecker() { ENTERING_REAL_TIME_CRITICAL_CODE }

  ~RealTimeChecker() { EXITING_REAL_TIME_CRITICAL_CODE }
};

class Timer {
public:
  Timer() : startTime(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    double t = get();
    std::cout << "Timer Result: " << t << " ms" << std::endl;
  }

  double get() {
    auto curTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          curTime - startTime)
                          .count();
    return duration;
  }

private:
  std::chrono::high_resolution_clock::time_point startTime;
};

} // namespace jsa
