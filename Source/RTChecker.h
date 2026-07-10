//
//  RTChecker.h
//  CQTDSP
//
//  Created by Juan Sierra on 6/15/25.
//

/**
 * @file RTChecker.h
 * @brief RAII guard asserting that no Eigen heap activity (allocation or
 *        deallocation) happens inside real-time critical sections.
 * @author Juan Sierra
 * @date 4/8/25
 * @copyright MIT License
 *
 * Armed by the REALTIME_CHECKS compile definition (CMake option of the same
 * name) on top of EIGEN_RUNTIME_NO_MALLOC; the checks fire through
 * eigen_assert, so they are active only in builds where assert is live
 * (NDEBUG not defined) and cost nothing in release builds.
 *
 * On Eigen >= 5 the guard also forbids deallocation explicitly: a free in
 * the audio callback is as non-deterministic as a malloc. Note that Eigen
 * 5.0.0/5.0.1 ship a typo (check_that_free_is_allowed tests the malloc
 * flag, fixed on upstream master), so guarded frees assert there regardless
 * — forbidding them deliberately keeps the guard's meaning identical across
 * Eigen versions.
 */

#pragma once

#include <Eigen/Core>

#ifdef REALTIME_CHECKS
#    if EIGEN_VERSION_AT_LEAST(5, 0, 0)
#        define ENTERING_REAL_TIME_CRITICAL_CODE           \
            Eigen::internal::set_is_malloc_allowed(false); \
            Eigen::internal::set_is_free_allowed(false);
#        define EXITING_REAL_TIME_CRITICAL_CODE          \
            Eigen::internal::set_is_free_allowed(true);  \
            Eigen::internal::set_is_malloc_allowed(true);
#    else
#        define ENTERING_REAL_TIME_CRITICAL_CODE \
            Eigen::internal::set_is_malloc_allowed(false);
#        define EXITING_REAL_TIME_CRITICAL_CODE \
            Eigen::internal::set_is_malloc_allowed(true);
#    endif
#else
#    define ENTERING_REAL_TIME_CRITICAL_CODE
#    define EXITING_REAL_TIME_CRITICAL_CODE
#endif

namespace jsa::cicuetea {

class RealTimeChecker
{
  public:
    RealTimeChecker() { ENTERING_REAL_TIME_CRITICAL_CODE }

    ~RealTimeChecker() { EXITING_REAL_TIME_CRITICAL_CODE }
};

} // namespace jsa::cicuetea
