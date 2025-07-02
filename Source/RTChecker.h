//
//  RTChecker.h
//  CQTDSP
//
//  Created by Juan Sierra on 6/15/25.
//

/**
 * @file RTChecker.h
 * @brief RIAA based check over mallocs only enabled in DEBUG mode
 * @author Juan Sierra
 * @date 4/8/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>

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

}
