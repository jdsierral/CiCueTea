//
//  Benchtools.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/22/25.
//

#pragma once

#include <Eigen/Core>
#include <iostream>

namespace jsa {

class Timer {
public:
    Timer() : startTime(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        double t = get();
        std::cout << "Timer Result: " << t << " ms" << std::endl;
    }
    
    double get() {
        using namespace std::chrono;
        auto curTime = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(curTime - startTime).count();
        return duration;
    }
    
private:
    std::chrono::high_resolution_clock::time_point startTime;
};

} // namespace jsa
