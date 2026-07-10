//
//  Benchtools.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/22/25.
//

#pragma once

#include <chrono>
#include <iostream>

namespace jsa::test {

// Scope timer. By default prints the elapsed time on destruction; construct
// with Timer(false) to keep it silent and read the value via get() instead.
class Timer
{
  public:
    explicit Timer(bool printOnExit = true) :
        verbose(printOnExit),
        startTime(std::chrono::high_resolution_clock::now()) {}
    ~Timer()
    {
        if (verbose)
            std::cout << "Timer Result: " << get() << " ms" << std::endl;
    }

    /// Elapsed time since construction in fractional milliseconds.
    double get()
    {
        using namespace std::chrono;
        auto curTime = high_resolution_clock::now();
        return duration<double, std::milli>(curTime - startTime).count();
    }

  private:
    bool                                           verbose;
    std::chrono::high_resolution_clock::time_point startTime;
};

} // namespace jsa::test
