//
//  RingBuffer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <armadillo>

#include "VectorOps.h"

namespace jsa {

template <typename T>
class DoubleBuffer {
public:
    void fill(T& value) {
        push(value);
        push(value);
    }
    void push(T& value) {
        advance();
        buffer[state] = value;
    }
    
    void advance() {
        state = !state;
    }
    
    T& next() {
        advance();
        return buffer[state];
    }
    
    const T& current() const { return buffer[state]; }
    const T& last() const { return buffer[!state]; }
    T& last() { return buffer[!state]; }
    T& current() { return buffer[state]; }
    
private:
    std::array<T, 2> buffer;
    bool state = false;
};

}
