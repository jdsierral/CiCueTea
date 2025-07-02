//
//  DoubleBuffer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

/**
 * @file DoubleBuffer.h
 * @brief Provides an implementation of a 2 step buffer to handle current and
 * last elements
 * @author Juan Sierra
 * @date 3/23/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>

namespace jsa {

/**
 * @class DoubleBuffer
 * @brief A double-buffer implementation for storing and switching between two values.
 * @tparam T The type of the values stored in the buffer.
 */
template <typename T>
class DoubleBuffer {
public:
    /**
     * @brief Fills both buffers with the given value.
     * @param value The value to fill the buffers with.
     */
    void fill(T& value) {
        push(value);
        push(value);
    }
    
    /**
     * @brief Pushes a value into the buffer and advances the state.
     * @param value The value to push into the buffer.
     */
    void push(T& value) {
        advance();
        buffer[state] = value;
    }
    
    /**
     * @brief Advances the buffer state to the next buffer.
     */
    void advance() {
        state = !state;
    }
    
    /**
     * @brief Advances the buffer state and retrieves the next buffer value.
     * @return A reference to the next buffer value.
     */
    T& next() {
        advance();
        return buffer[state];
    }
    
    /**
     * @brief Retrieves the current buffer value.
     * @return A const reference to the current buffer value.
     */
    const T& current() const { return buffer[state]; }
    
    /**
     * @brief Retrieves the last buffer value.
     * @return A const reference to the last buffer value.
     */
    const T& last() const { return buffer[!state]; }
    
    /**
     * @brief Retrieves the last buffer value.
     * @return A reference to the last buffer value.
     */
    T& last() { return buffer[!state]; }
    
    /**
     * @brief Retrieves the current buffer value.
     * @return A reference to the current buffer value.
     */
    T& current() { return buffer[state]; }
    
private:
    std::array<T, 2> buffer; ///< The double buffer storage.
    bool state = false;      ///< The current state of the buffer (true or false).
};



}
