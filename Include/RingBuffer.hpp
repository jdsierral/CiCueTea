//
//  RingBuffer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <Eigen/Core>

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
    T& current() { return buffer[state]; }
    
private:
    std::array<T, 2> buffer;
    bool state = false;
};

class MatBuffer {
    void setSize(Eigen::Index newSize, const Eigen::ArrayXXd& zero) {
        newSize = nextPow2(newSize);
        buffer.resize(newSize, zero);
    }
    
    void pushMat(const Eigen::ArrayXd& mat) {
        wp = constrain(wp, buffer.size());
        buffer[wp] = mat;
        wp++;
    }
    
    Eigen::Map<const Eigen::ArrayXXd> getMat() {
        rp = constrain(rp, buffer.size());
        auto block = buffer[rp];
        Eigen::Map<const Eigen::ArrayXXd> mat(block.data(), block.rows(), block.cols());
        rp++;
        return mat;
    }

private:
    std::vector<Eigen::ArrayXXd> buffer;
    size_t rp;
    size_t wp;
};

class BlockBuffer {
public:
    void setSize(Eigen::Index newSize, const Eigen::ArrayXd& zero) {
        newSize = nextPow2(newSize);
        buffer.resize(zero.size(), newSize);
        for (Eigen::Index n = 0; n < buffer.cols(); n++) {
            buffer.col(n) = zero;
        }
        wp = 0;
        rp = 0;
    }
    
    void pushBlock(const Eigen::ArrayXd& block) {
        wp = constrain(wp, buffer.size());
        buffer.col(wp) = block;
        wp++;
    }
    
    Eigen::Map<const Eigen::ArrayXd> getBlock() {
        rp = constrain(rp, buffer.size());
        auto segment = buffer.col(rp);
        Eigen::Map<const Eigen::ArrayXd> block(segment.data(), segment.size());
        rp++;
        return block;
    }

private:
    Eigen::ArrayXXd buffer;
    size_t rp;
    size_t wp;
};

class RingBuffer {
public:
    void setSize(Eigen::Index newSize) {
        newSize = nextPow2(newSize);
        buffer.resize(newSize);
        wp = 0;
        rp = 0;
    }
    
    void pushSample(double sample) {
        wp = constrain(wp, buffer.size());
        buffer(wp) = sample;
        wp++;
    }
    
    double getSample() {
        rp = constrain(wp, buffer.size());
        double sample = buffer(rp);
        rp++;
        return sample;
    }
    
private:
    Eigen::ArrayXd buffer;
    size_t rp;
    size_t wp;
};

}
