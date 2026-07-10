//
//  EmptyCQTProc.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//
//  Identity test doubles: one subclass per processor variant with an empty
//  processBlock(), so tests exercise the full analysis/synthesis chain with
//  no coefficient manipulation — output must reconstruct the input.
//

#pragma once

#include <Eigen/Core>

#include <CQT.hpp>
#include <CQTProcessor.hpp>

class CqtDense : public jsa::cicuetea::CqtDenseProcessor
{
  public:
    using jsa::cicuetea::CqtDenseProcessor::CqtDenseProcessor;
    void processBlock(Eigen::ArrayXXcd& /*block*/) override {}
};

class SliCqtDense : public jsa::cicuetea::SlidingCqtDenseProcessor
{
  public:
    using jsa::cicuetea::SlidingCqtDenseProcessor::SlidingCqtDenseProcessor;
    void processBlock(Eigen::ArrayXXcd& /*block*/) override {};
};

class CqtSparse : public jsa::cicuetea::CqtSparseProcessor
{
  public:
    using jsa::cicuetea::CqtSparseProcessor::CqtSparseProcessor;
    void processBlock(jsa::cicuetea::NsgfCqtSparse::Coefs& /*block*/) override {}
};

class SliCqtSparse : public jsa::cicuetea::SlidingCqtSparseProcessor
{
  public:
    using jsa::cicuetea::SlidingCqtSparseProcessor::SlidingCqtSparseProcessor;
    void processBlock(jsa::cicuetea::NsgfCqtSparse::Coefs& /*block*/) override {};
};
