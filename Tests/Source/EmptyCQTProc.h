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

class CqtDense : public jsa::CqtDenseProcessor
{
  public:
    using jsa::CqtDenseProcessor::CqtDenseProcessor;
    void processBlock(Eigen::ArrayXXcd& /*block*/) override {}
};

class SliCqtDense : public jsa::SlidingCqtDenseProcessor
{
  public:
    using jsa::SlidingCqtDenseProcessor::SlidingCqtDenseProcessor;
    void processBlock(Eigen::ArrayXXcd& /*block*/) override {};
};

class CqtSparse : public jsa::CqtSparseProcessor
{
  public:
    using jsa::CqtSparseProcessor::CqtSparseProcessor;
    void processBlock(jsa::NsgfCqtSparse::Coefs& /*block*/) override {}
};

class SliCqtSparse : public jsa::SlidingCqtSparseProcessor
{
  public:
    using jsa::SlidingCqtSparseProcessor::SlidingCqtSparseProcessor;
    void processBlock(jsa::NsgfCqtSparse::Coefs& /*block*/) override {};
};
