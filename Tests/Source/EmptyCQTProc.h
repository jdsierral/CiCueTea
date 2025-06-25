//
//  EmptyCQTProc.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

#pragma once

class CqtFull : public jsa::CqtFullProcessor {
public:
  using jsa::CqtFullProcessor::CqtFullProcessor;
  void processBlock(Eigen::ArrayXXcd &block) override {}
};

class SliCQTFull : public jsa::SlidingCQTFullProcessor {
public:
  using jsa::SlidingCQTFullProcessor::SlidingCQTFullProcessor;
  void processBlock(Eigen::ArrayXXcd &block) override {};
};

class CqtSparse : public jsa::CqtSparseProcessor {
public:
  using jsa::CqtSparseProcessor::CqtSparseProcessor;
  void processBlock(jsa::NsgfCqtSparse::Coefs &block) override {}
};

class SliCQTSparse : public jsa::SlidingCqtSparseProcessor {
public:
  using jsa::SlidingCqtSparseProcessor::SlidingCqtSparseProcessor;
  void processBlock(jsa::NsgfCqtSparse::Coefs &block) override {};
};
