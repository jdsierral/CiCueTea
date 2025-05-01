//
//  EmptyCQTProc.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

#pragma once

class cqtFull : public jsa::CqtFullProcessor
{
public:
    using jsa::CqtFullProcessor::CqtFullProcessor;
    void processBlock(Eigen::ArrayXXcd& block) override {}
};

class sliCQTFull : public jsa::SlidingCQTFullProcessor
{
public:
    using jsa::SlidingCQTFullProcessor::SlidingCQTFullProcessor;
    void processBlock(Eigen::ArrayXXcd& block) override {};
};

class cqtSparse : public jsa::CqtSparseProcessor
{
public:
    using jsa::CqtSparseProcessor::CqtSparseProcessor;
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {}
};

class sliCQTSparse : public jsa::SlidingCqtSparseProcessor
{
public:
    using jsa::SlidingCqtSparseProcessor::SlidingCqtSparseProcessor;
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {};
};
