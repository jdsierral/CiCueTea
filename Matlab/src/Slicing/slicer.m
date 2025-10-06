
% SLICER  Slice a signal into overlapping blocks
%   y = slicer(x, blockSize, overlapSize)
%
%   Splits the input signal x into overlapping blocks of size blockSize,
%   with overlap of overlapSize samples between consecutive blocks.
%
%   INPUTS:
%     x          : Input signal (vector)
%     blockSize  : Size of each block
%     overlapSize: Number of samples to overlap between blocks
%
%   OUTPUT:
%     y : Matrix of size [blockSize, nBlocks], each column is a block
%
%   See also: spectralSlicer, spectralSplicer, splicer

function y = slicer(x, blockSize, overlapSize)
    nSamps = length(x);
    hopSize = blockSize - overlapSize; % Step size between blocks
    % Compute indices for each block
    idx = (0:hopSize:nSamps-blockSize) + (1:blockSize)';
    % Extract blocks using computed indices
    y = x(idx);
end
