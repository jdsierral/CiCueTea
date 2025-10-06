
% SPLICER  Reconstruct a signal from overlapping blocks
%   y = splicer(x, overlapSize)
%
%   Reconstructs the original signal by overlap-adding the columns of x,
%   which are assumed to be overlapping blocks with overlapSize samples.
%
%   INPUTS:
%     x          : Matrix of size [blockSize, nBlocks], each column is a block
%     overlapSize: Number of samples overlapped between blocks
%
%   OUTPUT:
%     y : Reconstructed signal (vector)
%
%   See also: slicer, spectralSlicer, spectralSplicer

function y = splicer(x, overlapSize)
    [blockSize, nBlocks] = size(x);
    hopSize = blockSize - overlapSize; % Step size between blocks
    pos = 1; % Starting position for each block

    nSamps = hopSize * nBlocks + overlapSize; % Total length of output
    y = zeros(nSamps, 1);

    for i = 1:nBlocks
        ii = pos:pos+blockSize-1; % Indices for current block
        y(ii) = y(ii) + x(:,i);   % Overlap-add current block
        pos = pos + hopSize;      % Move to next block position
    end
end