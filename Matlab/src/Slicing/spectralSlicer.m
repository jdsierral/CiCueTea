
% SPECTRALSLICER  Slice a spectrogram into overlapping time blocks
%   XBlock = spectralSlicer(X, blockSize, overlapSize)
%
%   Splits the input spectrogram X into overlapping time blocks of size blockSize,
%   with overlap of overlapSize samples between consecutive blocks, for each band.
%
%   INPUTS:
%     X          : Input spectrogram (matrix: [nSamps, nBands])
%     blockSize  : Size of each time block
%     overlapSize: Number of samples to overlap between blocks
%
%   OUTPUT:
%     XBlock : 3D array of size [blockSize, nBands, nBlocks],
%              where each slice XBlock(:,:,i) is a time block
%
%   See also: slicer, spectralSplicer, splicer

function XBlock = spectralSlicer(X, blockSize, overlapSize)
    [nSamps, nBands] = size(X);
    hopSize = blockSize - overlapSize; % Step size between blocks
    nBlocks = ceil((nSamps - overlapSize) / hopSize); % Number of blocks
    XBlock = zeros(blockSize, nBands, nBlocks);
    for k = 1:nBands
        % Slice each band (column) into overlapping blocks
        XBlock(:,k,:) = slicer(X(:,k), blockSize, overlapSize);
    end
end