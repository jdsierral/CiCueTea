
% SPECTRALSPLICER  Reconstruct a spectrogram from overlapping time blocks
%   X = spectralSplicer(XBlock, overlapSize)
%
%   Reconstructs the original spectrogram from overlapping time blocks.
%   Each block is spliced together along the time axis for each band.
%
%   INPUTS:
%     XBlock     : 3D array of size [blockSize, nBands, nBlocks],
%                  as produced by spectralSlicer
%     overlapSize: Number of samples overlapped between blocks
%
%   OUTPUT:
%     X : Reconstructed spectrogram (matrix: [nSamps, nBands])
%
%   See also: spectralSlicer, slicer, splicer

function X = spectralSplicer(XBlock, overlapSize)
    [blockSize, nBands, nBlocks] = size(XBlock);
    hopSize = blockSize - overlapSize; % Step size between blocks
    nSamps = hopSize * nBlocks + overlapSize; % Total number of samples
    % Permute to [nBands, blockSize, nBlocks] for easier band-wise processing
    XBlock = permute(XBlock, [2, 1, 3]);

    X = zeros(nSamps, nBands);
    for k = 1:nBands
        % Splice blocks for each band along the time axis
        X(:,k) = splicer(squeeze(XBlock(k,:,:)), overlapSize);
    end
end
