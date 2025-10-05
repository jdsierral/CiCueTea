% SPECDEBUFFERING  Reconstruct spectral data from buffered blocks
%   X = SpecDebuffering(XBuf, blockSize, overlapSize, nBlocks)
%
%   Reconstructs the original spectral data from overlapping buffered blocks.
%   Handles both cell array (per-band) and matrix input.
%
%   INPUTS:
%     XBuf       : Buffered spectral data (cell array or 3D matrix)
%     blockSize  : Size of each block
%     overlapSize: Number of samples overlapped between blocks
%     nBlocks    : Number of blocks (used for cell input)
%
%   OUTPUT:
%     X : Reconstructed spectral data (cell array or matrix)
%
%   See also: SpecBuffering

function X = SpecDebuffering(XBuf, blockSize, overlapSize, nBlocks)

    if iscell(XBuf)
        % Cell input: process each band separately
        overlapRatio = overlapSize / blockSize;
        nBands = size(XBuf{1}, 1);
        X = cell(nBands, 1);
        % For each band, concatenate blocks and reconstruct
        for k = 1:nBands
            % Extract k-th band from each block and concatenate
            Xi = cellfun(@(x) x{k}.', XBuf, 'uni', 0);
            Xi = [Xi{:}];
            blockSize_k = size(Xi, 1);
            overlapSize_k = blockSize_k * overlapRatio;
            % Reconstruct original band using deBuffering
            X{k} = deBuffering(Xi, blockSize_k, overlapSize_k).';
        end
    else
        % Matrix input: reconstruct all bands at once
        assert(size(XBuf, 2) == blockSize);
        [nBands, blockSize, nBlocks] = size(XBuf);
        hopSize = blockSize - overlapSize;
        nSamps = hopSize * nBlocks + overlapSize;
        
        X = zeros(nBands, nSamps);
        for k = 1:nBands
            % Reconstruct each band using deBuffering
            X(k,:) = deBuffering(squeeze(XBuf(k,:,:)), blockSize, overlapSize);
        end
    end

end