% SPECBUFFERING  Buffer spectral data into overlapping blocks
%   XBuf = SpecBuffering(X, blockSize, overlapSize, nBlocks)
%
%   Buffers the input spectral data X into overlapping blocks for further
%   processing. Handles both cell array (per-band) and matrix input.
%
%   INPUTS:
%     X          : Input spectral data (cell array or matrix)
%     blockSize  : Size of each block
%     overlapSize: Number of samples to overlap between blocks
%     nBlocks    : Number of blocks (used for cell input)
%
%   OUTPUT:
%     XBuf : Buffered spectral data (cell array or 3D matrix)
%
%   See also: SpecDebuffering

function XBuf = SpecBuffering(X, blockSize, overlapSize, nBlocks)

    if iscell(X)
        % Cell input: process each band separately
        overlapRatio = overlapSize / blockSize;
        nBands = size(X, 1);
        XBuf = cell(nBlocks, 1); 
        for k = 1:nBands
            nSamps = length(X{k});
            % Recompute block and overlap size for exact fit
            blockSize = nSamps / (nBlocks * (1 - overlapRatio) + overlapRatio);
            overlapSize = blockSize * overlapRatio;
            % Buffer the band into overlapping blocks
            XBufI = buffering(X{k}, blockSize, overlapSize);
            % Store as cell array, one cell per block
            XBuf(k,:) = mat2cell(XBufI.', ones(nBlocks, 1), blockSize)';
        end
    else
        % Matrix input: buffer all bands at once
        [nBands, nSamps] = size(X);
        hopSize = blockSize - overlapSize;
        nBlocks = ceil((nSamps - overlapSize) / hopSize);

        XBuf = zeros(nBands, blockSize, nBlocks);
        for k = 1:nBands
            % Buffer each band into overlapping blocks
            XBuf(k,:,:) = buffering(X(k,:), blockSize, overlapSize);
        end
    end

end