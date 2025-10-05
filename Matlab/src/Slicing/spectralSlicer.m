function XBlock = spectralSlicer(X, blockSize, overlapSize)
    [nSamps, nBands] = size(X);
    hopSize = blockSize - overlapSize;
    nBlocks = ceil((nSamps - overlapSize) / hopSize);
    XBlock = zeros(blockSize, nBands, nBlocks);
    for k = 1:nBands
        XBlock(:,k,:) = slicer(X(:,k), blockSize, overlapSize);
    end
end