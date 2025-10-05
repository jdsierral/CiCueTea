function X = spectralSplicer(XBlock, overlapSize)
    [blockSize, nBands, nBlocks] = size(XBlock);
    hopSize = blockSize - overlapSize;
    nSamps = hopSize * nBlocks + overlapSize;
    XBlock = permute(XBlock, [2, 1, 3]);

    X = zeros(nSamps, nBands);
    for k = 1:nBands
        X(:,k) = splicer(squeeze(XBlock(k,:,:)), overlapSize);
    end
end
