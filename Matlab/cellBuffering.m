function xCell = cellBuffering(xMat, blockSize, overlapSize)
    [nBands, nSamps] = size(xMat);
    hopSize = blockSize - overlapSize;
    nBlocks = (nSamps - overlapSize) / hopSize;
    xCell = cell(nBlocks, 1);
    for i = 1:nBlocks
        p = (i - 1) * hopSize;
        xCell{i} = xMat(:,p+1:p+blockSize);
    end
end