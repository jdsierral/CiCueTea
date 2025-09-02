function XBuf = SpecBuffering(X, blockSize, overlapSize, nBlocks)

if iscell(X)
    overlapRatio = overlapSize / blockSize;
    nBands = size(X, 1);
    XBuf = cell(nBlocks, 1); 
    for k = 1:nBands
        nSamps = length(X{k});
        blockSize = nSamps / (nBlocks * (1 - overlapRatio) + overlapRatio);
        overlapSize = blockSize * overlapRatio;
        XBufI = buffering(X{k}, blockSize, overlapSize);
        XBuf(k,:) = mat2cell(XBufI.', ones(nBlocks, 1), blockSize)';
    end
else
    [nBands, nSamps] = size(X);
    hopSize = blockSize - overlapSize;
    nBlocks = ceil((nSamps - overlapSize) / hopSize);

    XBuf = zeros(nBands, blockSize, nBlocks);
    for k = 1:nBands
        XBuf(k,:,:) = buffering(X(k,:), blockSize, overlapSize);
    end
end

end