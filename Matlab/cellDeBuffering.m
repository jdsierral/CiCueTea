function x = cellDeBuffering(xCell, blockSize, overlapSize)
    nBlocks = length(xCell);
    hopSize = blockSize - overlapSize;
    nBands = size(xCell{1}, 1);
    x = zeros(nBands, nBlocks * hopSize + overlapSize);
    for i = 1:nBlocks
        p = (i - 1) * hopSize;
        x(:, p+1:p+blockSize) = x(:, p+1:p+blockSize) + xCell{i};
    end
end