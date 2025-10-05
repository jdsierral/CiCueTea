function y = splicer(x, overlapSize)
    [blockSize, nBlocks] = size(x);
    hopSize = blockSize - overlapSize;
    pos = 1;

    N = hopSize * nBlocks + overlapSize;
    y = zeros(N, 1);

    for i = 1:nBlocks
        ii = pos:pos+blockSize-1;
        y(ii) = y(ii) + x(:,i);
        pos = pos + hopSize;
    end
end