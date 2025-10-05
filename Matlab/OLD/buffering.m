function y = buffering(x, blockSize, overlapSize)
    hopSize = blockSize - overlapSize;
    N = length(x);
    idx = (0:hopSize:N-blockSize) + (1:blockSize)';
    y = x(idx);
end