function y = slicer(x, blockSize, overlapSize)
    nSamps = length(x);
    hopSize = blockSize - overlapSize;
    idx = (0:hopSize:nSamps-blockSize) + (1:blockSize)';
    y = x(idx);
end
