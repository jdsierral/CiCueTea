function y = deBuffering(x, blockSize, overlapSize)
hopSize = blockSize - overlapSize;
nBlocks = size(x, 2);
pos = 1;

N = hopSize * nBlocks + overlapSize;
y = zeros(N, 1);

for i = 1:nBlocks
    xi = x(:,i);
    y(pos:pos+blockSize-1) = y(pos:pos+blockSize-1) + xi;
    pos = pos + hopSize;
end

end