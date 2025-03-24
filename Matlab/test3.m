clear
clc

fs = 48000;
N = 2^20;
ppo = 3;
fMin = 100;
fMax = 10000;
fRef = 1500;

blockSize = 2^12;
hopSize = blockSize / 2;
overlapSize = blockSize - hopSize;
win = hann(blockSize, "periodic");
win = sqrt(win);
% win = win * sqrt(2)/4;

sShort = nsgfCQTInit("full", fs, blockSize, ppo, fMin, fMax, fRef);
sLong = nsgfCQTInit("full", fs, N, ppo, fMin, fMax, fRef);

t = (0:N-1)' / fs;
x = sin(2 * pi * fRef * t);
fScale = 1.2;
x = chirp(t, fMin * fScale, t(end), fMax / fScale, "logarithmic");
x = x .* kaiser(N, 20);

xBuf = buffering(x, blockSize, overlapSize);
Xcq = colfun(@(xi) nsgfCQT(xi, sShort), xBuf .* win, 'uni', 0);
Xcq = cellfun(@(x) x .* win.', Xcq, 'uni', 0);
X = cellDeBuffering(Xcq, blockSize, overlapSize);
Ycq = cellBuffering(X, blockSize, overlapSize);
Ycq = cellfun(@(x) x .* win.', Ycq, 'uni', 0);
yBuf = cellfun(@(Xi) nsgfICQT(Xi, sShort), Ycq, 'uni', 0);
yBuf = cell2mat(yBuf');
y = deBuffering(yBuf .* win, blockSize, overlapSize);
% y = nsgfICQT(X, sLong);

rms(x - y)

figure(1)
clf();
imagesc(domainColor(X))
set(gca, 'YDir', 'normal');

figure(2)
clf();
subplot 311
plot(x)

subplot 312
plot(y)

subplot 313
plot(y - x)


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