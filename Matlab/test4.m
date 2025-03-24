clear
clc

fs = 48000;
N = 2^20;
ppo = 3;
fMin = 100;
fMax = 10000;
fRef = 1500;

blockSize = 2^12;
hopSize = blockSize / 8;
overlapSize = blockSize - hopSize;
win = sqrt(hann(blockSize, "periodic"));
win = win * 1/2;

sShort = nsgfCQTInit("full", fs, blockSize, ppo, fMin, fMax, fRef);
sLong = nsgfCQTInit("full", fs, N, ppo, fMin, fMax, fRef);

t = (0:N-1)' / fs;
x = sin(2 * pi * fRef * t);
fScale = 1.2;
x = chirp(t, fMin * fScale, t(end), fMax / fScale, "logarithmic");
x = x .* kaiser(N, 20);

xBuf = buffering(x, blockSize, overlapSize);
Xcq = colfun(@(xi) nsgfCQT(xi, sShort), xBuf .* win, 'uni', 0);

nBands = sShort.nBands;
nBlocks = length(Xcq);
% X = zeros(nBands, nBlocks * hopSize + overlapSize);
X = cell(nBands, 1);

for i = 1:nBlocks
    for k = 1:nBands
        p = (i-1) * hopSize;
        X(:, p+1:p+blockSize) = X(:, p+1:p+blockSize) + Xcq{i} .* win.';
    end
end

y = nsgfICQT(X, sLong);

rms(x - y)

figure(1)
clf();
imagesc(domainColor(X))
set(gca, 'YDir', 'normal');

figure(2)
clf();
subplot 211
plot(x)

subplot 212
plot(y)


function cellDeBuffering()
end

function cellBuffering()
end