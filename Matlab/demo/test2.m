clear
clc

addpath(genpath("../src"));

fs = 48000;
nSamps = 2^18;
frac = 1;
fMin = 100;
fMax = 10000;

t = (0:nSamps-1)'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
w = kaiser(nSamps, 20);
x = x .* w;

s1 = nsgfCQTInit("full", fs, nSamps, frac);
s2 = nsgfCQTInit("sparse", fs, nSamps, frac);

X1 = nsgfCQT(x, s1);
X2 = nsgfCQT(x, s2);
X2 = nsgfRasterize(X2, s2);

% Get the time domain atoms of full version by inverse real fft of frequency 
% domain atoms
g1 = s1.g(1:end/2+1,:);
h1 = ifftshift(irfft(g1, s1.nSamps), 1);
h1 = h1 ./ max(abs(h1));

% Get the time domain atoms of sparse by inverse real fft of frequency 
% domain atoms after inserting them in a buffer of zeros at the righ offset for 
% each frame
h2 = zeros(size(h1));
for k = 1:s2.nBands
    g_i = zeros(s2.nSamps, 1);
    g_i(s2.idxs{k}) = s2.g{k};
    g_i = g_i(1:s2.nSamps/2+1);
    h2(:,k) = ifftshift(irfft(g_i, s2.nSamps));
end

h2 = h2 ./ max(abs(h2));

tax = s1.tax;
bax = s1.bax;

figure(1)
clf()
range = [-100, 0];
subplot 311
imagesc(tax, bax, dB(X1).');
colormap jet
colorbar
set(gca, "CLim", range)
title("Full Transform");
set(gca, "YDir", "normal");

subplot 312
imagesc(tax, bax, dB(X2).');
colormap jet
colorbar
set(gca, "CLim", range)
title("Rasterized Sparse Transform");
set(gca, "YDir", "normal");

subplot 313
imagesc(tax, bax, dB(X1-X2).');
colormap jet
colorbar
set(gca, "CLim", range)
title("Difference");
set(gca, "YDir", "normal");

figure(2)
clf();
subplot 211
semilogx(bax, pow2db(mean(abs(X1).^2)), "LineWidth", 2)
hold on
semilogx(bax, pow2db(mean(abs(X2).^2)), "LineWidth", 2)
xlabel("Frequency");
ylabel("Amplitude (dB)");
title("AutoSpectrum");

subplot 212
plot(tax, pow2db(mean(abs(X1).^2, 2)), "LineWidth", 2)
hold on
plot(tax, pow2db(mean(abs(X2).^2, 2)), "LineWidth", 2)
xlabel("Time")
ylabel("Amplitude (dB)")
title("Energy across time")

figure(3)
clf();
subplot 121
plot(tax, h1 + 2 * (1:s1.nBands), "LineWidth", 2);
xlim(0.02 * [-1, 1])
xlabel("Time")
yticks([]);
title("Full time-domain Atoms")

subplot 122
plot(tax, h2 + 2 * (1:s2.nBands), "LineWidth", 2);
xlim(0.02 * [-1, 1])
title("Sparse time-domain Atoms")
