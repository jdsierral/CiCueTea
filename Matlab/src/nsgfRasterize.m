function Xr = nsgfRasterize(X, s)
Xr = zeros(s.nSamps, s.nBands);
for k = 1:s.nBands
    Xr(:,k) = interpft(X{k} .* conj(s.phase{k}), s.nSamps);
end
end