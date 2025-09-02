function x = nsgfICQT(Xcq, s)
if s.type == "full"
    X = 1./ (2 * s.nSamps) * sum(fft(Xcq) .* s.gDual, 2);
elseif s.type == "sparse"
    X = zeros(s.nFreqs, 1);
    for k = 1:s.nBands
        nCoefs = length(s.idxs{k});
        Xi = Xcq{k};
        Xi = Xi .* conj(s.phase{k}) / (2 * nCoefs);
        Xi = fft(Xi) .* s.gDual{k};
        X(s.idxs{k}) = X(s.idxs{k}) + Xi;
    end
end
x = ifft(X, s.nSamps, "symmetric") .* s.nSamps;
end