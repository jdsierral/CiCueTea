function Xcq = nsgfCQT(x, s)
X = fft(x, s.nSamps) ./ s.nSamps;
    
if s.type == "full"
    Xcq = 2 * s.nSamps * ifft(X .* s.g);
elseif s.type == "sparse"
    Xcq = cell(s.nBands, 1);
    for k = 1:s.nBands
        nCoefs = length(s.idxs{k});
        Xi = ifft(X(s.idxs{k}) .* s.g{k});
        Xi = 2 * Xi .* nCoefs .* s.phase{k};
        Xcq{k} = Xi;
    end
end
end