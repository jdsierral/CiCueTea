% NSGRASTERIZE  Rasterize sparse CQT coefficients into a matrix
%   Xr = nsgfRasterize(X, s)
%
%   Converts a cell array of sparse Constant-Q Transform (CQT) coefficients
%   into the dense matrix representation, exactly: the per-band phase and
%   scaling are undone, the band's span spectrum is recovered, and it is
%   embedded at its true bins. The result equals the dense transform's
%   columns up to the sparsity threshold.
%
%   Note that naive bandlimited interpolation (interpft) is NOT correct
%   here: it presumes baseband input, but the coefficients carry the
%   (aliased) carrier, and the band's support typically straddles the wrap
%   point of the decimated spectrum — that produced large magnitude errors
%   between the decimated instants.
%
%   INPUTS:
%     X : Cell array of CQT coefficients (one cell per band)
%     s : Structure with transform parameters (see nsgfCQTInit)
%
%   OUTPUT:
%     Xr : Matrix of size [s.nSamps, s.nBands], equal to the dense transform
%
%   See also: nsgfCQT, nsgfICQT, nsgfCQTInit

function Xr = nsgfRasterize(X, s)
    Xr = complex(zeros(s.nSamps, s.nBands));
    for k = 1:s.nBands
        nCoefs = length(s.idxs{k});
        % Undo the phase modulation and forward scaling, recover the span
        % spectrum, and embed it at the band's true frequency bins
        Xi = X{k} .* conj(s.phase{k}) ./ (2 * nCoefs);
        Xspec = complex(zeros(s.nFreqs, 1));
        Xspec(s.idxs{k}) = fft(Xi);
        Xr(:,k) = 2 * s.nSamps * ifft(Xspec);
    end
end