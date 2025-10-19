% NSGRASTERIZE  Rasterize sparse CQT coefficients into a matrix
%   Xr = nsgfRasterize(X, s)
%
%   Converts a cell array of sparse Constant-Q Transform (CQT) coefficients
%   into a dense matrix representation by interpolating each band to the
%   common sample length. Useful for visualization or further processing.
%
%   INPUTS:
%     X : Cell array of CQT coefficients (one cell per band)
%     s : Structure with transform parameters (see nsgfCQTInit)
%
%   OUTPUT:
%     Xr : Matrix of size [s.nSamps, s.nBands] with interpolated coefficients
%
%   See also: nsgfCQT, nsgfICQT, nsgfCQTInit

function Xr = nsgfRasterize(X, s)
    % Preallocate output matrix
    Xr = zeros(s.nSamps, s.nBands);
    for k = 1:s.nBands
        % Undo phase and interpolate each band to dense length
        Xr(:,k) = interpft(X{k}, s.nSamps);
    end
end