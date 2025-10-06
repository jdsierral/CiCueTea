% NSGFVQTINIT  Initialize Nonstationary Gabor Frame for Variable-QT
%   s = nsgfVQTInit(type, fs, nSamps, fMap, frac, fMin, fMax, fRef, th)
%
%   Initializes the transform structure for the Nonstationary Gabor Frame
%   Constant-Q Transform (CQT). This function computes the transform center
%   frequencies, frequency responses, and other parameters required for the CQT.
%
%   INPUTS:
%     type   : (string) "full" or "sparse" (output type)
%     fs     : (double) Sampling frequency (Hz)
%     nSamps : (double) Number of samples for FFT
%     fMap   : (function_handle) function that linearizes the frequency axis 
%              (default: log2)
%     frac   : (double) Frequency resolution in octaves (default: 1/12)
%     fMin   : (double) Minimum frequency (Hz, default: 100)
%     fMax   : (double) Maximum frequency (Hz, default: 10000)
%     fRef   : (double) Reference frequency (Hz, default: 440)
%     th     : (double) Threshold for sparsity (default: 1e-6)
%
%   OUTPUT:
%     s : Structure containing transform parameters and precomputed values:
%         .fs, .nSamps, .nFreqs, .nBands, .fMin, .fMax, .fRef, .bax, .fax, .tax
%         .g, .gDual, .d, .type
%         (for sparse type) .idxs, .phase
%
%   The function checks for valid parameter ranges and ensures the transform
%   respects the Nyquist limit and octave constraints. For 'sparse' type, filters
%   below the threshold are zeroed and indices are stored for efficient processing.
%
%   See also: nsgfCQT, nsgfICQT, nsgfRasterize

function s = nsgfVQTInit(type, fs, nSamps, fMap, frac, fMin, fMax, fRef, th)
    arguments
        type (1,1) string {mustBeMember(type, ["full","sparse"])}
        fs (1,1) double {mustBePositive}
        nSamps (1,1) double {mustBeInteger, mustBePositive}
        fMap (1,1) function_handle = @log2 
        frac (1,1) double {mustBePositive} = 1/12
        fMin (1,1) double {mustBePositive} = 100
        fMax (1,1) double {mustBePositive, mustBeGreaterThan(fMax, fMin)} = 10000
        fRef (1,1) double {mustBePositive} = 440
        th (1,1) double {mustBePositive, mustBeLessThan(th, 1)} = 1e-6
    end

    % Check for valid frequency and Q settings
    if fMax * 2 >= fs; error("Not respecting nyquist limit"); end
    if fMin * 2 > fMax; error("Leave at least 1 octave"); end
    % if (fs / (fMin * (exp2(frac) - 1))) > nSamps; error("Q is too big"); end

    % Calculate number of bands above and below reference frequency
    nBandsUp = ceil(1/frac * (fMap(fMax) - fMap(fRef)));
    nBandsDn = ceil(1/frac * (fMap(fRef) - fMap(fMin)));
    bax = fMap(fRef) + (frac * (-nBandsDn:nBandsUp)');  % Center frequencies
    nBands = length(bax);                               % Total number of bands
    nFreqs = nSamps;                                    % Number of FFT bins
    tax = (-nSamps/2:nSamps/2-1)' / fs;                 % Time axis
    fax = (0:nFreqs-1)' * fs / nFreqs;                  % Frequency axis


    % Create Gaussian-shaped frequency responses for each band
    c = log(4) / (frac.^2);
    g = exp( -c * (bax.' - fMap(fax)).^2);
        

    % Ensure full coverage at the frequency range edges
    g(fMap(fax) <= bax(1), 1) = 1;
    g(fMap(fax) >= bax(end), end) = 1;
    

    % For sparse type, zero out small values for efficiency
    if type == "sparse"
        g(g <= th) = 0;
    end


    % Compute frame operator diagonal (energy normalization)
    d = sum(g.^2, 2);

    % Check for invertibility (no zero diagonal entries below Nyquist)
    if (any(d(fax<fs/2) == 0))
        error("Q is too high for blockSize")
    end
    gDual = g ./ d;

    % Zero out frequencies above Nyquist
    g(fax > fs/2,:) = 0;
    gDual(fax > fs/2,:) = 0;
    
    % Store parameters in output structure
    s = struct( "fs", fs, ...
                "nSamps", nSamps, ...
                "nFreqs", nFreqs, ...
                "nBands", nBands, ...
                "fMin", fMin, ...
                "fMax", fMax, ...
                "fRef", fRef, ...
                "bax", bax, ...
                "fax", fax, ...
                "tax", tax, ...
                "g", g, ...
                "d", d, ...
                "gDual", gDual, ...
                "fMap", fMap, ...
                "type", type);

    if type == "sparse"
        % Prepare sparse representation: store only nonzero indices and values
        idxs = cell(nBands, 1);
        gCell = cell(nBands, 1);
        gDualCell = cell(nBands, 1);
        phase = cell(nBands, 1);
        
        for k = 1:nBands
            ii = find(g(:,k) ~= 0);     % Indices of nonzero filter values
            ii = padIdxs(ii);           % Pad to next power of 2 for FFT efficiency
            nCoefs = length(ii);
            offset = ii(1) - 1;
            idxs{k} = ii;
            % Precompute phase factors for each band
            phase{k} = exp(1j * 2 * pi * offset * (0:nCoefs-1)'/nCoefs);
            gCell{k} = g(ii,k);         % Store nonzero filter values
            gDualCell{k} = gDual(ii,k); % Store nonzero dual filter values
        end
        
        g = gCell;
        gDual = gDualCell;
        s.idxs = idxs;
        s.phase = phase;
    end

    s.g = g;            % Analysis windows (or cell array for sparse)
    s.d = d;            % Frame operator diagonal
    s.gDual = gDual;    % Dual windows (or cell array for sparse)
    s.type = type;      % 'full' or 'sparse'
end

function ii = padIdxs(ii)
    % Pad index vector to at least 4 and to next power of 2 for FFT efficiency
    i0 = ii(1);
    nIdx = length(ii);
    if nIdx < 4
        nIdx = 4;
    end
    nIdx = 2^nextpow2(nIdx);
    ii = (i0:i0+nIdx-1);
end