%MEL2FREQ Mel to frequency conversion
%
% The mel scale is a linearization of the pitch space perception. The ideas
% is that musical intervals are not exactly logarithmically spaced in the
% frequency axis. Accordingly, a span of 4 perfect octaves (2^4) is
% percevied as wider than what 4 musical octaves should be for example.
% Accordingly, psychoacoustical analysis was performed to obtain
% statistical proof of said pitch space compresion. 
%
% Accordingly it mainly applies for perception of pitchspace and does not
% take into account the implications of more complex tones.
%
% (Notice that this concept has been critized even by one of its main
% authors and needs further revision. It is possible that other scales
% might be more relevant)

% See also: freq2erb, erb2freq, bark2freq, freq2bark, mel2freq, freq2mel

function freq = mel2freq(mel)
     freq = 700 * (exp(mel / 1127) - 1);
end

 