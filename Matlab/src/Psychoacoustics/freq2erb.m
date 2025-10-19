% FREQ2ERB frequency to equivalent rectangular bandwidth conversion
%
% usage: erb = freq2erb(freq)
% 
% This function simply computes the equivalent rectangular bandwidth
% position of the corresponding frequency. Keep in mind that ERBs are a
% perceptually based scale and do not extend or generalize outside their
% intended purpose. Use with care for this reason both in extreme cases
% where the purpose doesnt entirely apply and where the frequency range
% doesnt entirely apply.
%
% The ERB scale is a model of listening critical bands where a dense ERB
% shares particular perceptual characteristics based on masking curves,
% equal loudness perception and even complex tone integration. Notice
% however that any of these can be affected by other objective signal
% properties like loudness and timbre.
% 
% Reference: 
% [1] Moore and Glasberb "A Revision of Zwicker's Loudness Model,"
% ACTA Acustica, vol. 82, pp. 335-345, 1996
%
% See also: freq2erb, erb2freq, bark2freq, freq2bark, mel2freq, freq2mel

function erb = freq2erb(freq)
    erb = 21.4 * log10(4.37 .* freq / 1000 + 1.0);
end