% ERB2FREQ equivalent rectangular bandwidth to frequency conversion
%
% usage: freq = erb2freq(freq)
% 
% This function simply computes the equivalent rectangular bandwidth
% position of the corresponding frequency. Keep in mind that ERBs are a
% perceptually based scale and do not extend or generalize outside their
% intended purpose. Use with care for this reason both in extreme cases
% where the purpose doesnt entirely apply and where the frequency range
% doesnt entirely apply.
%
% The ERB scale is a model of listening critical bands where a full ERB
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

function freq = erb2freq(erb)

    freq = 1000 * (10.^(erb/21.4) - 1.0) / 4.37;
    
end