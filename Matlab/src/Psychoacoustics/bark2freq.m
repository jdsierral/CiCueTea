% BARK2FREQ bark scale to frequency conversion
%
% usage: freq = bark2freq(freq)
%
% The bark scale is a psychoacoustically defined scale of perception of
% frequency space in which the perception of such space is linearized.
% The idea behind it is that human perception is based on critical bands of
% hearing that share specific behaviors. This implies that at a certain
% frequency neighoring frequencies are perceived in similar or distinctive
% ways in terms of masking, loudness, amplitude interaction, phase, etc.
% The bandwidth of such frequencies is not constant across frequency so
% the purpose of the bark scale is to quantify such bandwidth across
% frequencies. Accordingly, the frequencies themselves are not relevant but
% the bandwidth of said critical bandwidths which is equivalent to a delta
% of 1 bark.
%
% REF H. Traunmüller (1990) "Analytical expressions for the tonotopic 
% sensory scale" J. Acoust. Soc. Am. 88: 97-100. 
%
% See also: freq2erb, erb2freq, bark2freq, freq2bark, mel2freq, freq2mel

function freq = bark2freq(bark)

  freq = 1960./(26.81./(bark+0.53)-1);
  
end