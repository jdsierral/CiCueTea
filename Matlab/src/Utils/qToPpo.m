
% QTOPPO  Convert Q-factor to periods per octave (PPO)
%   ppo = qToPpo(q)
%
%   Converts the Q-factor of a constant-Q  to the corresponding
%   number of periods per octave (PPO).
%
%   INPUT:
%     q   : Q-factor (scalar or vector)
%
%   OUTPUT:
%     ppo : Periods per octave (scalar or vector)
%
%   See also: ppoToQ

function ppo = qToPpo(q)
    % PPO formula for constant-Q transform
    ppo = 1 ./ log2(1.0 + 1.0 ./ q);
end