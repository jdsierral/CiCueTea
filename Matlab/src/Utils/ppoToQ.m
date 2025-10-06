
% PPOTOQ  Convert periods per octave (PPO) to Q-factor
%   q = ppoToQ(ppo)
%
%   Converts the number of periods per octave (PPO) to the corresponding
%   Q-factor for a constant-Q transform.
%
%   INPUT:
%     ppo : Periods per octave (scalar or vector)
%
%   OUTPUT:
%     q   : Q-factor (scalar or vector)
%
%   See also: qToPpo

function q = ppoToQ(ppo)
    % Q-factor formula for constant-Q transform
    q = 1 ./ (exp2(1.0 / ppo) - 1);
end

