function q = ppoToQ(ppo)
    q = 1 ./ (exp2(1.0 / ppo) - 1);
end

