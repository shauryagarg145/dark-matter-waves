function p = percentile_nostat(x, pct)
%PERCENTILE_NOSTAT  Linear-interpolation percentile, no toolbox required.
%   Works identically in MATLAB (without the Statistics and Machine
%   Learning Toolbox) and in Octave (without the statistics package).
%
%   p = percentile_nostat(x, pct)   pct in [0, 100]
x = sort(x(:));
n = numel(x);
if n == 1
    p = x(1);
    return;
end
q  = (pct/100) * (n - 1) + 1;          % fractional index, 1-based
i0 = floor(q);
i1 = min(n, i0 + 1);
f  = q - i0;
p  = (1 - f) * x(i0) + f * x(i1);
end
