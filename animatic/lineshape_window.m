function [vals, info] = lineshape_window(orb, t_start_days, window_hours, edges, opts)
if nargin < 5, opts = struct(); end
u     = get_opt(opts,'u',220);
n0    = get_opt(opts,'nT0',720);        % initial uniform samples
tolv  = get_opt(opts,'tol_speed',0.5);  % max speed change per step, in bin widths
tolw  = get_opt(opts,'tol_logw',0.2);   % max |dlog w| per step
maxN  = get_opt(opts,'maxN',2e6);
GM    = 1.32712440018e11;
uhat  = get_opt(opts,'uhat',[]);
if isempty(uhat), uhat = orb.r_lab(1,:)/norm(orb.r_lab(1,:)); end

nb = numel(edges)-1;  e1 = edges(1);  de = (edges(end)-edges(1))/nb;
T  = window_hours/24;
t  = t_start_days + linspace(0,T,n0).';
[s,w] = branch_samples(t);

while numel(t) < maxN
    ds  = max(abs(diff(s)),[],2);
    dlw = max(abs(diff(log(w))),[],2);
    lo  = min(min(s(1:end-1,:),s(2:end,:)),[],2);      % skip intervals far outside the bins
    hi  = max(max(s(1:end-1,:),s(2:end,:)),[],2);
    rel = hi >= e1 & lo <= e1 + nb*de;
    bad = rel & (ds > tolv*de | dlw > tolw);
    if ~any(bad), break; end
    tm = 0.5*(t(1:end-1)+t(2:end));
    tn = tm(bad);
    [sn,wn] = branch_samples(tn);
    [t,ix] = sort([t; tn]);
    s = [s; sn];  s = s(ix,:);
    w = [w; wn];  w = w(ix,:);
end

% ---- deposit each interval uniformly over the speed bins it spans ----------
vals = zeros(nb,1);
dtf  = diff(t)/T;
for b = 1:2
    sa = s(1:end-1,b);  sb = s(2:end,b);
    m  = 0.5*(w(1:end-1,b)+w(2:end,b)).*dtf;
    lo = min(sa,sb);  hi = max(sa,sb);
    i0 = floor((lo-e1)/de)+1;  i1 = floor((hi-e1)/de)+1;
    same = (i0==i1);
    ok = same & i0>=1 & i0<=nb;
    vals = vals + accumarray(i0(ok), m(ok), [nb 1]);
    for k = find(~same & i1>=1 & i0<=nb).'
        bins = max(i0(k),1):min(i1(k),nb);
        L = min(e1+bins*de,hi(k)) - max(e1+(bins-1)*de,lo(k));
        vals(bins) = vals(bins) + m(k)*L(:)/(hi(k)-lo(k));
    end
end

if nargout > 1
    info.rho = trapz(t, w(:,1)+w(:,2))/T;     % time-averaged enhancement
    info.t_days = t;  info.nsamples = numel(t);
end

    function [s,w] = branch_samples(tq)
        r  = interp_orbit(orb.t_days, orb.r_lab, tq);
        vl = interp_orbit(orb.t_days, orb.v_lab, tq);
        d  = sqrt(sum(r.^2,2));  rh = r./d;
        th = max(acos(max(-1,min(1,rh*uhat.'))), 1e-12);
        Phi = GM./d;  vloc = sqrt(u^2+2*Phi);  psi0 = acos(u./vloc);
        ang = [solve_psi(u,Phi,th,     psi0,         pi*ones(size(th))), ...
               solve_psi(u,Phi,pi-th,  zeros(size(th)), psi0)];
        ep  = uhat - (rh*uhat.').*rh;
        ep  = ep./max(sqrt(sum(ep.^2,2)),1e-15);
        s = zeros(numel(tq),2);  w = s;
        for b = 1:2
            sg = sign(u - vloc.*cos(ang(:,b)));
            n  = cos(ang(:,b)).*rh + (sin(ang(:,b)).*sg).*ep;
            s(:,b) = sqrt(sum((vloc.*n - vl).^2,2));
            w(:,b) = 1./jac(u,Phi,ang(:,b));
        end
    end
end
% (keep get_opt, interp_orbit, jac, theta_inf, solve_psi below)

% =============================== helpers =====================================

function v = get_opt(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

function c = cross3(a, b)
c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)];
end

function y = interp_orbit(t, Y, tq)
% Spline-interpolate a uniformly sampled table, using only a small chunk of it
% (interp1 'spline' on all ~5e5 rows would be wasteful for every frame).
tq = tq(:);
dt = t(2) - t(1);
i0 = max(1,        floor((min(tq) - t(1))/dt) + 1 - 5);
i1 = min(numel(t), ceil( (max(tq) - t(1))/dt) + 1 + 5);
if min(tq) < t(1) || max(tq) > t(end)
    error('Requested times [%.3f, %.3f] d fall outside the table [%.3f, %.3f] d.', ...
          min(tq), max(tq), t(1), t(end));
end
y = interp1(t(i0:i1), Y(i0:i1, :), tq, 'spline');
end

function J = jac(u, Phi, psi)
% notebook: jacobian_closed_form
v = sqrt(u^2 + 2*Phi);
D = u^2 + Phi - u * v .* cos(psi);
J = abs(1 - (Phi ./ D).^2);
end

function th = theta_inf(psi, u, Phi)
% notebook: theta_inf  (angle at infinity as a function of local angle psi)
v     = sqrt(u^2 + 2*Phi);
numer = v .* sin(psi) .* (u - v .* cos(psi));
denom = u * v .* cos(psi) - (v .* cos(psi)).^2 + Phi;
th    = atan(numer ./ denom);
neg   = th < 0;
th(neg) = th(neg) + pi;
end

function psi = solve_psi(u, Phi, target, lo, hi)
% Replaces scipy.optimize.minimize(|theta_inf(psi)-target|, bounds=[lo,hi]).
% theta_inf is monotonically increasing from 0 to pi on each of the two branches
% ([0, arccos(u/v)] and [arccos(u/v), pi]), so plain vectorised bisection works
% and is exact to machine precision.
for k = 1:100
    mid = 0.5*(lo + hi);
    f   = theta_inf(mid, u, Phi) - target;
    below = f < 0;                       % root lies above mid
    lo(below)  = mid(below);
    hi(~below) = mid(~below);
end
psi = 0.5*(lo + hi);
end
