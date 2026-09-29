function [vals, info] = lineshape_window(orb, t_start_days, window_hours, edges, opts)
%LINESHAPE_WINDOW  Time-integrated speed lineshape over one window.
%
%   Port of the loop in cells 25-29 of annual.ipynb (the "for day in
%   np.linspace(0,1,400)" block), vectorised so that no Python-style loops
%   over the azimuth samples are needed.
%
%   [vals, info] = lineshape_window(orb, t_start_days, window_hours, edges, opts)
%
%   orb            struct from export_earth_orbit.py  (load('earth_orbit.mat'))
%                  needs: t_days (Nx1), r_lab (Nx3, km), v_lab (Nx3, km/s)
%   t_start_days   start of the window, in days since the table's t0
%   window_hours   length of the window to integrate over (e.g. 6..24)
%   edges          speed-bin edges [km/s] (fixed across frames for an animation)
%
%   opts (all optional):
%     .u          stream speed at infinity            [220 km/s]   (notebook 'spd')
%     .nT         time samples inside the window      [400]
%     .nAz        azimuth samples per time sample     [4000]
%     .az_mode    'valid'    (default) put all nAz samples on the part of the
%                            azimuth range where the geometry is real, [0, az_max].
%                            Same distribution as the notebook in the limit
%                            nAz -> inf, but no samples are wasted (near opposition
%                            the valid arc can be ~0.01 rad wide out of pi/2).
%                 'notebook' uses linspace(0,pi/2,nAz) and drops invalid points,
%                            sample-for-sample identical to the notebook.
%     .uhat       1x3 unit vector: stream direction at infinity.
%                 default = r_lab(1,:)/|r_lab(1,:)|  (= notebook vel_str/spd,
%                 i.e. direction of the lab position at t0)
%     .normal     1x3 rotation axis used for the azimuth sweep.
%                 default = cross(r(t0+2d), r(t0+52d)) normalised (notebook cell 17)
%     .average    true (default): divide by nT so frames are comparable.
%                 false: plain sum, exactly like the notebook's `vals`.
%
%   vals   (numel(edges)-1) x 1   lineshape in each speed bin
%   info   struct with per-sample diagnostics (ang1, ang2, theta, dist, ...)

if nargin < 5, opts = struct(); end
u    = get_opt(opts, 'u',   220);
nT   = get_opt(opts, 'nT',  400);
nAz  = get_opt(opts, 'nAz', 4000);
mode = get_opt(opts, 'az_mode', 'valid');
avg  = get_opt(opts, 'average', true);

GM = 1.32712440018e11;                 % Sun, km^3/s^2  (= G*M_sun in astropy)

% ---- reference directions (fixed for the whole animation) -----------------
uhat = get_opt(opts, 'uhat', []);
if isempty(uhat)
    uhat = orb.r_lab(1,:) / norm(orb.r_lab(1,:));
end
nrm = get_opt(opts, 'normal', []);
if isempty(nrm)
    ra = interp_orbit(orb.t_days, orb.r_lab, 2);
    rb = interp_orbit(orb.t_days, orb.r_lab, 52);
    nrm = cross3(ra, rb);
end
nrm = nrm / norm(nrm);

% ---- sample the orbit inside the window ------------------------------------
tq   = t_start_days + linspace(0, window_hours/24, nT).';   % days
r    = interp_orbit(orb.t_days, orb.r_lab, tq);              % nT x 3   [km]
vlab = interp_orbit(orb.t_days, orb.v_lab, tq);              % nT x 3   [km/s]
dist = sqrt(sum(r.^2, 2));
rhat = r ./ dist;

theta = acos(max(-1, min(1, rhat * uhat.')));   % angle(stream dir, r)   (notebook 'theta')
Phi   = GM ./ dist;                             % potential, km^2/s^2    (notebook '-grav(dist)')
vloc  = sqrt(u^2 + 2*Phi);                      % local speed            (notebook 'spd_loc')
psi0  = acos(u ./ vloc);                        % boundary between the two branches

% ---- the two solutions (notebook: minimize(func,...) -> ang1, ang2) --------
ang1 = solve_psi(u, Phi, theta,      psi0,          pi*ones(nT,1));
ang2 = solve_psi(u, Phi, pi - theta, zeros(nT,1),   psi0);

% density factors 1/|dv_inf/dv|
w1 = 1 ./ jac(u, Phi, ang1);
w2 = 1 ./ jac(u, Phi, ang2);

% ---- azimuth sweep + histogram ---------------------------------------------
az0   = linspace(0, pi/2, nAz).';          % notebook's grid
nb    = numel(edges) - 1;
e1    = edges(1);  de = (edges(end) - edges(1)) / nb;   % uniform edges assumed
vals  = zeros(nb, 1);

for j = 1:nT
    rh = rhat(j,:);
    for branch = 1:2
        if branch == 1, ang = ang1(j); w = w1(j); else, ang = ang2(j); w = w2(j); end

        if strcmp(mode, 'notebook')
            a     = az0;
            ratio = cos(ang) ./ cos(a);          % notebook: cos(ang)/cos(azimuth)
            ok    = abs(ratio) <= 1;             % numpy gave NaN here; MATLAB would go complex
            a     = a(ok);  ratio = ratio(ok);
        else
            azmax = acos(min(1, abs(cos(ang))));         % largest az with |ratio| <= 1
            a     = linspace(0, azmax, nAz).';
            ratio = max(-1, min(1, cos(ang) ./ cos(a))); % clip rounding at the end point
        end
        nok = numel(a);
        if nok == 0, continue; end
        p  = pi/2 - asin(ratio);                 % 'polar'
        ca = cos(a);  sa = sin(a);

        % step 1 (notebook rotationr): rotate rhat about (normal x rhat) by p
        %         -> n1 = rhat*cos p - aperp*sin p
        aperp = nrm - rh * (nrm*rh.');           % part of normal perpendicular to rhat
        aperp = aperp / norm(aperp);
        n1x = rh(1)*cos(p) - aperp(1)*sin(p);
        n1y = rh(2)*cos(p) - aperp(2)*sin(p);
        n1z = rh(3)*cos(p) - aperp(3)*sin(p);

        % step 2 (notebook rotationn): rotate n1 about normal by az (Rodrigues)
        dotn = nrm(1)*n1x + nrm(2)*n1y + nrm(3)*n1z;
        cx = nrm(2)*n1z - nrm(3)*n1y;
        cy = nrm(3)*n1x - nrm(1)*n1z;
        cz = nrm(1)*n1y - nrm(2)*n1x;
        nx = n1x.*ca + cx.*sa + nrm(1)*dotn.*(1-ca);
        ny = n1y.*ca + cy.*sa + nrm(2)*dotn.*(1-ca);
        nz = n1z.*ca + cz.*sa + nrm(3)*dotn.*(1-ca);

        % speed relative to the lab
        sx = vloc(j)*nx - vlab(j,1);
        sy = vloc(j)*ny - vlab(j,2);
        sz = vloc(j)*nz - vlab(j,3);
        spd = sqrt(sx.^2 + sy.^2 + sz.^2);

        idx = floor((spd - e1)/de) + 1;
        in  = idx >= 1 & idx <= nb;
        if any(in)
            vals = vals + accumarray(idx(in), w/nok, [nb 1]);
        end
    end
end

if avg, vals = vals / nT; end

if nargout > 1
    info = struct('t_days', tq, 'dist_km', dist, 'theta', theta, 'ang1', ang1, ...
                  'ang2', ang2, 'w1', w1, 'w2', w2, 'vloc', vloc, ...
                  'vlab_speed', sqrt(sum(vlab.^2,2)), 'normal', nrm, 'uhat', uhat);
end
end

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
