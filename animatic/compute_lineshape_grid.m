%% compute_lineshape_grid.m
% Computes the time-integrated lineshapes for the 3x3 "annual" grid, for two
% lab positions (equator vs south pole), and EXPORTS them for plotting in
% Python/matplotlib (plot_lineshape_grid.py).  No plotting is done here.
%
% Panel p is evaluated on the day when the Earth's in-plane angle from the
% stream direction is 0, 45, 90, ... , 360 degrees (the 9th panel is the
% Earth back where it started).
%
% Output: lineshape_grid_export.mat  (MATLAB v7, readable with scipy.io.loadmat)
%   mass     [nP x nS x nbins]  weighted mass in each speed bin   (NOT a density)
%   edges    [nP x (nbins+1)]   speed-bin edges per panel, km/s
%   rho      [nP x nS]          mean(w1+w2): local density relative to unperturbed stream
%   theta_lab, angles_deg, t_cross, t_start, window_hours, stream_speed, t0_iso, site_labels
%
% Needs in the same folder:
%   lineshape_window.m
%   orbit_equator.mat, orbit_southpole.mat   (from export_earth_orbit.py, see below)
%
%   python export_earth_orbit.py --lat   0 --lon 0 --height 0 --days 368 --out /data/orbit_equator.mat
%   python export_earth_orbit.py --lat -90 --lon 0 --height 0 --days 368 --out /data/orbit_southpole.mat
%
% (--days 368 so that the 360-degree panel, ~day 365.26, plus its window fits.)

clear; close all; clc;

%% ------------------------------ settings ------------------------------------
sites(1).file = 'data/orbit_equator.mat';    sites(1).label = 'Equator';
sites(1).color = [0.85 0.33 0.10];      sites(1).tag = 'eq';
sites(2).file = 'data/orbit_southpole.mat';  sites(2).label = 'South Pole';
sites(2).color = [0.00 0.45 0.74];      sites(2).tag = 'SP';

angles_deg   = 0:45:360;     % stream/Earth angle for each of the 9 panels
window_hours = 6;            % integration time of the "experiment" (a few hours)
window_anchor = 'start';     % 'start': window begins at the crossing time
                             % 'center': window is centred on it (panel 1 is clamped to t=0)
start_hour_utc = [];         % [] = use window_anchor above.  Or e.g. 0 / 6 / 12 / 18: start every
                             % window at that UTC hour on the day of the crossing, so all panels
                             % share the same time of day (local mean time = UTC + lon/15 h).
                             % Worth doing: for narrow lineshapes (theta ~ 180) the result depends
                             % strongly on WHEN in the day the window starts (see README note).
stream_speed = 220;          % km/s
nT           = 4320;          % time samples per window
nAz          = 4000;         % azimuth samples per time sample
nbins        = 800;          % speed bins per panel (the speed range is zoomed per panel)
support_thresh = 1e-3;       % zoom range = where f > this x peak (either site)
export_file  = 'data/lineshape_grid_export.mat';

%% ------------------------------ load orbits ---------------------------------
nS = numel(sites);
for s = 1:nS
    orb(s) = load(sites(s).file);                   %#ok<SAGROW>
end
dtd = orb(1).t_days(2) - orb(1).t_days(1);
if orb(1).t_days(end) < 365.3 + window_hours/24
    error('Orbit table too short: re-export with --days 368.');
end

%% ------------ ONE stream, shared by both sites ------------------------------
% The stream direction and the azimuth axis must be identical for the two
% labs, otherwise they would be looking at slightly different streams (the
% two lab positions differ by ~6400 km, ~4e-5 rad as seen from the Sun, which
% matters a lot near the caustic). Both are built from the Earth-CENTRE
% position, which is the same in both tables.
re = orb(1).r_earth;
if max(abs(orb(2).r_earth(:) - re(:))) > 1
    warning('r_earth differs between the two tables -- were they exported with the same --t0?');
end
uhat   = re(1,:) / norm(re(1,:));                    % stream direction (aligned with Earth at t0)
at     = @(d) re(round(d/dtd) + 1, :);
normal = cross(at(2), at(52));  normal = normal / norm(normal);   % orbit normal (azimuth axis)

%% ------------ day on which each target angle occurs -------------------------
rh  = re ./ sqrt(sum(re.^2, 2));
cth = rh * uhat.';
sth = cross(repmat(uhat, size(rh,1), 1), rh, 2) * normal.';
ang = unwrap(atan2(sth, cth));                        % in-plane angle of Earth from the stream, 0 -> 2pi
t_cross = interp1(ang, orb(1).t_days, angles_deg*pi/180);
switch window_anchor
    case 'center', t_start = max(0, t_cross - 0.5*window_hours/24);
    otherwise,     t_start = t_cross;
end
if ~isempty(start_hour_utc)
    t_start = floor(t_cross) + start_hour_utc/24;    % table t=0 is 00:00 UTC
end

%% ------------------------------ compute -------------------------------------
nP = numel(angles_deg);
opts   = struct('u', stream_speed, 'nT', nT, 'nAz', nAz, 'uhat', uhat, 'normal', normal);
opts_c = struct('u', stream_speed, 'nT', 200, 'nAz', 2000, 'uhat', uhat, 'normal', normal);
edges_c = linspace(150, 300, 3001);                   % coarse pass, only used to find the support
mass  = zeros(nP, nS, nbins);
edges = zeros(nP, nbins + 1);
rho   = zeros(nP, nS);
theta_lab = zeros(nP, nS);
for p = 1:nP
    % pass 1: where does the lineshape live? (union over both sites)
    lo = inf;  hi = -inf;
    for s = 1:nS
        Vc = lineshape_window(orb(s), t_start(p), window_hours, edges_c, opts_c);
        ii = find(Vc > support_thresh * max(Vc));
        lo = min(lo, edges_c(ii(1)));  hi = max(hi, edges_c(ii(end)+1));
    end
    m = 0.05*(hi - lo) + 0.05;
    edges(p,:) = linspace(lo - m, hi + m, nbins + 1);
    % pass 2: identical bins for both sites
    for s = 1:nS
        [V, info] = lineshape_window(orb(s), t_start(p), window_hours, edges(p,:), opts);
        mass(p,s,:)      = V(:);                      % mass per bin; sum(V) = mean(w1+w2)
        rho(p,s)         = info.rho; % mean(info.w1 + info.w2);
        theta_lab(p,s)   = 1; % mean(info.theta);
    end
    fprintf('panel %d/%d  theta=%3d deg  day %8.3f   rho: %s = %.4g, %s = %.4g\n', ...
            p, nP, angles_deg(p), t_cross(p), sites(1).tag, rho(p,1), sites(2).tag, rho(p,2));
end

t0_iso      = orb(1).meta.t0_iso;
site_labels = char(sites.label);
save(export_file, 'mass', 'edges', 'rho', 'theta_lab', 'angles_deg', 't_cross', 't_start', ...
     'window_hours', 'stream_speed', 't0_iso', 'site_labels', 'nT', 'nAz', '-v7');
fprintf('wrote %s   (now run: python plot_lineshape_grid.py)\n', export_file);
