%% make_animatic.m
% Annual-modulation lineshape animatic.
%   1. loads the orbit table written by export_earth_orbit.py
%   2. for a series of start days through the year, integrates the lineshape
%      over a window of `window_hours`  (lineshape_window.m)
%   3. renders one video frame per start day
%
% Needs in the same folder: lineshape_window.m, earth_orbit.mat
% Works in MATLAB (VideoWriter -> .mp4) and in Octave (falls back to PNG frames).

clear; close all; clc;

%% ------------------------------ settings ------------------------------------
orbit_file   = 'earth_orbit.mat';

window_hours = 24;          % integrate the lineshape over this many hours (6..24)
first_day    = 0;           % days since t0 (t0 = 2013-06-22 00:00 UTC, as in the notebook)
last_day     = 365;
frame_step_d = 0.25;           % one frame every 3 days -> ~122 frames

stream_speed = 220;         % km/s, notebook 'spd'
nT           = 400;         % time samples inside each window   (notebook: 400)
nAz          = 4000;        % azimuth samples per time sample   (notebook: 50000, but on a
                            %   wasteful grid -- see az_mode in lineshape_window.m)
edges        = linspace(210, 240, 701);   % speed bins [km/s], FIXED across all frames

% Normalisation: lineshape_window already returns values on a CONSISTENT
% absolute scale from one frame to the next (same u, nT, nAz, edges every
% call), so the raw values are directly comparable -- no per-frame rescaling
% is applied (that would hide the enhancement; 'unit_area' mode below
% brings the old behaviour back if you ever want it).
%
%   yscale = 'fixed'      (default) : raw values, same units every frame.
%   yscale = 'unit_area'            : legacy -- divide every frame by its
%                                      own integral. Hides the enhancement.
yscale       = 'fixed';

% The dynamic range is enormous: a quiet day sits around 1e-2 to 1e-1, the
% broad antipodal bump (theta=180, ~day 183 for this stream direction)
% rises to order 1, and the exact crossing (theta=0, day 0 / day 365) is a
% real caustic that reaches ~1e2-1e3 for these settings -- it would climb
% further if window_hours were shrunk further (it is a genuine divergence,
% only numerically regularised by the finite window/time-sampling). A
% single linear y-axis can't show the quiet baseline and the crossing at
% once, so by default the axis is logarithmic: this shows the finite
% antipodal bump AND the much taller crossing on the same plot, with
% nothing clipped off.
use_log_y    = true;        % false = old linear axis with percentile-based clipping

% --- log-axis settings (used when use_log_y = true) ---
yfloor_pct   = 5;           % floor = this percentile of the NONZERO values in each frame
                            %   (the "quiet shoulder" level, not the literal minimum, which
                            %   is set by a single near-empty edge bin and isn't informative)
ymax_log     = [];          % [] = automatic: global max over the whole year x margin --
                            %   no clipping needed, log scale shows the true peak directly
ymax_log_margin = 1.3;

% --- linear-axis settings (used when use_log_y = false) ---
ymax         = [];          % [] = automatic (percentile-based, WITH clipping+annotation)
ymax_percentile = 99;
ymax_margin  = 1.4;
enh_ylim     = [0.95 5];    % y-range of the (linear) focusing strip in that mode

video_name   = 'lineshape_annual';   % -> lineshape_annual.mp4
fps          = 20;
cache_file   = 'lineshape_frames.mat';
recompute    = true;        % false = reuse cache_file if it exists

%% ------------------------------ load orbit ----------------------------------
orb = load(orbit_file);
t0_num = datenum(orb.meta.t0_iso(1:19), 'yyyy-mm-dd HH:MM:SS');
if orb.t_days(end) < last_day + window_hours/24
    error('Orbit table only spans %.1f days; re-run the exporter with --days larger.', orb.t_days(end));
end

days = first_day:frame_step_d:last_day;
nF   = numel(days);
nb   = numel(edges) - 1;
de   = (edges(end) - edges(1)) / nb;
vc   = 0.5*(edges(1:end-1) + edges(2:end));

%% ------------------------------ compute frames ------------------------------
if ~recompute && exist(cache_file, 'file')
    load(cache_file, 'V', 'days');
    nF = numel(days);
else
    V   = zeros(nb, nF);      % lineshape per frame
    opts = struct('u', stream_speed, 'nT', nT, 'nAz', nAz);
    tic
    for k = 1:nF               % <-- change to  parfor k = 1:nF  if you have the Parallel Toolbox
        V(:,k) = lineshape_window(orb, days(k), window_hours, edges, opts);
        fprintf('frame %3d/%3d  day %6.1f   (%.1f s elapsed)\n', k, nF, days(k), toc);
    end
    save(cache_file, 'V', 'days', 'edges', 'window_hours', 'stream_speed', '-v7');
end

% The literal relative density: int f(v) dv over the window, in the SAME
% fixed units as V itself (no renormalisation), so this is directly "how
% much more (or less) dark matter is passing through, right now, compared
% to a quiet day" -- exactly the top panel's area, tracked across the year.
tot = sum(V, 1) * de;

switch yscale
    case 'unit_area'
        Vp = V ./ (sum(V, 1) * de);           % each column integrates to 1 -- HIDES the effect
        ylab = 'probability density  [(km/s)^{-1}]  (per-frame unit area)';
    otherwise
        Vp = V;                               % same units in every frame -- SHOWS the effect
        ylab = 'lineshape  (fixed normalisation, arb.)';
end

peakk = max(Vp, [], 1);                       % per-frame peak height

if use_log_y
    if isempty(ymax_log)
        ymax_log = ymax_log_margin * max(Vp(:));
    end
    % Per-frame floor: the level of the typical/quiet part of THIS frame's
    % distribution, not its literal minimum. A global floor would either
    % sit far below every quiet frame (if set from a near-singular frame's
    % tiny edge-bin values) or clip the bulk of a quiet frame (if set too
    % high), so this is computed per frame from that frame's own nonzero
    % values, then floored again by the smallest such value across the
    % year so the axis limit itself stays fixed across frames.
    yfloor_frame = nan(1, nF);
    for k = 1:nF
        nz = Vp(Vp(:,k) > 0, k);
        if isempty(nz), nz = ymax_log; end
        yfloor_frame(k) = percentile_nostat(nz, yfloor_pct);
    end
    yfloor = min(yfloor_frame);
    clipped = false(1, nF);                    % nothing is clipped on a log axis
    fprintf('log y-axis: [%.3g, %.3g]  (global max x%.2g; floor = %.0f%%ile of nonzero values)\n', ...
            yfloor, ymax_log, ymax_log_margin, yfloor_pct);
else
    if isempty(ymax)
        ymax = ymax_margin * percentile_nostat(peakk, ymax_percentile);
    end
    clipped = peakk > ymax;                     % frames whose true peak is off-scale
    fprintf('ymax = %.4g  (%.0f%%ile of per-frame peaks x%.2g);  %d/%d frames clipped: day(s) %s\n', ...
            ymax, ymax_percentile, ymax_margin, sum(clipped), nF, mat2str(days(clipped)));
end

%% ------------------------------ render --------------------------------------
fig = figure('Color', 'w', 'Position', [50 50 1280 720]);

% whole orbit (daily), for the mini-map
dtd  = orb.t_days(2) - orb.t_days(1);            % table step [days]
dd   = (0:365)';
ridx = min(numel(orb.t_days), round(dd / dtd) + 1);
AU   = 1.495978707e8;
rorb = orb.r_lab(ridx, :) / AU;
uhat = orb.r_lab(1,:) / norm(orb.r_lab(1,:));

use_video = true;
try
    vw = VideoWriter(video_name, 'MPEG-4');
    vw.FrameRate = fps;  vw.Quality = 95;
    open(vw);
catch
    use_video = false;
    warning('VideoWriter not available -> writing PNG frames instead.');
end

for k = 1:nF
    clf(fig);
    dnum = t0_num + days(k);

    % ---- (top) lineshape ----
    subplot(2, 2, [1 2]);
    if use_log_y
        % area() needs a finite baseline -- 0 is -Inf in log space -- so the
        % curve (and the fill) are clamped to the floor; this just makes
        % empty/near-empty bins sit visibly at the bottom of the panel
        % instead of breaking the plot.
        yc = max(Vp(:,k), yfloor);
        area(vc, yc, 'basevalue', yfloor, 'FaceColor', [0.27 0.51 0.71], 'EdgeColor', [0.1 0.25 0.45]);
        set(gca, 'YScale', 'log');
        ylim([yfloor ymax_log]);
    else
        area(vc, Vp(:,k), 'FaceColor', [0.27 0.51 0.71], 'EdgeColor', [0.1 0.25 0.45]);
        ylim([0 ymax]);
        if clipped(k)
            text(edges(end), 0.97*ymax, sprintf('peak off-scale: %.3g  ', peakk(k)), ...
                 'Color', [0.85 0.2 0.1], 'FontWeight', 'bold', 'HorizontalAlignment', 'right', ...
                 'VerticalAlignment', 'top');
        end
    end
    xlim([edges(1) edges(end)]);
    xlabel('speed in lab frame  [km/s]');  ylabel(ylab);
    title(sprintf('%s   |   integrating %g h', datestr(dnum, 'dd mmm yyyy'), window_hours), ...
          'FontSize', 14);
    grid on;

    % ---- (bottom-left) relative density vs time ----
    % This is int V(v) dv for that frame's window -- literally the area under
    % the top panel -- so it is a direct "how much more/less dark matter is
    % passing through right now compared to a quiet day" readout, not a proxy.
    subplot(2, 2, 3);
    semilogy(days, tot, '-', 'Color', [0.6 0.6 0.6]);  hold on;
    semilogy(days(k), tot(k), 'o', 'MarkerFaceColor', [0.85 0.2 0.1], 'MarkerEdgeColor', 'none', ...
             'MarkerSize', 9);
    xlim([days(1) days(end)+1]);
    if use_log_y
        ylim([0.9*min(tot) 1.2*max(tot)]);  % same philosophy as the top panel: show it all
        title('relative density (= area under top panel)');
    else
        ylim(enh_ylim);
        title('relative density (off-scale at t_0)');
    end
    xlabel('days since t_0');  ylabel('relative density');  grid on;

    % ---- (bottom-right) mini-map of the orbit ----
    subplot(2, 2, 4);
    plot(rorb(:,1), rorb(:,2), '-', 'Color', [0.6 0.6 0.6]);  hold on;
    plot(0, 0, 'o', 'MarkerFaceColor', [0.95 0.7 0.1], 'MarkerEdgeColor', 'none', 'MarkerSize', 12);
    ri = min(numel(orb.t_days), round(days(k) / dtd) + 1);
    plot(orb.r_lab(ri,1)/AU, orb.r_lab(ri,2)/AU, 'o', 'MarkerFaceColor', [0.1 0.4 0.85], ...
         'MarkerEdgeColor', 'none', 'MarkerSize', 9);
    % stream direction: passes through the Sun and reaches Earth's t0 position (theta = 0 there)
    quiver(-2.4*uhat(1), -2.4*uhat(2), 3.4*uhat(1), 3.4*uhat(2), 0, 'Color', [0.85 0.2 0.1], ...
           'LineWidth', 1.5);
    text(-2.4*uhat(1) + 0.15, -2.4*uhat(2), 'DM stream', 'Color', [0.85 0.2 0.1]);
    axis equal;  xlim([-2.6 2.6]);  ylim([-2.6 2.6]);
    xlabel('x_{ecl}  [AU]');  ylabel('y_{ecl}  [AU]');  title('Earth & stream direction');  grid on;

    drawnow;
    if use_video
        writeVideo(vw, getframe(fig));
    else
        print(fig, sprintf('%s_%04d.png', video_name, k), '-dpng', '-r100');
    end
end
if use_video, close(vw); fprintf('wrote %s.mp4\n', video_name); end