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
% call), so the raw values are directly comparable. A quiet frame (far from
% the stream-crossing direction) has an integrated area of ~0.2 regardless
% of which day it is; the crossing itself is a real, near-singular caustic
% (an idealised single-speed stream lined up exactly with Earth's t0
% position formally diverges there) that can reach 10^3-10^4x that.
%
%   yscale = 'fixed'      (default) : NO per-frame rescaling at all -- every
%                                      frame plotted in the same units, so
%                                      the focusing enhancement shows up as
%                                      a real peak/plateau rising out of a
%                                      flat baseline, exactly as it should.
%   yscale = 'unit_area'            : legacy behaviour -- divide every frame
%                                      by its own integral. Useful only if
%                                      you want to compare SHAPES and
%                                      deliberately hide the enhancement.
yscale       = 'fixed';
ymax         = [];          % [] = automatic: set from the visible secondary
                            %   bump (see ymax_percentile/ymax_margin below),
                            %   so the exact-crossing frames spike off the
                            %   top of the panel -- deliberately, see below.
ymax_percentile = 99;       % percentile of per-frame peak heights used to
                            %   set ymax automatically (ignored if ymax set)
ymax_margin  = 1.4;         % headroom multiplier on top of that percentile
enh_ylim     = [0.95 5];    % y-range of the focusing strip; the caustic at t0 (theta -> 0)
                            %   diverges, exactly as in the notebook, so it goes off-scale

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
    load(cache_file, 'V', 'enh', 'days');
    nF = numel(days);
else
    V   = zeros(nb, nF);      % lineshape per frame
    enh = zeros(1, nF);       % mean focusing weight (w1+w2) in the window
    opts = struct('u', stream_speed, 'nT', nT, 'nAz', nAz);
    tic
    for k = 1:nF               % <-- change to  parfor k = 1:nF  if you have the Parallel Toolbox
        [V(:,k), info] = lineshape_window(orb, days(k), window_hours, edges, opts);
        enh(k) = mean(info.w1 + info.w2);
        fprintf('frame %3d/%3d  day %6.1f   (%.1f s elapsed)\n', k, nF, days(k), toc);
    end
    save(cache_file, 'V', 'enh', 'days', 'edges', 'window_hours', 'stream_speed', '-v7');
end

switch yscale
    case 'unit_area'
        Vp = V ./ (sum(V, 1) * de);           % each column integrates to 1 -- HIDES the effect
        ylab = 'probability density  [(km/s)^{-1}]  (per-frame unit area)';
    otherwise
        Vp = V;                               % same units in every frame -- SHOWS the effect
        ylab = 'lineshape  (fixed normalisation, arb.)';
end

peakk = max(Vp, [], 1);                       % per-frame peak height
if isempty(ymax)
    ymax = ymax_margin * percentile_nostat(peakk, ymax_percentile);
end
clipped = peakk > ymax;                       % frames whose true peak is off-scale
fprintf('ymax = %.4g  (%.0f%%ile of per-frame peaks x%.2g);  %d/%d frames clipped: day(s) %s\n', ...
        ymax, ymax_percentile, ymax_margin, sum(clipped), nF, mat2str(days(clipped)));

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
    area(vc, Vp(:,k), 'FaceColor', [0.27 0.51 0.71], 'EdgeColor', [0.1 0.25 0.45]);
    xlim([edges(1) edges(end)]);  ylim([0 ymax]);
    xlabel('speed in lab frame  [km/s]');  ylabel(ylab);
    title(sprintf('%s   |   integrating %g h', datestr(dnum, 'dd mmm yyyy'), window_hours), ...
          'FontSize', 14);
    grid on;
    if clipped(k)
        text(edges(end), 0.97*ymax, sprintf('peak off-scale: %.3g  ', peakk(k)), ...
             'Color', [0.85 0.2 0.1], 'FontWeight', 'bold', 'HorizontalAlignment', 'right', ...
             'VerticalAlignment', 'top');
    end

    % ---- (bottom-left) focusing enhancement vs time ----
    subplot(2, 2, 3);
    semilogy(days, enh, '-', 'Color', [0.6 0.6 0.6]);  hold on;
    semilogy(days(k), enh(k), 'o', 'MarkerFaceColor', [0.85 0.2 0.1], 'MarkerEdgeColor', 'none', ...
             'MarkerSize', 9);
    xlim([days(1) days(end)+1]);  ylim(enh_ylim);
    xlabel('days since t_0');  ylabel('mean relative density');
    title('gravitational-focusing enhancement (off-scale at t_0)');  grid on;

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
