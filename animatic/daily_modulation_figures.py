#!/usr/bin/env python
"""
daily_modulation_figures.py
---------------------------
Three-panel figure for the "Modelling the time-dependent velocity" section:

 (a) Sun-lab distance over one day: the lab's offset from the Earth's centre, r_lab - r_earth,
     for the equator, a mid-latitude site and the South Pole, with the (much larger, smooth)
     change of the Earth-Sun distance over the same 24 h shown for scale.
 (b) Lab-frame speed of a stream, |v_str - v_lab(t)|, over the same day: the orbital drift
     (the pole sees only this) plus the sidereal-day sinusoid from the Earth's spin, whose
     semi-amplitude  d_d v_eq cos(lat)  is printed and compared with the fitted value.
 (c) The focusing density rho = w1 + w2 within +-30 min of the moment the Earth crosses the
     stream axis: the three labs miss the axis by different distances (set by their offset from
     the Earth's centre), so the spike seen at each is different.

The stream is the "cherry-picked" one of the chapter: it is aligned with the Earth's centre at
12:00 UT on 5 March 2013, with speed u = 220 km/s at infinity (single speed, single direction).
Panel (b) uses the stream's solar-frame velocity as a fixed vector (no focusing), which isolates
the effect of the lab's motion; focusing changes v_str[r_lab] only slowly over a day away from
the axis.

Tables (made with export_earth_orbit.py; this script makes them for you if missing, --make-tables):
    python export_earth_orbit.py --t0 "2013-03-05 00:00:00" --days 1.0       --dt 60 --lat LAT --lon 0 --height 0 --out day_SITE.mat
    python export_earth_orbit.py --t0 "2013-03-05 11:30:01" --days 0.0416667 --dt 2  --lat LAT --lon 0 --height 0 --out fine_SITE.mat
(the fine grid starts 1 s off the hour so that no sample falls exactly on the singular axis).
SITE/LAT = equator/0, midlat/51.4779, southpole/-90.

Usage:
    python daily_modulation_figures.py --make-tables
    python daily_modulation_figures.py --data-dir . --out daily_modulation --split
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ----------------------------- style (same as plot_lineshape_grid.py) --------
STYLE = {
    "font.family": ["EB Garamond", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}
SITES = [  # tag, label, geodetic latitude [deg], colour
    ("equator", "Equator", 0.0, "#d62728"),
    ("midlat", r"$45^\circ$N", 45.0, "#2ca02c"),
    ("southpole", "South Pole", -90.0, "#1f77b4"),
]
GREY = "0.45"
# -----------------------------------------------------------------------------

GM_SUN = 1.32712440018e11     # km^3 s^-2
U_STREAM = 220.0              # km/s, speed of the stream at infinity
T_ALIGN_UT = (12, 0, 0)       # stream aligned with the Earth's centre at 12:00:00 UT on the table date
T_SID = 86164.0905            # s, sidereal day
V_EQ = 2 * np.pi * 6378.137 / T_SID
OBLIQUITY = np.deg2rad(23.4393)


# ============================== focusing (port of lineshape_window.m) ==========
def theta_inf(psi, u, Phi):
    v = np.sqrt(u**2 + 2 * Phi)
    numer = v * np.sin(psi) * (u - v * np.cos(psi))
    denom = u * v * np.cos(psi) - (v * np.cos(psi)) ** 2 + Phi
    th = np.arctan(numer / denom)
    return np.where(th < 0, th + np.pi, th)


def solve_psi(u, Phi, target, lo, hi):
    """theta_inf rises monotonically 0 -> pi on each branch, so plain bisection is exact."""
    lo = np.array(lo, float) * np.ones_like(target)
    hi = np.array(hi, float) * np.ones_like(target)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        below = theta_inf(mid, u, Phi) - target < 0
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def jac(u, Phi, psi):
    v = np.sqrt(u**2 + 2 * Phi)
    D = u**2 + Phi - u * v * np.cos(psi)
    return np.abs(1 - (Phi / D) ** 2)


def focusing_weights(r, uhat, u=U_STREAM):
    """r: (N,3) lab position relative to the Sun [km] -> theta, w1, w2 (density of the two branches)."""
    d = np.linalg.norm(r, axis=1)
    rh = r / d[:, None]
    theta = np.arctan2(np.linalg.norm(np.cross(rh, uhat), axis=1), rh @ uhat)   # accurate near 0
    Phi = GM_SUN / d
    psi0 = np.arccos(u / np.sqrt(u**2 + 2 * Phi))
    ang1 = solve_psi(u, Phi, theta, psi0, np.pi)
    ang2 = solve_psi(u, Phi, np.pi - theta, 0.0, psi0)
    return theta, 1 / jac(u, Phi, ang1), 1 / jac(u, Phi, ang2)


# ============================== data ===========================================
def load_table(path):
    m = loadmat(path, squeeze_me=True)
    t0 = datetime.strptime(str(m["meta"]["t0_iso"].item() if hasattr(m["meta"], "item") else m["meta"].t0_iso)[:19],
                           "%Y-%m-%d %H:%M:%S")
    return dict(t_s=np.asarray(m["t_s"], float), t0=t0, r_lab=np.asarray(m["r_lab"], float),
                v_lab=np.asarray(m["v_lab"], float), r_e=np.asarray(m["r_earth"], float),
                v_e=np.asarray(m["v_earth"], float))


def make_tables(data_dir, exporter, modules_dir):
    for tag, _, lat, _ in SITES:
        for kind, t0, days, dt in (("day", "2013-03-05 00:00:00", 1.0, 60),
                                   ("fine", "2013-03-05 11:30:01", 0.0416667, 2)):
            out = os.path.join(data_dir, f"{kind}_{tag}.mat")
            if os.path.exists(out):
                continue
            cmd = [sys.executable, exporter, "--t0", t0, "--days", str(days), "--dt", str(dt),
                   "--lat", str(lat), "--lon", "0", "--height", "0", "--out", out]
            if modules_dir:
                cmd += ["--modules-dir", modules_dir]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


def stream_geometry(day):
    """uhat = direction of the Earth's centre at the alignment time; n_hat = orbit normal."""
    i = int(round((T_ALIGN_UT[0] * 3600 + T_ALIGN_UT[1] * 60 + T_ALIGN_UT[2]) / (day["t_s"][1] - day["t_s"][0])))
    r, v = day["r_e"][i], day["v_e"][i]
    uhat = r / np.linalg.norm(r)
    n = np.cross(r, v)
    return uhat, n / np.linalg.norm(n)


def sidereal_amplitude(t_s, y):
    """Least-squares amplitude of the sidereal-period sinusoid in y(t) (km/s or km)."""
    w = 2 * np.pi / T_SID
    A = np.c_[np.sin(w * t_s), np.cos(w * t_s), np.ones_like(t_s)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.hypot(c[0], c[1])


# ============================== panels =========================================
def panel_distance(ax, days, hours):
    for tag, label, _, col in SITES:
        d = days[tag]
        ax.plot(hours, (np.linalg.norm(d["r_lab"], axis=1) - np.linalg.norm(d["r_e"], axis=1)) / 1e3,
                color=col, lw=1.6, label=label)
    d = days["equator"]
    #re = np.linalg.norm(d["r_e"], axis=1)
    #ax.plot(hours, (re - re.mean()) / 1e3, color=GREY, lw=1.4, ls="--", label="orbit (Earth's centre)")
    ax.set_ylabel(r"$r_{\rm lab} - r_\oplus$  [$10^3$ km]")
    ax.legend(frameon=False, loc="lower left")


def panel_speed(ax, days, hours, uhat, results):
    for tag, label, lat, col in SITES:
        d = days[tag]
        vp = np.linalg.norm(U_STREAM * uhat - d["v_lab"], axis=1)
        ax.plot(hours, vp - vp.mean(), color=col, lw=1.6, label=label)
        ve = np.linalg.norm(U_STREAM * uhat - days[tag]["v_e"], axis=1)
        results[tag] = dict(amp_fit=sidereal_amplitude(d["t_s"], vp - ve), lat=lat)
    ax.set_ylabel(r"$|\mathbf{u}_{\rm str} - \mathbf{v}_{\rm lab}| - \langle |\mathbf{u}_{\rm str} - \mathbf{v}_{\rm lab}| \rangle$  [km s$^{-1}$]")
    #ax.set_ylabel(r"$v_{\rm det} - \langle v_{\rm det} \rangle$ [km s$^{-1}$]")
    ax.legend(frameon=False, loc="lower left")


def panel_density(ax, fines, uhat, results):
    t_align = T_ALIGN_UT[0] * 3600 + T_ALIGN_UT[1] * 60 + T_ALIGN_UT[2]
    for tag, label, _, col in SITES:
        f = fines[tag]
        minutes = (f["t_s"] + (f["t0"] - datetime(2013, 3, 5)).total_seconds() - t_align) / 60.0
        th, w1, w2 = focusing_weights(f["r_lab"], uhat)
        rho = w1 + w2
        ax.semilogy(minutes, rho, color=col, lw=1.6, label=label)
        d = np.linalg.norm(f["r_lab"][0])
        k = np.argmax(rho)
        results[tag].update(rho_peak=rho[k], t_peak_min=minutes[k], theta_min=th.min(), miss_km=th.min() * d)
    f = fines["equator"]
    minutes = (f["t_s"] + (f["t0"] - datetime(2013, 3, 5)).total_seconds() - t_align) / 60.0
    _, w1, w2 = focusing_weights(f["r_e"], uhat)
    ax.semilogy(minutes, w1 + w2, color=GREY, lw=1.4, ls="--", label="Earth's centre")
    #ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
    ax.set_ylabel(r"density enhancement  $\rho / \rho_\infty$")
    ax.set_xlabel("minutes from alignment (12:00 UT)")
    ax.legend(frameon=False, loc="upper right")


def build(args):
    plt.rcParams.update(STYLE)
    days = {t: load_table(os.path.join(args.data_dir, f"day_{t}.mat")) for t, *_ in SITES}
    fines = {t: load_table(os.path.join(args.data_dir, f"fine_{t}.mat")) for t, *_ in SITES}
    uhat, n_orbit = stream_geometry(days["equator"])
    hours = days["equator"]["t_s"] / 3600.0
    results = {}

    def draw(ax, which):
        if which == "a":
            panel_distance(ax, days, hours)
        elif which == "b":
            panel_speed(ax, days, hours, uhat, results)
        else:
            panel_density(ax, fines, uhat, results)
        if which in "ab":
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 3))
            ax.set_xlabel("UT on 5 March  [h]")
        ax.text(0.03, 0.96, f"({which})", transform=ax.transAxes, va="top", ha="left", fontsize=15)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, which in zip(axes, "abc"):
        draw(ax, which)
    fig.tight_layout()
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=args.dpi)
    if args.split:
        for which in "abc":
            f1, a1 = plt.subplots(figsize=(5.6, 4.8))
            draw(a1, which)
            f1.tight_layout()
            f1.savefig(f"{args.out}_{which}.pdf")
            f1.savefig(f"{args.out}_{which}.png", dpi=args.dpi)
            plt.close(f1)

    # ---- numbers quoted in the text ------------------------------------------
    dd = np.linalg.norm(np.cross(uhat, np.array([0, np.sin(OBLIQUITY), np.cos(OBLIQUITY)])))
    print(f"stream direction (ecliptic) = {np.round(uhat, 4)};  d_d = {dd:.3f}")
    re = np.linalg.norm(days["equator"]["r_e"], axis=1)
    print(f"orbital change of r_earth over the 24 h: {re.max() - re.min():.3e} km (peak-to-peak)")
    print(f"{'site':<10} {'lat':>7}  {'pred. semi-amp':>14}  {'fitted':>8}   (km/s)")
    for tag, label, lat, _ in SITES:
        pred = dd * V_EQ * np.cos(np.deg2rad(lat))
        print(f"{tag:<10} {lat:7.2f}  {pred:14.4f}  {results[tag]['amp_fit']:8.4f}")
    for tag, label, lat, _ in SITES:
        d = days[tag]
        off = np.linalg.norm(d["r_lab"], axis=1) - np.linalg.norm(d["r_e"], axis=1)
        print(f"{tag:<10} r_lab-r_earth over the day: {off.min():+.0f} .. {off.max():+.0f} km"
              f"   | closest approach to axis {results[tag]['miss_km']:.0f} km"
              f" (theta_min = {results[tag]['theta_min']:.2e} rad), peak rho = {results[tag]['rho_peak']:.3g}"
              f" at {results[tag]['t_peak_min']:+.1f} min")
    print(f"wrote {args.out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="plots/daily_modulation")
    ap.add_argument("--split", action="store_true", help="also save panels (a), (b), (c) separately")
    ap.add_argument("--make-tables", action="store_true", help="run export_earth_orbit.py for any missing table")
    ap.add_argument("--exporter", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_earth_orbit.py"))
    ap.add_argument("--modules-dir", default=None, help="folder containing vel_conversions.py")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if args.make_tables:
        make_tables(args.data_dir, args.exporter, args.modules_dir)
    build(args)


if __name__ == "__main__":
    main()
