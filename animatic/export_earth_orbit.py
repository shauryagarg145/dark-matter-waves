#!/usr/bin/env python
"""
export_earth_orbit.py
---------------------
Tabulate the Sun-centred position and velocity of the lab (and of the Earth's
centre) over a long time span using YOUR vel_conversions.sun_to_lab_position,
and save it for MATLAB.  Because the table is produced by the very same
function the notebook calls, MATLAB sees exactly the r and v_lab of the notebook.

    r, dist, v, speed = sun_to_lab_position(times, site, frame="ecliptic")

Frame: HeliocentricMeanEcliptic (equinox J2000), as in vel_conversions.py.

Usage:
    python export_earth_orbit.py                          # 1 yr from 2013-06-22, 60 s step
    python export_earth_orbit.py --days 366 --dt 60 --out earth_orbit.mat
    python export_earth_orbit.py --lat 51.4779 --lon -0.0015 --height 45
    python export_earth_orbit.py --modules-dir /path/to/folder/with/vel_conversions.py
    python export_earth_orbit.py --out earth_orbit.csv    # CSV instead of .mat

Output (.mat) variables, N = number of time samples:
    t_s        N x 1   seconds since t0 (uniform SI seconds, TT scale)
    t_days     N x 1   days since t0
    jd_tt      N x 1   Julian date (TT)
    r_lab      N x 3   lab position relative to Sun          [km]
    v_lab      N x 3   lab velocity relative to Sun          [km/s]
    r_earth    N x 3   Earth-centre position relative to Sun [km]  (no site term)
    v_earth    N x 3   Earth-centre velocity relative to Sun [km/s]
    meta       struct  t0_iso, lat_deg, lon_deg, height_m, dt_s, frame, units
"""
import argparse
import os
import sys
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
from astropy.utils import iers

# Don't try to download IERS tables; the bundled ones are plenty here
# (Earth rotation is 0.46 km/s; the IERS corrections are ~1e-6 of that).
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "ignore"


def import_vel_conversions(modules_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (modules_dir, here, os.getcwd()):
        if p and os.path.isfile(os.path.join(p, "vel_conversions.py")):
            sys.path.insert(0, p)
            break
    try:
        from vel_conversions import sun_to_lab_position
    except ImportError as e:
        sys.exit("Could not import vel_conversions.py -- put it next to this script "
                 f"or pass --modules-dir.  ({e})")
    return sun_to_lab_position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="2013-06-22 00:00:00",
                    help="start time (UTC). Default = your notebook's t0")
    ap.add_argument("--days", type=float, default=366.0, help="span to tabulate [days]")
    ap.add_argument("--dt", type=float, default=60.0, help="time step [s]")
    ap.add_argument("--lat", type=float, default=51.4779)
    ap.add_argument("--lon", type=float, default=-0.0015)
    ap.add_argument("--height", type=float, default=45.0, help="[m]")
    ap.add_argument("--modules-dir", default=None,
                    help="folder containing vel_conversions.py (default: next to this script / cwd)")
    ap.add_argument("--chunk", type=int, default=100000,
                    help="samples per call to sun_to_lab_position (limits memory)")
    ap.add_argument("--out", default="earth_orbit.mat")
    args = ap.parse_args()

    sun_to_lab_position = import_vel_conversions(args.modules_dir)
    fmt = "csv" if args.out.lower().endswith(".csv") else "mat"

    t0 = Time(args.t0, scale="utc")
    n = int(np.floor(args.days * 86400.0 / args.dt)) + 1
    t_s = np.arange(n) * args.dt
    times = t0.tt + TimeDelta(t_s, format="sec")      # uniform grid, no leap-second surprises

    site = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg,
                         height=args.height * u.m)

    r_l = np.empty((n, 3)); v_l = np.empty((n, 3))
    r_e = np.empty((n, 3)); v_e = np.empty((n, 3))
    for i0 in range(0, n, args.chunk):
        sl = slice(i0, min(n, i0 + args.chunk))
        # lab = Earth centre + site offset (includes the daily-rotation velocity)
        r, _, v, _ = sun_to_lab_position(times[sl], site, frame="ecliptic")
        r_l[sl] = r.to_value(u.km).T
        v_l[sl] = v.to_value(u.km / u.s).T
        # Earth centre only (no site term) -- lets you switch the daily term off in MATLAB
        r, _, v, _ = sun_to_lab_position(times[sl], None, frame="ecliptic")
        r_e[sl] = r.to_value(u.km).T
        v_e[sl] = v.to_value(u.km / u.s).T
        print(f"  {sl.stop}/{n} samples", flush=True)

    meta = dict(t0_iso=args.t0 + " UTC", lat_deg=args.lat, lon_deg=args.lon,
                height_m=args.height, dt_s=args.dt,
                frame="HeliocentricMeanEcliptic (J2000), Sun-centred",
                units="km, km/s, s")

    if fmt == "mat":
        from scipy.io import savemat
        savemat(args.out, dict(t_s=t_s[:, None], t_days=(t_s / 86400.0)[:, None],
                               jd_tt=times.jd[:, None], r_lab=r_l, v_lab=v_l,
                               r_earth=r_e, v_earth=v_e, meta=meta),
                do_compression=True)
    else:
        hdr = ("t_s,t_days,jd_tt,rlab_x_km,rlab_y_km,rlab_z_km,"
               "vlab_x_kms,vlab_y_kms,vlab_z_kms,"
               "rearth_x_km,rearth_y_km,rearth_z_km,"
               "vearth_x_kms,vearth_y_kms,vearth_z_kms")
        tab = np.column_stack([t_s, t_s / 86400.0, times.jd, r_l, v_l, r_e, v_e])
        np.savetxt(args.out, tab, delimiter=",", header=hdr, comments="", fmt="%.10e")

    # ---- sanity printout ----------------------------------------------------
    d = np.linalg.norm(r_l, axis=1)
    sp = np.linalg.norm(v_e, axis=1)
    rot = np.linalg.norm(v_l - v_e, axis=1)
    print(f"wrote {args.out}  ({n} rows, dt = {args.dt} s, span = {args.days} d)")
    print(f"Sun-lab distance  : {d.min():.4e} .. {d.max():.4e} km  (perihelion ~1.471e8, aphelion ~1.521e8)")
    print(f"Earth-centre speed: {sp.min():.3f} .. {sp.max():.3f} km/s  (expect ~29.29 .. 30.29)")
    print(f"site rotation speed: {rot.min():.3f} .. {rot.max():.3f} km/s  "
          f"(0.4651*cos(lat) = {0.4651*np.cos(np.deg2rad(args.lat)):.3f})")
    zmax = np.abs(r_e[:, 2]).max()
    print(f"max |z| of Earth in ecliptic: {zmax:.0f} km  (should be << 1 AU; orbit lies in the ecliptic)")


if __name__ == "__main__":
    main()
