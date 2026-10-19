"""
stream_orbit_scan.py
====================
Local density distribution of fine-grained axion streams after solar focusing,
evaluated along the *real* orbit of the Earth (replaces the toy `earth_direction`
rotation in stream_enhancements.py).

Re-uses from your stream_enhancements.py:
    local_velocities, jacobian, stream_density, SIGMA
and from helper.py (via stream_enhancements): numStreams.

What the Monte Carlo does
-------------------------
The density of the n-th densest stream, rho_n, is FIXED by the skew-lognormal
(stream_density(n)).  What is random is the stream *velocity*, so each repeat
draws fresh solar-frame velocities  w ~ N(w_mean, sigma^2 I)  for every stream and
evaluates  A_n(t) = rho_local / rho_inf  at the Earth's position r(t).  Only the
position r(t) (direction AND distance) of the Earth enters the density; the
Earth's velocity does not.  (Velocity matters for the lineshape - next chapter.)

Frames
------
The orbit table (export_earth_orbit.py) is Sun-centred HeliocentricMeanEcliptic J2000.
The stream velocities must be in the SAME frame, so the mean wind is built from the
Sun's galactic velocity and rotated into the ecliptic (wind_mean_ecliptic).

Typical use
-----------
    python export_earth_orbit.py --dt 3600 --out earth_orbit.mat   # 1 hr is plenty here
    python stream_orbit_scan.py --orbit earth_orbit.mat            # small test run
    python stream_orbit_scan.py --orbit earth_orbit.mat --full     # thesis-quality run
    python stream_orbit_scan.py --astropy                          # no export file: astropy ephemeris
"""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import numStreams                                   # your skew-lognormal N(>rho)
from stream_enhancements import local_velocities, jacobian, stream_density, SIGMA

GM_SUN = 1.32712440018e11          # km^3 s^-2
AU_KM = 1.495978707e8              # km

# Sun's velocity in galactic Cartesian (U, V, W) [km/s].  CHANGE to match your analysis.
DEFAULT_VSUN_GAL = (11.1, 12.24 + 233.0, 7.25)

# ----------------------------------------------------------------- frames ---
_T_ICRS_TO_GAL = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                           [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                           [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])
_EPS = np.radians(23.4392911)      # J2000 mean obliquity
_R_ICRS_TO_ECL = np.array([[1, 0, 0],
                           [0, np.cos(_EPS), np.sin(_EPS)],
                           [0, -np.sin(_EPS), np.cos(_EPS)]])


def to_ecliptic(vec, frame="galactic"):
    """Rotate a vector (or (N,3) array) from `frame` in {'galactic','icrs','ecliptic'} to ecliptic J2000."""
    vec = np.asarray(vec, float)
    if frame == "ecliptic":
        return vec
    icrs = vec @ _T_ICRS_TO_GAL if frame == "galactic" else vec     # (T^T v) for rows == v @ T
    return icrs @ _R_ICRS_TO_ECL.T


def wind_mean_ecliptic(v_sun_gal=DEFAULT_VSUN_GAL):
    """Mean solar-frame stream velocity = -(Sun's galactic velocity), in ecliptic J2000."""
    return -to_ecliptic(v_sun_gal, "galactic")


# ------------------------------------------------------------------ orbit ---
def _jd_to_dt64(jd):
    sec = np.round((np.asarray(jd) - 2440587.5) * 86400.0).astype("int64")
    return np.datetime64("1970-01-01T00:00:00") + sec.astype("timedelta64[s]")


def _make_orbit(jd, r_km, v_kms, label):
    dist = np.linalg.norm(r_km, axis=1)
    return dict(jd=jd, t_days=jd - jd[0], dates=_jd_to_dt64(jd), r_km=r_km, v_kms=v_kms,
                dist_km=dist, rhat=r_km / dist[:, None], phi=GM_SUN / dist, label=label)


def load_orbit(path, step_days=1.0, use="lab"):
    """
    Read the table written by export_earth_orbit.py (.mat / .csv / .npz) and thin it to
    one row every `step_days`.  use='lab' -> r_lab, v_lab (includes site rotation),
    use='earth' -> Earth centre.  For the DENSITY the two are identical to ~1e-5.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        from scipy.io import loadmat
        m = loadmat(path)
        jd, t_days = m["jd_tt"].ravel(), m["t_days"].ravel()
        r, v = (m["r_lab"], m["v_lab"]) if use == "lab" else (m["r_earth"], m["v_earth"])
    elif ext == ".csv":
        tab = np.loadtxt(path, delimiter=",", skiprows=1)
        jd, t_days = tab[:, 2], tab[:, 1]
        a, b = (3, 6) if use == "lab" else (9, 12)
        r, v = tab[:, a:b], tab[:, a + 3:b + 3]
    elif ext == ".npz":
        m = np.load(path)
        jd, t_days = m["jd_tt"].ravel(), m["t_days"].ravel()
        r, v = (m["r_lab"], m["v_lab"]) if use == "lab" else (m["r_earth"], m["v_earth"])
    else:
        raise ValueError(f"unknown orbit file type: {ext}")
    dt = np.median(np.diff(t_days))
    stride = max(1, int(round(step_days / dt)))
    s = slice(None, None, stride)
    return _make_orbit(np.asarray(jd)[s], np.asarray(r)[s], np.asarray(v)[s], f"{os.path.basename(path)} ({use})")


def orbit_from_astropy(t0="2013-06-22 00:00:00", days=366.0, step_days=1.0):
    """
    Fallback if you don't want to go through vel_conversions: Earth-centre position relative to
    the Sun from astropy's built-in ephemeris, rotated into ecliptic J2000.
    (Good to ~arcmin, i.e. irrelevant for this chapter.)
    """
    import astropy.units as u
    from astropy.time import Time, TimeDelta
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    n = int(np.floor(days / step_days)) + 1
    t = Time(t0, scale="utc").tt + TimeDelta(np.arange(n) * step_days * 86400.0, format="sec")
    with solar_system_ephemeris.set("builtin"):
        pe, ve = get_body_barycentric_posvel("earth", t)
        ps, vs = get_body_barycentric_posvel("sun", t)
    r = (pe - ps).xyz.to_value(u.km).T @ _R_ICRS_TO_ECL.T
    v = (ve - vs).xyz.to_value(u.km / u.s).T @ _R_ICRS_TO_ECL.T
    return _make_orbit(t.jd, r, v, "astropy builtin ephemeris (Earth centre)")


def nearest_index(orbit, iso):
    return int(np.abs(orbit["dates"] - np.datetime64(iso)).argmin())


def _date_str(orbit, i, fmt="%d %b"):
    return orbit["dates"][i].astype("datetime64[s]").item().strftime(fmt)


# ------------------------------------------------------------ enhancement ---
def enhancement_branches(w, rhat, phi):
    """(A_direct, A_swung) = 1/|J| of the two local velocities that map to w. Shapes (N,)."""
    v_dir, v_swg = local_velocities(w, rhat, phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_d = 1.0 / np.abs(jacobian(v_dir, rhat, phi))
        a_s = 1.0 / np.abs(jacobian(v_swg, rhat, phi))
    return np.nan_to_num(a_d, nan=1.0, posinf=1e12), np.nan_to_num(a_s, nan=0.0, posinf=1e12)


# ------------------------------------------------------------ Monte Carlo ---
def scan(orbit, date_idx, w_mean, n_streams=10_000, n_repeats=100, sigma=SIGMA,
         branch="both", do_hist=True, n_bins_rho=40, seed=0, verbose=True,
         factors=(1.1, 1.5, 2.0, 5.0)):
    """
    For every repeat draw fresh stream velocities, then evaluate the enhancement at each orbit
    sample in `date_idx` (same draws for all dates -> differences between dates are purely
    orbital).  Returns a dict; T = len(date_idx), K = n_repeats:

      hist_old[B], bins_rho[B+1], lrho_max   original log10(rho) PDF and its end
      hist_new[T,K,B]                         PDF of log10(rho*A) per repeat (if do_hist)
      A_rho[T,K]      sum(rho A)/sum(rho): density-weighted enhancement of the stream population
      A_mean[T,K]     unweighted mean A (heavy tailed -> noisy; diagnostic only)
      rho_max_loc[T,K] densest stream's LOCAL density  max_n rho_n A_n   [units of rho_DM]
      n_star[T,K]     stream number (1 = densest) that attains it
      n_exceed[T,K,F] how many streams end up denser than  f * rho_1  for each f in `factors`
                      (rho_1 = original densest stream).  NB: A >= 1 always, so f = 1 is trivially
                      satisfied by stream 1 itself - hence factors > 1.
    """
    rng = np.random.default_rng(seed)
    date_idx = np.asarray(date_idx, int)
    T, K, N = len(date_idx), n_repeats, n_streams
    rho = stream_density(np.arange(1, N + 1))
    lrho = np.log10(rho)
    bins_rho = np.linspace(lrho.min() - 1e-9, lrho.max() + 1.5, n_bins_rho + 1)
    dx = np.diff(bins_rho)
    hist_old = np.histogram(lrho, bins_rho)[0] / (N * dx)

    hist_new = np.zeros((T, K, n_bins_rho)) if do_hist else None
    A_rho, A_mean = np.zeros((T, K)), np.zeros((T, K))
    rho_max_loc, n_star = np.zeros((T, K)), np.zeros((T, K), int)
    factors = np.asarray(factors, float)
    n_exceed = np.zeros((T, K, len(factors)), int)
    rhat, phi = orbit["rhat"][date_idx], orbit["phi"][date_idx]

    t0 = time.time()
    for k in range(K):
        w = rng.normal(0.0, sigma, (N, 3)) + np.asarray(w_mean)
        for j in range(T):
            a_d, a_s = enhancement_branches(w, rhat[j], phi[j])
            A = a_d + a_s if branch == "both" else a_d
            rl = rho * A
            if do_hist:
                # normalise by ALL streams (those boosted past the last bin still count)
                hist_new[j, k] = np.histogram(np.log10(rl), bins_rho)[0] / (N * dx)
            A_rho[j, k] = rl.sum() / rho.sum()
            A_mean[j, k] = A.mean()
            im = int(rl.argmax())
            rho_max_loc[j, k], n_star[j, k] = rl[im], im + 1
            n_exceed[j, k] = [(rl > f * rho[0]).sum() for f in factors]
        if verbose and (k + 1) % max(1, K // 10) == 0:
            print(f"  repeat {k + 1}/{K}   ({time.time() - t0:.1f} s)", flush=True)
    return dict(date_idx=date_idx, n_streams=N, n_repeats=K, branch=branch, w_mean=np.asarray(w_mean),
                sigma=sigma, rho1=rho[0], lrho_max=lrho.max(), bins_rho=bins_rho, hist_old=hist_old,
                hist_new=hist_new, A_rho=A_rho, A_mean=A_mean, rho_max_loc=rho_max_loc,
                n_star=n_star, n_exceed=n_exceed, factors=factors)


def _fidx(res, factor):
    i = np.where(np.isclose(res["factors"], factor))[0]
    if len(i) == 0:
        raise ValueError(f"factor {factor} not in res['factors'] = {res['factors']}")
    return int(i[0])


def peak_report(res, orbit, factor=2.0):
    """Print when each summary statistic peaks (median over repeats)."""
    d = res["date_idx"]
    p_exc = (res["n_exceed"][:, :, _fidx(res, factor)] >= 1).mean(1)
    stats = {"median A_rho (density-weighted enhancement)": np.median(res["A_rho"], 1),
             "median max local stream density / rho_1": np.median(res["rho_max_loc"], 1) / res["rho1"],
             f"P(some stream > {factor:g} x original densest)": p_exc}
    print("\nPeaks over the scanned dates:")
    for name, y in stats.items():
        if y.max() <= 0:
            print(f"  {name:52s}: never in this run (increase n_repeats / n_streams)")
            continue
        i = int(np.argmax(y))
        print(f"  {name:52s}: {_date_str(orbit, d[i], '%d %b')}   (value {y[i]:.5g}; "
              f"min {y.min():.5g} on {_date_str(orbit, d[int(np.argmin(y))])})")


# ------------------------------------------------------------------ plots ---
def _band_std(ax, x, mean, std, color, floor=1e-9):
    ax.fill_between(x, np.maximum(mean - 2 * std, floor), mean + 2 * std, color=color, alpha=0.20, lw=0)
    ax.fill_between(x, np.maximum(mean - std, floor), mean + std, color=color, alpha=0.45, lw=0)
    ax.plot(x, np.where(mean > 0, mean, np.nan), color=color, lw=1.6)


def _band_pct(ax, x, y, color, label=None):
    """y: (len(x), K) -> median with 68% / 95% bands over the repeats."""
    lo95, lo68, med, hi68, hi95 = np.percentile(y, [2.5, 16, 50, 84, 97.5], axis=1)
    ax.fill_between(x, lo95, hi95, color=color, alpha=0.20, lw=0)
    ax.fill_between(x, lo68, hi68, color=color, alpha=0.45, lw=0)
    ax.plot(x, med, color=color, lw=1.6, label=label)


def _c(i, n):
    return plt.cm.viridis(0.85 * i / max(n - 1, 1))


def _fit_cumulative(rho_lo=1e-18, rho_hi=1.0, n=20001):
    """
    Skewed log-normal fit of O'Hare & Pierobon (2025), built on Vogelsberger & White (2011).
      g(x) = skewnorm.pdf(x; -2.10, loc=-11.37, scale=7.15),  x = ln(rho/rho_DM)
             = mass fraction of rho_DM per unit ln(rho)
      dN/dln(rho) = g/rho   ->   N(>rho) = int_{ln rho}^inf g(x) e^{-x} dx
      F(>rho)     = int_{ln rho}^inf g(x) dx   (fraction of rho_DM in streams denser than rho)
    Returns rho, N(>rho), F(>rho) on a log grid.
    """
    from scipy.stats import skewnorm
    from scipy.integrate import cumulative_trapezoid
    x = np.linspace(np.log(rho_lo), 3.0, n)                    # upper limit 3 >> peak of g
    g = skewnorm.pdf(x, -2.10, loc=-11.37, scale=7.15)
    N = cumulative_trapezoid((g * np.exp(-x))[::-1], -x[::-1], initial=0)[::-1]
    F = skewnorm.sf(x, -2.10, loc=-11.37, scale=7.15)
    keep = np.exp(x) <= rho_hi
    return np.exp(x)[keep], N[keep], F[keep]


def _sci(x, d=1):
    """1234.5 -> '1.2\\times10^{3}' for use inside mathtext."""
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return rf"{m:.{d}f}\times10^{{{e}}}"


def plot_initial_distribution(n_streams=10_000, fname="plots/initial_stream_distribution.png", dpi=300,
                              overlay_helper=True):
    """
    Left : cumulative number of streams N(>rho) over the full range (~1e14 streams), with the cut at
           N_str = n_streams: streams above the cut are modelled individually, the rest is the smooth halo.
    Right: fraction of rho_DM carried by streams denser than rho, i.e. how much the N_str streams hold.
    overlay_helper: dashed line = your helper.numStreams, as a consistency check against the fit.
    """
    rho, N, F = _fit_cumulative()
    ok = N > 0
    x_cut = np.interp(np.log(n_streams), np.log(N[ok])[::-1], np.log(rho[ok])[::-1])   # N decreases with rho
    rho_cut = np.exp(x_cut)
    f_top = np.interp(np.log(rho_cut), np.log(rho), F)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.axvspan(rho[0], rho_cut, color="0.85", lw=0)
        ax.axvspan(rho_cut, rho[-1], color="C0", alpha=0.10, lw=0)
        ax.axvline(rho_cut, color="k", ls="--", lw=1)
        ax.set_xlim(rho[0], 1)
        ax.set_xlabel(r"stream density $\rho/\rho_{DM}$")
    a.set_yscale("log")
    a.plot(rho, N, color="C3", lw=2, label="skewed log-normal fit")
    if overlay_helper:
        try:
            g = np.geomspace(7.7e-8, 1, 400)
            Nh = np.asarray(numStreams(g), float)
            a.plot(g, Nh, "k--", lw=1, label=r"helper.numStreams (check)")
        except Exception as e:                                    # noqa: BLE001
            print("  (helper overlay skipped:", e, ")")
    a.axhline(n_streams, color="k", ls="--", lw=1)
    a.set_ylim(0.3, 1e15)
    a.set_ylabel(r"streams denser than $\rho$,  $N(>\rho)$")
    a.text(rho[0] * 2.5, 2.5e14, r"$\sim10^{14}$ streams", fontsize=11)
    a.text(rho[0] * 2.5, n_streams * 1.6, rf"cut: $N_{{\rm str}}={_sci(n_streams, 0)}$", fontsize=11)
    a.text(rho_cut * 1.6, 3, rf"modelled individually" "\n" rf"($\rho>{_sci(rho_cut)}\,\rho_{{DM}}$)",
           fontsize=10, color="C0")
    a.text(rho[0] * 2.5, 3, "smooth background", fontsize=10, color="0.35")
    a.legend(loc="upper right", fontsize=9)
    b.plot(rho, F, color="C3", lw=2)
    b.set_ylim(0, 1.02)
    b.set_ylabel(r"fraction of $\rho_{DM}$ in streams $>\rho$")
    b.annotate(rf"top $N_{{\rm str}}$ streams: {100 * f_top:.0f}% of $\rho_{{DM}}$", (rho_cut, f_top),
               (rho[0] * 30, 0.8), arrowprops=dict(arrowstyle="->"), fontsize=11)
    b.text(rho[0] * 30, 0.62, rf"rest ({100 * (1 - f_top):.0f}%) $\to$ smooth SHM", fontsize=11)
    print(f"  N_str = {n_streams:.0e}: rho_cut = {rho_cut:.3e} rho_DM, top streams hold {100 * f_top:.1f}% of rho_DM")
    fig.savefig(fname, dpi=dpi)
    return fig


def plot_wind_geometry(orbit, w_mean, sigma=SIGMA, fname="plots/wind_geometry.png", dpi=300, n_draw=200_000):
    """
    Left : ecliptic latitude of stream directions (the Earth's orbit lies at beta = 0) and of the mean wind.
    Right: angle between r_hat(t) and the mean-wind direction through the year, and the enhancement A(t)
           of one stream moving exactly with the mean wind (Chapter-3 result on the real orbit).
    """
    rng = np.random.default_rng(0)
    w = rng.normal(0, sigma, (n_draw, 3)) + w_mean
    beta = np.degrees(np.arcsin(w[:, 2] / np.linalg.norm(w, axis=1)))
    wb = np.degrees(np.arcsin(w_mean[2] / np.linalg.norm(w_mean)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    a.hist(beta, 90, range=(-90, 90), density=True, color="C0", alpha=0.7)
    a.axvline(0, color="k", ls="--", label="Earth's orbital plane")
    a.axvline(wb, color="C3", label=rf"mean wind ($\beta$ = {wb:.0f}°)")
    a.set_xlabel(r"ecliptic latitude of stream direction $\beta$ [deg]")
    a.set_ylabel("PDF")
    a.legend()
    dates = orbit["dates"]
    wh = w_mean / np.linalg.norm(w_mean)
    ang = np.degrees(np.arccos(np.clip(orbit["rhat"] @ wh, -1, 1)))
    A = np.array([sum(enhancement_branches(w_mean[None, :], orbit["rhat"][i], orbit["phi"][i])) [0]
                  for i in range(len(dates))])
    b.plot(dates, ang, color="C0")
    b.set_ylabel(r"angle($\hat r$, mean wind) [deg]", color="C0")
    b2 = b.twinx()
    b2.plot(dates, A, color="C3")
    b2.set_ylabel(r"$A$ for a stream with $\mathbf{w}=\bar{\mathbf{w}}$", color="C3")
    b.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.savefig(fname, dpi=dpi)
    return fig


def plot_logrho_epochs(res, orbit, fname="plots/logrho_pdf_epochs.png", dpi=300, factor=2.0):
    """Six-panel (or J-panel) version of your log-rho figure, one panel per Earth position/date."""
    J = len(res["date_idx"])
    ncol = 3 if J > 4 else 2
    nrow = int(np.ceil(J / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.8 * nrow), sharex=True, sharey=True,
                            constrained_layout=True)
    axs = np.atleast_2d(axs)
    for ax in axs.ravel()[J:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(r"$\log_{10}(\rho / \rho_{DM})$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$P(\log_{10}(\rho / \rho_{DM}))$")
    b = res["bins_rho"]
    x = 0.5 * (b[1:] + b[:-1])
    h0 = res["hist_old"]
    flat = axs.ravel()[:J]
    for j, ax in enumerate(flat):
        col = _c(j, J)
        h = res["hist_new"][j]
        _band_std(ax, x, h.mean(0), h.std(0, ddof=1) if h.shape[0] > 1 else 0 * x, col)
        ax.plot([], [], color=col, lw=1.6, label=r"with focusing (mean, $\pm1\sigma,\pm2\sigma$)")
        ax.plot(x, np.where(h0 > 0, h0, np.nan), "k", marker="+", lw=0.8, label="original", zorder=5)
        ax.axvline(res["lrho_max"], color="k", ls=":", lw=1.2, label="end of original distribution", zorder=4)
        p = (res["n_exceed"][j][:, _fidx(res, factor)] >= 1).mean()
        ax.text(0.04, 0.05, rf"P($\geq$1 stream $>{factor:g}\,\rho_1$) = {p:.2f}"
                + "\n" + rf"$\langle A\rangle_\rho$ = {np.median(res['A_rho'][j]):.3f}",
                transform=ax.transAxes, fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3 * h0.max())
        ax.set_xlim(b[0], res["lrho_max"] + 1.2)
        ax.set_title(_date_str(orbit, res["date_idx"][j], "%d %b"))
    flat[ncol - 1].legend(loc="upper right", fontsize=10)
    fig.suptitle("Stream density distribution, before/after solar focusing (real Earth orbit)")
    fig.savefig(fname, dpi=dpi)
    return fig


def plot_year(res, orbit, fname="plots/density_vs_date.png", marks=None, dpi=300):
    """
    Annual evolution (median and 68%/95% bands over velocity draws):
      (a) density-weighted enhancement of the whole stream population  <A>_rho
      (b) local density of the densest stream relative to the original densest stream
      (c) probability that at least one stream ends up denser than the original densest one
    `marks`: optional list of orbit indices (e.g. the six epochs) drawn as dotted lines.
    """
    d = res["date_idx"]
    dates = orbit["dates"][d]
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True, constrained_layout=True)
    _band_pct(a, dates, res["A_rho"], "C0")
    a.set_ylabel(r"$\langle A\rangle_\rho = \sum\rho_n A_n / \sum \rho_n$")
    _band_pct(b, dates, np.maximum(res["rho_max_loc"] / res["rho1"] - 1, 1e-8), "C1")
    b.set_yscale("log")
    b.set_ylabel(r"$\max_n\,\rho_n A_n\,/\,\rho_{1} - 1$")
    for fi, f in enumerate(res["factors"]):
        c.plot(dates, (res["n_exceed"][:, :, fi] >= 1).mean(1), lw=1.6, color=plt.cm.plasma(0.15 + 0.7 * fi / max(len(res["factors"]) - 1, 1)),
               label=rf"$f={f:g}$")
    c.set_ylabel(r"$P(\geq 1$ stream $> f\,\rho_1)$")
    c.set_ylim(-0.02, 1.02)
    c.legend(title="threshold", ncol=2)
    for ax in (a, b, c):
        if marks is not None:
            for m in marks:
                ax.axvline(orbit["dates"][m], color="gray", ls=":", lw=1)
    c.xaxis.set_major_locator(mdates.MonthLocator())
    c.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle(f"{res['n_streams']:,} streams, {res['n_repeats']} velocity draws")
    fig.savefig(fname, dpi=dpi)
    return fig


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit", default=None, help="file from export_earth_orbit.py (.mat/.csv/.npz)")
    ap.add_argument("--astropy", action="store_true", help="use astropy's built-in ephemeris instead")
    ap.add_argument("--t0", default="2013-06-22 00:00:00", help="start time for --astropy")
    ap.add_argument("--use", default="lab", choices=["lab", "earth"])
    ap.add_argument("--nstreams", type=int, default=100000)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--scan-step", type=float, default=15.0, help="days between dates in the year scan")
    ap.add_argument("--epochs", default=None, help="comma-separated ISO dates for the 6-panel figure")
    ap.add_argument("--wind-vector", type=float, nargs=3, default=None,
                    help="mean stream velocity; if omitted: -(Sun's galactic velocity) rotated to ecliptic")
    ap.add_argument("--wind-frame", default="ecliptic", choices=["ecliptic", "galactic", "icrs"],
                    help="frame of --wind-vector")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--full", action="store_true", help="thesis-size run: 10^4 streams, 500 draws, daily scan")
    args = ap.parse_args()
    if args.full:
        args.nstreams, args.repeats, args.scan_step = 10_000, 500, 1.0
    os.makedirs(args.out_dir, exist_ok=True)

    if args.orbit:
        orbit = load_orbit(args.orbit, step_days=1.0, use=args.use)
    else:
        orbit = orbit_from_astropy(args.t0, 366.0, 1.0)
    print("orbit:", orbit["label"], f"| {len(orbit['jd'])} samples | r = {orbit['dist_km'].min() / AU_KM:.4f}"
          f"..{orbit['dist_km'].max() / AU_KM:.4f} AU")
    w_mean = (to_ecliptic(np.array(args.wind_vector), args.wind_frame) if args.wind_vector
              else wind_mean_ecliptic())
    print("mean wind (ecliptic J2000) [km/s]:", np.round(w_mean, 1), " |w| =", round(float(np.linalg.norm(w_mean)), 1))

    year = orbit["t_days"] <= 365.25
    n_year = int(year.sum())
    # --- year scan (no histograms) ---
    targets = np.arange(0, 365.25 + 1e-9, args.scan_step)
    yidx = np.unique(np.clip(np.searchsorted(orbit["t_days"], targets), 0, n_year - 1))
    print(f"\nyear scan: {len(yidx)} dates x {args.repeats} draws x {args.nstreams} streams")
    t0 = time.time()
    yres = scan(orbit, yidx, w_mean, args.nstreams, args.repeats, args.sigma, do_hist=False)
    print(f"  done in {time.time() - t0:.1f} s")
    peak_report(yres, orbit)

    # --- six epochs (with histograms) ---
    if args.epochs:
        eidx = [nearest_index(orbit, s) for s in args.epochs.split(",")]
    else:
        eidx = [int(np.abs(orbit["t_days"] - 365.25 * k / 6).argmin()) for k in range(6)]
    print("\nepochs:", [_date_str(orbit, i, "%Y-%m-%d") for i in eidx])
    eres = scan(orbit, eidx, w_mean, args.nstreams, args.repeats, args.sigma, do_hist=True)

    out = lambda s: os.path.join(args.out_dir, s)
    plot_initial_distribution(args.nstreams, out("initial_stream_distribution.png"), args.dpi)
    plot_wind_geometry(orbit, w_mean, args.sigma, out("wind_geometry.png"), args.dpi)
    plot_logrho_epochs(eres, orbit, out("logrho_pdf_epochs.png"), args.dpi)
    plot_year(yres, orbit, out("density_vs_date.png"), marks=eidx, dpi=args.dpi)
    print("saved figures to", args.out_dir)


if __name__ == "__main__":
    main()