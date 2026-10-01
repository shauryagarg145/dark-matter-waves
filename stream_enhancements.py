"""
Solar gravitational-focusing enhancement of fine-grained dark-matter streams,
repeated over many velocity draws and several Earth positions.

Replaces the fsolve loop in densityDiff.ipynb with an exact analytic inverse of
the v -> v_inf map, so everything is vectorised.

Definitions
-----------
w   : stream velocity at infinity (solar frame)            [km/s]
v   : local velocity at Earth that maps to w               [km/s]
rho_local = rho_inf * sum_branches 1/|det(dv_inf/dv)|
A         = rho_local / rho_inf              (your new_dens/dens)
delta     = A - 1                            (relative density increase)

For a given w there are TWO local velocities at Earth (direct, and a branch
that has swung round the Sun). The notebook's fsolve keeps only the J>0
(direct) one; branch="both" (default) sums them, branch="direct" reproduces
the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
from helper import numStreams                  # skew-lognormal dN/drho

PHI = 887.13                                   # G M_sun / 1 AU  [km^2/s^2]
V_C = np.array([218.0, -119.0, 21.0])          # stream-mean velocity (notebook)
SIGMA = 167.0                                  # km/s

import matplotlib
#from pyfonts import load_google_font

font = {'family' : 'EB Garamond',
        'weight' : 'normal',
        'size'   : 22}
#matplotlib.rcParams['font.serif'] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "cm"

matplotlib.rc('font', **font)

SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 23

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# ---------------------------------------------------------------- physics ---
def earth_direction(theta):
    """Unit vector Sun->Earth, same convention as the notebook's v_inf()."""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def local_velocities(w, rhat, phi=PHI):
    """
    Both local velocities v (at position rhat, |r| = 1 AU) whose Keplerian
    trajectory has asymptotic velocity w. Exact, vectorised over w (N,3).
    Returns (v_direct, v_swung), each (N,3).

    The orbit lies in the plane spanned by w and rhat; solving r(psi) = R for
    the impact parameter gives a quadratic in beta = b u^2 / (G M) with exactly
    one positive root per sense of rotation.
    """
    u = np.linalg.norm(w, axis=1)
    e1 = -w / u[:, None]                           # direction of position at t -> -inf
    c0 = np.clip(e1 @ rhat, -1, 1)                 # cos(psi0)
    e2 = rhat[None, :] - c0[:, None] * e1
    s0 = np.linalg.norm(e2, axis=1)
    e2 /= np.maximum(s0, 1e-300)[:, None]
    eps = phi / u**2
    psi0 = np.arccos(c0)
    out = []
    for sense in (+1, -1):
        psi = psi0 if sense > 0 else 2 * np.pi - psi0
        sp, cp = np.sin(psi), np.cos(psi)
        beta = (sp + np.sqrt(sp**2 + 4 * eps * (1 - cp))) / (2 * eps)
        v_r = -(u / beta) * (sp + beta * cp)
        v_t = beta * eps * u
        that = (-s0[:, None] * e1 + c0[:, None] * e2) if sense > 0 \
               else (s0[:, None] * e1 - c0[:, None] * e2)
        out.append(v_r[:, None] * rhat[None, :] + v_t[:, None] * that)
    return out[0], out[1]


def jacobian(v, rhat, phi=PHI):
    """det(dv_inf/dv) = 1 - (Phi/D)^2, D = u^2 + Phi - u (v . rhat)  [signed v.rhat]."""
    u = np.sqrt(np.sum(v * v, axis=1) - 2 * phi)
    D = u**2 + phi - u * (v @ rhat)
    return 1 - (phi / D) ** 2


def enhancement(w, rhat, branch="both"):
    """A = rho_local / rho_inf for each stream."""
    v_dir, v_swg = local_velocities(w, rhat)
    A = 1.0 / np.abs(jacobian(v_dir, rhat))
    if branch == "both":
        A = A + 1.0 / np.abs(jacobian(v_swg, rhat))
    return A


# --------------------------------------------------------- stream densities ---
def _build_density_inverse(rho_min=7.7e-8, rho_max=1.0, n_grid=200_000):
    g = np.geomspace(rho_min, rho_max, n_grid)
    f = numStreams(g)
    assert np.all(np.diff(f) < 0), "numStreams must be monotone decreasing for the inverse to be unique"
    return np.log(f)[::-1], np.log(g)[::-1]
_LF, _LG = _build_density_inverse()


def stream_density(n):
    """
    Density of stream number n, i.e. the rho solving numStreams(rho) = n.
    Same definition as helper.get_density but solved by exact log-log
    interpolation (vectorised); helper.get_density's optimiser stops
    converging for n >~ 3e3.
    """
    return np.exp(np.interp(np.log(np.asarray(n, float)), _LF, _LG))


# ------------------------------------------------------------- experiment ---
def run(n_streams=100_000, n_repeats=100, thetas=None, branch="both",
        bins=np.linspace(-7, 2, 91), n_keep=800, n_bins_rho=40, seed=0):
    """
    For each repeat, draw fresh stream velocities and evaluate A at every Earth
    angle (same draws for all angles, so panel differences are purely temporal).

    Returns a dict with (T = angles, K = repeats):
      hist_new[T, K, B] : P(log10 rho_new) per repeat (density=True, as in your cell 5)
      hist_old[B]       : P(log10 rho) of the original densities
      bins_rho[B+1]     : shared bin edges for the two above
      hist[T, K, nb]    : P(log10 delta) per repeat, on `bins`
      mean_A[T, K]      : mean enhancement per repeat
      A_keep[T, K, M]   : A for M log-spaced streams, stream numbers idx + 1
    """
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    rng = np.random.default_rng(seed)
    T, K = len(thetas), n_repeats
    rhats = [earth_direction(t) for t in thetas]

    rho = stream_density(np.arange(1, n_streams + 1))          # original stream densities
    lrho = np.log10(rho)
    bins_rho = np.linspace(lrho.min() - 1e-9, lrho.max() + 1.5, n_bins_rho + 1)  # room for x30 boosts
    hist_old = np.histogram(lrho, bins_rho, density=True)[0]
    idx = np.unique(np.geomspace(1, n_streams, n_keep).astype(int) - 1)

    hist_new = np.zeros((T, K, n_bins_rho))
    hist = np.zeros((T, K, len(bins) - 1))
    mean_A = np.zeros((T, K))
    A_keep = np.zeros((T, K, len(idx)))
    for k in range(K):
        w = rng.normal(0, SIGMA, (n_streams, 3)) + V_C
        for i, rh in enumerate(rhats):
            A = enhancement(w, rh, branch)
            hist_new[i, k] = np.histogram(np.log10(rho * A), bins_rho, density=True)[0]
            hist[i, k] = np.histogram(np.log10(np.maximum(A - 1, 1e-300)), bins)[0]
            mean_A[i, k] = A.mean()
            A_keep[i, k] = A[idx]
    hist = hist / (n_streams * np.diff(bins))                  # PDF in log10(delta)
    return dict(thetas=thetas, branch=branch, bins_rho=bins_rho, hist_new=hist_new,
                hist_old=hist_old, bins=bins, hist=hist, mean_A=mean_A, A_keep=A_keep, idx=idx)


# ------------------------------------------------------------------ plots ---
def _color(i, T):
    return plt.cm.viridis(0.85 * i / max(T - 1, 1))


def _grid(T, xlabel, ylabel, width=4.8, height=3.8):
    """Grid of single panels (3 columns if T > 4); unused panels hidden. Returns fig, flat axes[:T]."""
    ncol = 3 if T > 4 else 2
    nrow = int(np.ceil(T / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(width * ncol, height * nrow),
                            sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_2d(axs)
    for ax in axs.ravel()[T:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    return fig, axs.ravel()[:T]


def _band(ax, x, y, color):
    """y: (K, len(x)) -> median line with 68% and 95% bands over the repeats."""
    lo95, lo68, med, hi68, hi95 = np.percentile(y, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(x, lo95, hi95, color=color, alpha=0.20, lw=0)
    ax.fill_between(x, lo68, hi68, color=color, alpha=0.45, lw=0)
    ax.plot(x, med, color=color, lw=1.6)


def _band_std(ax, x, mean, std, color, floor=1e-9):
    """Mean line with 1-sigma / 2-sigma bands (lower edges clipped at `floor` for log axes)."""
    ax.fill_between(x, np.maximum(mean - 2 * std, floor), mean + 2 * std, color=color, alpha=0.20, lw=0)
    ax.fill_between(x, np.maximum(mean - std, floor), mean + std, color=color, alpha=0.45, lw=0)
    ax.plot(x, np.where(mean > 0, mean, np.nan), color=color, lw=1.6)


def plot_logrho_pdf(res, fname="logrho_pdf_bands.png"):
    """
    Your cell-5 plot, repeated over velocity resamples and Earth angles.
    P(log10 rho) of all streams from all resamples pooled (= mean of the per-resample
    histograms) with +-1 / +-2 sigma bands (std of the per-resample histograms);
    the original distribution is the thin black line on top.
    """
    th, b = res["thetas"], res["bins_rho"]
    T, K = len(th), res["hist_new"].shape[1]
    x = 0.5 * (b[1:] + b[:-1])
    h0 = res["hist_old"]
    fig, axs = _grid(T, r"$\log_{10}\rho$", r"$P(\log\rho)$")
    for i, ax in enumerate(axs):
        col = _color(i, T)
        _band_std(ax, x, res["hist_new"][i].mean(axis=0), res["hist_new"][i].std(axis=0, ddof=1), col)
        ax.plot([], [], color=col, lw=1.6, label=r"with focusing (mean, $\pm1\sigma,\pm2\sigma$)")
        ax.plot(x, np.where(h0 > 0, h0, np.nan), "k", lw=0.8, label="original", zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3 * h0.max())
        ax.set_xlim(b[0], b[-1] - 0.5)                         # original max + 1 decade
        ax.set_title(f"$\\theta$ = {np.degrees(th[i]):.0f}°")
    axs[0].legend(loc="lower left")
    fig.suptitle(f"Stream density distribution, before/after solar focusing")
    fig.savefig(fname, dpi=400)
    return fig


def plot_delta_pdf(res, fname="delta_pdf_bands.png"):
    """P(log10 delta) per Earth angle; band = spread over the repeated velocity draws."""
    th, bins = res["thetas"], res["bins"]
    T, K = len(th), res["hist"].shape[1]
    x = 0.5 * (bins[1:] + bins[:-1])
    fig, axs = _grid(T, r"$\log_{10}\,(\rho_{\rm new}/\rho_{\rm old}-1)$", r"$P(\log_{10}\delta)$", 4.6, 3.6)
    for i, ax in enumerate(axs):
        _band(ax, x, res["hist"][i] + 1e-12, _color(i, T))
        lo, md, hi = np.percentile(res["mean_A"][i], [16, 50, 84])
        ax.set_title(f"$\\theta$ = {np.degrees(th[i]):.0f}°  ({th[i] / 2 / np.pi:.2f} yr)")
        ax.text(0.04, 0.05, f"$\\langle A\\rangle$ = {md:.4f}  [{lo:.4f}, {hi:.4f}]",
                transform=ax.transAxes, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3)
        ax.set_xlim(bins[0], bins[-1])
    fig.suptitle(f"Relative density increase over streams  (bands: 68% / 95% over "
                 f"{K} velocity resamples, branch = {res['branch']})")
    fig.savefig(fname, dpi=400)
    return fig


def plot_densities(res, fname="densities_vs_stream.png"):
    """
    Detectable stream densities rho_n * A_n vs stream number n.
    Top: original density (black) and enhanced density (median, 68% / 95% bands over
    resamples). Bottom: the fractional increase, so the few-percent effect is visible.
    """
    th = res["thetas"]
    T, K = len(th), res["A_keep"].shape[1]
    ncol = 3 if T > 4 else 2
    nrow = int(np.ceil(T / ncol))
    n = res["idx"] + 1
    rho0 = stream_density(n)
    fig = plt.figure(figsize=(4.8 * ncol, 4.6 * nrow), constrained_layout=True)
    for i, sf in enumerate(np.ravel(fig.subfigures(nrow, ncol))[:T]):
        col = _color(i, T)
        top, bot = sf.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
        A = res["A_keep"][i]                                   # (K, M)
        _band(top, n, rho0[None, :] * A, col)
        top.loglog(n, rho0, "k", lw=0.9, label="original", zorder=5)
        _band(bot, n, np.maximum(A - 1, 1e-12), col)
        bot.set_xscale("log")
        bot.set_yscale("log")
        bot.set_ylim(1e-6, 1e2)
        top.set_title(f"$\\theta$ = {np.degrees(th[i]):.0f}°")
        top.set_ylabel(r"stream density $\rho$")
        bot.set_ylabel(r"$\rho_{\rm new}/\rho-1$")
        bot.set_xlabel("stream number")
        if i == 0:
            top.legend()
    fig.suptitle(f"Detectable stream densities (bands: 68% / 95%)")
    fig.savefig(fname, dpi=400)
    return fig


# --------------------------------------------------------------- self-test ---
def _selftest():
    """Round trip through the notebook's forward map; A >= 1; A_both = 2 A_direct - 1."""
    rng = np.random.default_rng(1)
    w = rng.normal(0, SIGMA, (5000, 3)) + V_C
    for th in (0.0, 1.3, 4.0):
        rh = earth_direction(th)
        for v in local_velocities(w, rh):
            u = np.sqrt(np.sum(v * v, 1) - 2 * PHI)
            D = u**2 + PHI - u * (v @ rh)
            w_fwd = ((u**2)[:, None] * v + (u * PHI)[:, None] * rh - (u * (v @ rh))[:, None] * v) / D[:, None]
            assert np.abs(w_fwd - w).max() < 1e-4, "inverse failed round trip"
        A_dir, A_both = enhancement(w, rh, "direct"), enhancement(w, rh, "both")
        assert A_dir.min() >= 1 and A_both.min() >= 1, "focusing lowered a density"
        assert np.allclose(A_both, 2 * A_dir - 1, rtol=1e-6, atol=1e-8), "identity A_both = 2 A_direct - 1"
    print("self-test passed")


if __name__ == "__main__":
    import time
    _selftest()
    t0 = time.time()
    res = run(n_streams=10000, n_repeats=10000, branch="both")
    print(f"run time: {time.time() - t0:.1f} s")
    plot_logrho_pdf(res)
    plot_densities(res)
    plot_delta_pdf(res)                        # optional extra figures
    plt.show()