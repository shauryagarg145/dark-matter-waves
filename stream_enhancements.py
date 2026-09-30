"""
Solar gravitational-focusing enhancement of fine-grained dark-matter streams,
repeated over many velocity draws and several Earth positions.

Replaces the fsolve loop in densityDiff.ipynb with an exact analytic inverse of
the v -> v_inf map, so everything is vectorised (100 repeats x 6 times x 1e5
streams runs in well under a minute).

Definitions
-----------
w   : stream velocity at infinity (solar frame)            [km/s]
v   : local velocity at Earth that maps to w               [km/s]
rho_local = rho_inf * sum_branches 1/|det(dv_inf/dv)|
A         = rho_local / rho_inf              (your new_dens/dens)
delta     = A - 1                            (relative density *increase*)

For a given w there are TWO local velocities at Earth (direct, and a branch
that has swung round the Sun). The notebook's fsolve keeps only the J>0
(direct) one; branch="both" (default) sums them, branch="direct" reproduces
the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
from helper import numStreams                  # your skew-lognormal dN/drho

PHI = 887.13                                   # G M_sun / 1 AU  [km^2/s^2]
V_C = np.array([218.0, -119.0, 21.0])          # stream-mean velocity (notebook)
SIGMA = 167.0                                  # km/s


# ---------------------------------------------------------------- physics ---
def earth_direction(theta):
    """Unit vector Sun->Earth, same convention as the notebook's v_inf()."""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def local_velocities(w, rhat, phi=PHI):
    """
    Both local velocities v (at position rhat, |r| = 1 AU) whose Keplerian
    trajectory has asymptotic velocity w. Exact, vectorised over w (N,3).
    Returns (v_direct, v_swung), each (N,3).

    Orbit lies in the plane spanned by w and rhat; solving r(psi)=R for the
    impact parameter gives a quadratic in beta = b u^2 / (G M) with exactly one
    positive root per sense of rotation.
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
    interpolation (vectorised). helper.get_density's optimiser stops converging
    for n >~ 3e3 (e.g. misses the target by ~50% at n=1.4e4).
    """
    return np.exp(np.interp(np.log(np.asarray(n, float)), _LF, _LG))


# ------------------------------------------------------------- experiment ---
def run(n_streams=100_000, n_repeats=100, thetas=None, branch="both",
        bins=np.linspace(-7, 2, 91), n_keep=800, seed=0):
    """
    For each repeat, draw fresh stream velocities and evaluate A at every Earth
    angle (same draws for all angles, so panel differences are purely temporal).

    Returns dict with
      hist[T, K, nbins]  : PDF of log10(delta) per repeat
      mean_A[T, K]       : mean enhancement per repeat
      A_keep[T, K, M]    : A for M log-spaced streams (stream numbers n_keep_idx+1)
    """
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    rng = np.random.default_rng(seed)
    T, K = len(thetas), n_repeats
    hist = np.zeros((T, K, len(bins) - 1))
    mean_A = np.zeros((T, K))
    idx = np.unique(np.geomspace(1, n_streams, n_keep).astype(int) - 1)
    A_keep = np.zeros((T, K, len(idx)))
    rhats = [earth_direction(t) for t in thetas]
    clipped = 0.0
    for k in range(K):
        w = rng.normal(0, SIGMA, (n_streams, 3)) + V_C
        for i, rh in enumerate(rhats):
            A = enhancement(w, rh, branch)
            d = np.log10(np.maximum(A - 1, 1e-300))
            hist[i, k] = np.histogram(d, bins, density=False)[0]
            clipped += np.mean((d < bins[0]) | (d >= bins[-1]))
            mean_A[i, k] = A.mean()
            A_keep[i, k] = A[idx]
    hist = hist / (n_streams * np.diff(bins))          # PDF in log10(delta)
    print(f"fraction of streams outside histogram range: {clipped / (T * K):.2e}")
    return dict(thetas=thetas, bins=bins, hist=hist, mean_A=mean_A, A_keep=A_keep, idx=idx, branch=branch)


# ------------------------------------------------------------------ plots ---
def _band(ax, x, y, color, label=None):
    """y: (K, len(x)) -> median line + 68% and 95% bands over repeats."""
    lo95, lo68, med, hi68, hi95 = np.percentile(y, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(x, lo95, hi95, color=color, alpha=0.20, lw=0)
    ax.fill_between(x, lo68, hi68, color=color, alpha=0.45, lw=0)
    ax.plot(x, med, color=color, lw=1.6, label=label)


def plot_delta_pdf(res, fname="delta_pdf_bands.png"):
    """P(log10 delta) per Earth angle, band = spread over repeated velocity draws."""
    th, bins = res["thetas"], res["bins"]
    x = 0.5 * (bins[1:] + bins[:-1])
    T = len(th)
    ncol = 3 if T > 4 else 2
    fig, axs = plt.subplots(int(np.ceil(T / ncol)), ncol, figsize=(4.6 * ncol, 3.6 * np.ceil(T / ncol)),
                            sharex=True, sharey=True, constrained_layout=True)
    cmap = plt.cm.viridis
    for i, ax in enumerate(np.ravel(axs)[:T]):
        col = cmap(0.85 * i / max(T - 1, 1))
        _band(ax, x, np.nan_to_num(res["hist"][i], nan=0.0) + 1e-12, col)
        m = res["mean_A"][i]
        lo, md, hi = np.percentile(m, [16, 50, 84])
        ax.set_title(f"$\\theta$ = {np.degrees(th[i]):.0f}°  ({th[i] / 2 / np.pi:.2f} yr)", fontsize=10)
        ax.text(0.04, 0.05, f"$\\langle A\\rangle$ = {md:.4f}  [{lo:.4f}, {hi:.4f}]",
                transform=ax.transAxes, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3)
        ax.set_xlim(bins[0], bins[-1])
    for ax in np.ravel(axs)[T:]:
        ax.axis("off")
    for ax in np.atleast_2d(axs)[-1]:
        ax.set_xlabel(r"$\log_{10}\,(\rho_{\rm new}/\rho_{\rm old}-1)$")
    for ax in np.atleast_2d(axs)[:, 0]:
        ax.set_ylabel(r"$P(\log_{10}\delta)$")
    fig.suptitle(f"Relative density increase over streams  (bands: 68% / 95% over "
                 f"{res['hist'].shape[1]} velocity resamples, branch = {res['branch']})", fontsize=11)
    fig.savefig(fname, dpi=400)
    return fig


def plot_densities(res, fname="densities_vs_stream.png"):
    """
    Actual detectable stream densities rho_new = rho_n * A_n vs stream number n.
    Top of each panel: original density (black) and the enhanced density
    (median, 68%, 95% bands over the velocity resamples; dotted = max over resamples).
    Bottom: same thing as fractional increase, so the few-percent effect is visible.
    """
    th = res["thetas"]; T = len(th); ncol = 3 if T > 4 else 2; nrow = int(np.ceil(T / ncol))
    n = res["idx"] + 1
    rho0 = stream_density(n)
    fig = plt.figure(figsize=(4.8 * ncol, 4.6 * nrow), constrained_layout=True)
    subs = np.ravel(fig.subfigures(nrow, ncol))
    for i, sf in enumerate(subs[:T]):
        col = plt.cm.viridis(0.85 * i / max(T - 1, 1))
        top, bot = sf.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
        A = res["A_keep"][i]
        _band(top, n, rho0[None, :] * A, col)
        top.loglog(n, rho0, "k", lw=0.9, label="original")
        top.loglog(n, rho0 * A.max(axis=0), ":", color=col, lw=0.8, label="max over resamples")
        _band(bot, n, np.maximum(A - 1, 1e-12), col)
        bot.plot(n, np.maximum(A.max(axis=0) - 1, 1e-12), ":", color=col, lw=0.8)
        top.set_xscale("log"); top.set_yscale("log"); bot.set_yscale("log")
        bot.set_ylim(1e-6, 1e2)
        top.set_title(f"$\\theta$ = {np.degrees(th[i]):.0f}°", fontsize=10)
        top.set_ylabel(r"stream density $\rho$")
        bot.set_ylabel(r"$\rho_{\rm new}/\rho-1$"); bot.set_xlabel("stream number")
        if i == 0: top.legend(fontsize=7)
    for sf in subs[T:]:
        sf.set_visible(False)
    fig.suptitle(f"Detectable stream densities (bands: 68% / 95% over {A.shape[0]} velocity resamples, "
                 f"branch = {res['branch']})", fontsize=11)
    fig.savefig(fname, dpi=400)
    return fig


# --------------------------------------------------------------- self-test ---
def _selftest():
    """Round-trip through the notebook's forward map + identity check."""
    rng = np.random.default_rng(1)
    w = rng.normal(0, SIGMA, (5000, 3)) + V_C
    for th in (0.0, 1.3, 4.0):
        rh = earth_direction(th)
        for v in local_velocities(w, rh):
            u = np.sqrt(np.sum(v * v, 1) - 2 * PHI)
            D = u**2 + PHI - u * (v @ rh)
            wf = ((u**2)[:, None] * v + (u * PHI)[:, None] * rh - (u * (v @ rh))[:, None] * v) / D[:, None]
            assert np.abs(wf - w).max() < 1e-4, "inverse failed round trip"
        Ad, Ab = enhancement(w, rh, "direct"), enhancement(w, rh, "both")
        assert np.allclose(Ab, 2 * Ad - 1, rtol=1e-6, atol=1e-8), "identity A_both = 2 A_direct - 1"
    print("self-test passed")


if __name__ == "__main__":
    import time
    _selftest()
    t0 = time.time()
    res = run(n_streams=100_000, n_repeats=100, branch="both")
    print(f"run time: {time.time() - t0:.1f} s")
    plot_densities(res)
    plot_delta_pdf(res)

    plt.show()