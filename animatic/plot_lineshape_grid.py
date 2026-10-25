#!/usr/bin/env python
"""
plot_lineshape_grid.py
----------------------
3x3 grid of annual lineshapes (equator vs south pole overlaid), from the file
written by compute_lineshape_grid.m.

x axis : x = (omega/m_a - 1) * 1e6          (as in O'Hare & Pierobon)
         with omega = m_a (1 + v^2/2)  =>   x = 0.5e6 (v/c)^2     (NOT linear in v)
y axis : f(x), normalised so that  int f dx = rho  (local density relative to the
         unperturbed stream; rho = 1 away from focusing).

The MATLAB file stores the MASS per speed bin (not a density).  We transform the
bin EDGES  v -> x  and divide by the new, non-uniform bin widths, so the area under
every curve is preserved exactly (the Jacobian dv/dx is handled by construction).

Usage:
    python plot_lineshape_grid.py
    python plot_lineshape_grid.py --data lineshape_grid_export.mat --out lineshape_grid
    python plot_lineshape_grid.py --xlim 0.22 0.37 --y-decades 6
    python plot_lineshape_grid.py --insets none            # no zoom insets
    python plot_lineshape_grid.py --insets 5               # force an inset in panel 5
    python plot_lineshape_grid.py --usetex                 # LaTeX text rendering
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

C_KMS = 299792.458  # speed of light [km/s]

# ----------------------------- style (edit freely) ---------------------------
COLORS = ["#d62728", "#1f77b4"]          # site 1 (equator), site 2 (south pole); red+blue overlap -> purple
ALPHA_FILL = 0.20                         # low alpha so overlaps stay visible
LW = 1.2
STYLE = {
    "font.family": "EB Garamond",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}
# -----------------------------------------------------------------------------


def v_to_x(v_kms):
    """Lab-frame speed [km/s] -> x = (omega/m_a - 1) * 1e6, using omega = m(1+v^2/2)."""
    return 0.5e6 * (np.asarray(v_kms) / C_KMS) ** 2


def pretty_date(t0, days):
    d = t0 + timedelta(days=float(days))
    return f"{d.day} {d:%B}", f"{d:%H:%M}"


def load(path):
    m = loadmat(path, squeeze_me=True)
    labels = [str(s).strip() for s in np.atleast_1d(m["site_labels"])]
    t0 = datetime.strptime(str(m["t0_iso"])[:19], "%Y-%m-%d %H:%M:%S")
    return dict(mass=np.asarray(m["mass"], float), edges=np.asarray(m["edges"], float),
                rho=np.asarray(m["rho"], float), angles=np.atleast_1d(m["angles_deg"]),
                t_start=np.atleast_1d(m["t_start"]), window=float(m["window_hours"]),
                labels=labels, t0=t0)


def densities(d):
    """-> list over panels of (x_edges, [f_site0, f_site1, ...]) with  int f dx = mass sum."""
    out = []
    for p in range(d["mass"].shape[0]):
        xe = v_to_x(d["edges"][p])
        dx = np.diff(xe)
        out.append((xe, [d["mass"][p, s] / dx for s in range(d["mass"].shape[1])]))
    return out


def draw(ax, xe, fs, colors, base, labels=None):
    """Draw each lineshape as a CLOSED step polygon that returns to `base` at both ends.

    base = the value that stands for "zero": the axis floor on a log axis, 0 on a linear one.
    Everything outside a curve's support is zero (nothing is drawn there), the outline drops
    vertically to `base` at the first/last non-empty bin, and empty bins inside the support
    sit on `base` instead of leaving gaps.  Built from the exact bin edges (no 'mid' step).
    """
    for s, f in enumerate(fs):
        idx = np.flatnonzero(f > 0)
        if idx.size == 0:
            continue
        i0, i1 = idx[0], idx[-1]
        v = np.where(f[i0:i1 + 1] > base, f[i0:i1 + 1], base)   # bins inside the support
        e = xe[i0:i1 + 2]                                         # their edges
        x = np.repeat(e, 2)                                       # e0,e0,e1,e1,...,en,en
        y = np.concatenate(([base], np.repeat(v, 2), [base]))     # base,v0,v0,...,v_{n-1},v_{n-1},base
        ax.fill(x, y, color=colors[s], alpha=ALPHA_FILL, lw=0)
        ax.plot(x, y, color=colors[s], lw=LW, solid_joinstyle="miter",
                label=None if labels is None else labels[s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lineshape_grid_export.mat")
    ap.add_argument("--out", default="plots/lineshape_grid")
    ap.add_argument("--xlim", nargs=2, type=float, default=None,
                    help="shared x-limits in units of 1e-6; default: union of all panels")
    ap.add_argument("--y-decades", type=float, default=6.0,
                    help="shared y-axis spans this many decades below the maximum")
    ap.add_argument("--linear-y", action="store_true", help="linear y axis (shared)")
    ap.add_argument("--insets", default="auto",
                    help="'auto' (narrow panels), 'none', or comma list of 1-based panel numbers")
    ap.add_argument("--tags", nargs="+", default=["eq", "SP"],
                    help="short subscripts for the rho labels, one per site")
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    plt.rcParams.update(STYLE)
    if args.usetex:
        plt.rcParams["text.usetex"] = True

    d = load(args.data)
    dens = densities(d)
    nP, nS = d["mass"].shape[:2]

    # ---- shared limits ------------------------------------------------------
    if args.xlim is None:
        xlo = min(xe[0] for xe, _ in dens)
        xhi = max(xe[-1] for xe, _ in dens)
        pad = 0.01 * (xhi - xlo)
        xlim = (xlo - pad, xhi + pad)
    else:
        xlim = tuple(args.xlim)
    fmax = max(max(f.max() for f in fs) for _, fs in dens)
    ytop = 2.0 * fmax
    ybot = ytop * 10.0 ** (-args.y_decades)

    # ---- which panels get a zoom inset --------------------------------------
    def occupied_width(p, frac=1e-4):
        xe, fs = dens[p]
        xc = 0.5 * (xe[1:] + xe[:-1])
        fmaxp = max(f.max() for f in fs)
        idx = np.where(np.max(np.vstack(fs), axis=0) > frac * fmaxp)[0]
        return xc[idx[0]], xc[idx[-1]]

    if args.insets == "none":
        inset_panels = []
    elif args.insets == "auto":
        inset_panels = [p for p in range(nP)
                        if np.subtract(*occupied_width(p)[::-1]) < 0.10 * (xlim[1] - xlim[0])]
    else:
        inset_panels = [int(k) - 1 for k in args.insets.split(",")]

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    for p, ax in enumerate(axes.flat):
        xe, fs = dens[p]
        draw(ax, xe, fs, COLORS, 0.0 if args.linear_y else ybot,
             labels=d["labels"] if p == 0 else None)
        ax.set_xlim(*xlim)
        if args.linear_y:
            ax.set_ylim(0, ytop)
        else:
            ax.set_yscale("log")
            ax.set_ylim(ybot, ytop)

        date, hhmm = pretty_date(d["t0"], d["t_start"][p])
        ax.set_title(rf"$\theta_\infty = {d['angles'][p]:.0f}^\circ$  —  {date}")
        txt = "\n".join(rf"$\rho_{{\rm {args.tags[s]}}} = {d['rho'][p, s]:.3g}$" for s in range(nS))
        ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))

        if p in inset_panels:
            lo, hi = occupied_width(p)
            w = hi - lo
            axi = ax.inset_axes([0.50, 0.36, 0.47, 0.56])
            draw(axi, xe, fs, COLORS, 0.0 if args.linear_y else 1e-300)
            axi.set_xlim(lo - 0.15 * w, hi + 0.15 * w)
            sel = (0.5 * (xe[1:] + xe[:-1]) > lo - 0.15 * w) & (0.5 * (xe[1:] + xe[:-1]) < hi + 0.15 * w)
            pk = max(f[sel].max() for f in fs)
            if args.linear_y:
                axi.set_ylim(0, 1.1 * pk)
            else:
                axi.set_yscale("log")
                axi.set_ylim(pk * 1e-3, 2 * pk)
            axi.tick_params(labelsize=7)
            axi.xaxis.get_offset_text().set_fontsize(7)
            ax.indicate_inset_zoom(axi, edgecolor="0.4", alpha=0.8)

    for ax in axes[-1]:
        ax.set_xlabel(r"$(\omega/m_a - 1)\times 10^{6}$")
    for ax in axes[:, 0]:
        #ax.set_ylabel(r"$f(x)$,   $\int f\,{\rm d}x=\rho$")
        ax.set_ylabel(r"$ \langle S (\omega) \rangle _T$")
    axes[0, 0].legend(loc="lower left", frameon=False)
    #fig.suptitle(rf"Lineshape integrated over {d['window']:g} h", y=0.995)
    fig.tight_layout()
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=args.dpi)

    # ---- sanity: the v -> x change of variables must preserve the area ------------
    area = np.array([[np.sum(f * np.diff(xe)) for f in fs] for xe, fs in dens])
    print(f"max |int f dx - sum(mass)| / rho = {np.max(np.abs(area - d['mass'].sum(axis=2)) / d['rho']):.2e}"
          f"   max |sum(mass) - rho| / rho = {np.max(np.abs(d['mass'].sum(axis=2) - d['rho']) / d['rho']):.2e}")

    # ---- console summary ----------------------------------------------------
    print(f"x-limits (shared): {xlim[0]:.4f} .. {xlim[1]:.4f}   inset panels: {[p + 1 for p in inset_panels]}")
    print(f"{'panel':>5} {'theta':>6}  {'date':<12} {'start UT':>8}   " +
          "  ".join(f"rho_{l:<10}" for l in d["labels"]))
    for p in range(nP):
        date, hhmm = pretty_date(d["t0"], d["t_start"][p])
        print(f"{p + 1:>5} {d['angles'][p]:>6.0f}  {date:<12} {hhmm:>8}   " +
              "  ".join(f"{d['rho'][p, s]:<14.4g}" for s in range(nS)))
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()