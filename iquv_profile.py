#!/usr/bin/env python3
"""
Plot dynamic spectrum + polarimetric pulse profile (I, debiased L, PA) for an
IQUV candidate .npz file, plus a stack of tscrunch zoom-in panels (each with
its own PA + I/L sub-panels) placed alongside the main plot. The peak is
found ONCE on the native-resolution profile and reused (as a scaled index)
for every zoom panel -- it is never re-found after scrunching.

Usage:
    python plot_iquv_profile.py cand1_61226_999999182_dm56_67_iquv.npz [--out out.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PA_SIGMA_THRESH = 3.0          # only plot PA where L/sigma_L exceeds this
ZOOM_HALF_WIDTH_NATIVE = 600   # +/- native samples shown at 1x zoom (tight!)
SCRUNCH_FACTORS = [1, 2, 4, 6, 8, 10, 12, 16, 20]


def debias_L(Q, U, sigma):
    """Debiased linear polarisation (Everett & Weisberg 2001)."""
    L_meas = np.sqrt(Q**2 + U**2)
    ratio_sq = np.clip((L_meas / sigma) ** 2 - 1.0, 0.0, None)
    L = np.where(L_meas / sigma > 1.57, sigma * np.sqrt(ratio_sq), 0.0)
    return L


def off_pulse_rms(profile, on_lo, on_hi):
    """RMS of a 1D profile, using samples outside [on_lo, on_hi)."""
    mask = np.ones(profile.shape[0], dtype=bool)
    mask[on_lo:on_hi] = False
    off = profile[mask]
    return np.std(off)


def remove_baseline_2d(arr, time_axis=1):
    """Remove a per-channel baseline (median along time) from a (nchan, nsamp)
    array. Robust to a narrow bright pulse since the median is dominated by
    off-pulse samples."""
    baseline = np.median(arr, axis=time_axis, keepdims=True)
    return arr - baseline


def remove_baseline_1d(arr):
    """Remove a scalar median baseline from a 1D profile."""
    return arr - np.median(arr)


def tscrunch_1d(arr, factor):
    """Simple, robust tscrunch (mean) for 1D arrays. Truncates any leftover
    samples that don't fill a full block."""
    if factor == 1:
        return arr.copy()
    n = arr.shape[0]
    n_keep = (n // factor) * factor
    return arr[:n_keep].reshape(-1, factor).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_file")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--pa-thresh", type=float, default=PA_SIGMA_THRESH,
                     help="L/sigma_L threshold to show PA points")
    ap.add_argument("--zoom-half-width", type=int, default=ZOOM_HALF_WIDTH_NATIVE,
                     help="+/- native samples shown in the 1x zoom panel")
    args = ap.parse_args()

    d = np.load(args.npz_file, allow_pickle=True)

    stokes = d["stokes"]              # (4, nchan, nsamp) -> I, Q, U, V
    pol_order = [p for p in d["pol_order"]]
    tsamp_s = float(d["tsamp_s"])
    nchan = int(d["nchan"])
    nsamp = int(d["nsamp"])
    fch1 = float(d["fch1_mhz"])
    foff = float(d["foff_mhz"])
    cand_id = str(d["cand_id"])
    cand_dm = float(d["cand_dm"])
    cand_snr = float(d["cand_snr"])
    cand_mjd = float(d["cand_mjd"])

    idx = {p: i for i, p in enumerate(pol_order)}
    I = stokes[idx["I"]]  # (nchan, nsamp)
    Q = stokes[idx["Q"]]
    U = stokes[idx["U"]]
    V = stokes[idx["V"]] if "V" in idx else None

    # ------------------------------------------------------------------
    # Baseline removal: subtract a per-channel median (robust to a narrow
    # bright pulse) from every Stokes parameter BEFORE any further
    # analysis, so the dynamic spectrum, profiles, peak-finding, and S/N
    # all operate on baseline-subtracted data.
    # ------------------------------------------------------------------
    I = remove_baseline_2d(I)
    Q = remove_baseline_2d(Q)
    U = remove_baseline_2d(U)
    if V is not None:
        V = remove_baseline_2d(V)

    # Frequency axis (for labeling / dspec extent)
    freqs = fch1 + foff * np.arange(nchan)
    fmin, fmax = min(freqs[0], freqs[-1]), max(freqs[0], freqs[-1])

    # ------------------------------------------------------------------
    # Frequency-averaged (band-integrated) time series, native resolution
    # ------------------------------------------------------------------
    I_prof = I.mean(axis=0)
    Q_prof = Q.mean(axis=0)
    U_prof = U.mean(axis=0)
    V_prof = V.mean(axis=0) if V is not None else None

    time_native = np.arange(nsamp) * tsamp_s * 1e3  # ms

    # ------------------------------------------------------------------
    # Peak finding — ONCE, on the native-resolution I profile. This
    # sample index is reused (rescaled) everywhere below; it is never
    # re-derived after scrunching.
    # ------------------------------------------------------------------
    peak_idx_native = int(np.argmax(I_prof))

    # On-pulse window (native samples) around the peak, for off-pulse stats
    width_ms = float(d["cand_width_ms"])
    if width_ms and width_ms > 0:
        half_on_native = max(int((width_ms * 1e-3 / tsamp_s) * 1.5), 20)
    else:
        half_on_native = max(nsamp // 40, 20)
    on_lo = max(peak_idx_native - half_on_native, 0)
    on_hi = min(peak_idx_native + half_on_native, nsamp)

    # ------------------------------------------------------------------
    # Off-pulse noise -> debiased L, PA, and significance mask (native res)
    # ------------------------------------------------------------------
    sigma_Q = off_pulse_rms(Q_prof, on_lo, on_hi)
    sigma_U = off_pulse_rms(U_prof, on_lo, on_hi)
    sigma_QU = 0.5 * (sigma_Q + sigma_U)

    L_prof = debias_L(Q_prof, U_prof, sigma_QU)
    PA_prof = 0.5 * np.degrees(np.arctan2(U_prof, Q_prof))
    L_meas = np.sqrt(Q_prof ** 2 + U_prof ** 2)
    L_sig = np.divide(L_meas, sigma_QU, out=np.zeros_like(L_meas), where=sigma_QU > 0)
    pa_mask = L_sig > args.pa_thresh

    # ==================================================================
    # FIGURE LAYOUT
    #   Left column : PA / I+L / dynamic-spectrum (native resolution, full window)
    #   Right column: stack of landscape zoom panels, one per tscrunch
    #                 factor, each with its own mini PA row + I/L row,
    #                 all centered on the SAME peak.
    # ==================================================================
    n_zoom = len(SCRUNCH_FACTORS)
    fig = plt.figure(figsize=(18, 3.0 * n_zoom))

    gs_outer = GridSpec(
        1, 2, width_ratios=[1.1, 1.6], wspace=0.18,
        figure=fig, top=0.96, bottom=0.05, left=0.06, right=0.98,
    )

    # ---- Left column: main overview plot ----
    gs_left = gs_outer[0, 0].subgridspec(
        3, 1, height_ratios=[1.0, 1.6, 2.4], hspace=0.08
    )
    ax_pa = fig.add_subplot(gs_left[0, 0])
    ax_prof = fig.add_subplot(gs_left[1, 0], sharex=ax_pa)
    ax_dspec = fig.add_subplot(gs_left[2, 0], sharex=ax_pa)

    # ---- PA panel ----
    ax_pa.scatter(time_native[pa_mask], PA_prof[pa_mask], s=4, c="k")
    ax_pa.set_ylabel("PA (deg)")
    ax_pa.set_ylim(-90, 90)
    ax_pa.tick_params(labelbottom=False)
    ax_pa.set_title(
        f"Cand {cand_id}  MJD {cand_mjd:.6f}  DM {cand_dm:.2f}  SNR {cand_snr:.1f}"
    )

    # ---- I / L profile panel ----
    ax_prof.plot(time_native, I_prof, color="k", lw=0.8, label="I")
    ax_prof.plot(time_native, L_prof, color="crimson", lw=0.8, label="L (debiased)")
    if V_prof is not None:
        ax_prof.plot(time_native, V_prof, color="royalblue", lw=0.6, alpha=0.7, label="V")
    ax_prof.axhline(0, color="gray", lw=0.5)
    ax_prof.set_ylabel("Flux (a.u.)")
    ax_prof.legend(loc="upper right", fontsize=8, frameon=False)
    ax_prof.tick_params(labelbottom=False)

    # ---- Dynamic spectrum panel ----
    extent = [time_native[0], time_native[-1], fmin, fmax]
    vmax = np.percentile(I, 99.5)
    vmin = np.percentile(I, 5)
    ax_dspec.imshow(
        I, aspect="auto", origin="lower", extent=extent,
        cmap="viridis", vmin=vmin, vmax=vmax, interpolation='none'
    )
    ax_dspec.set_ylabel("Freq (MHz)")
    ax_dspec.set_xlabel("Time (ms)")

    peak_t_ms = time_native[peak_idx_native]
    for ax in (ax_pa, ax_prof, ax_dspec):
        ax.axvline(peak_t_ms, color="orange", lw=0.6, ls="--", alpha=0.7)

    # ==================================================================
    # RIGHT COLUMN: stacked landscape zoom panels (tscrunch series)
    # Each zoom "row" is itself split into a PA sub-row and an I/L
    # sub-row, sharing the x-axis. The peak index is transformed by
    # integer division only -- never re-found after scrunching.
    # ==================================================================
    gs_zoom_stack = gs_outer[0, 1].subgridspec(n_zoom, 1, hspace=0.35)

    for row, factor in enumerate(SCRUNCH_FACTORS):
        gs_row = gs_zoom_stack[row, 0].subgridspec(
            2, 1, height_ratios=[1.0, 1.6], hspace=0.0
        )
        ax_pa_z = fig.add_subplot(gs_row[0, 0])
        ax_prof_z = fig.add_subplot(gs_row[1, 0], sharex=ax_pa_z)

        # ---- tscrunch band-averaged I, Q, U (and V) ----
        I_scr = tscrunch_1d(I_prof, factor)
        Q_scr = tscrunch_1d(Q_prof, factor)
        U_scr = tscrunch_1d(U_prof, factor)
        V_scr = tscrunch_1d(V_prof, factor) if V_prof is not None else None
        tsamp_scr = tsamp_s * factor

        # peak index: pure coordinate transform of the native peak index,
        # NOT re-derived from the scrunched data
        peak_idx_scr = peak_idx_native // factor

        # off-pulse region, transformed the same way, for this factor's noise stats
        on_lo_scr = on_lo // factor
        on_hi_scr = max(on_hi // factor, on_lo_scr + 1)
        sigma_Q_scr = off_pulse_rms(Q_scr, on_lo_scr, on_hi_scr)
        sigma_U_scr = off_pulse_rms(U_scr, on_lo_scr, on_hi_scr)
        sigma_QU_scr = 0.5 * (sigma_Q_scr + sigma_U_scr)

        L_scr = debias_L(Q_scr, U_scr, sigma_QU_scr)
        PA_scr = 0.5 * np.degrees(np.arctan2(U_scr, Q_scr))
        L_meas_scr = np.sqrt(Q_scr ** 2 + U_scr ** 2)
        L_sig_scr = np.divide(L_meas_scr, sigma_QU_scr,
                               out=np.zeros_like(L_meas_scr), where=sigma_QU_scr > 0)
        pa_mask_scr = L_sig_scr > args.pa_thresh

        # zoom window, in scrunched-sample units, always centered on peak_idx_scr
        half_width_scr = max(args.zoom_half_width // factor, 5)
        lo = max(peak_idx_scr - half_width_scr, 0)
        hi = min(peak_idx_scr + half_width_scr, I_scr.shape[0])

        t_scr = (np.arange(I_scr.shape[0]) - peak_idx_scr) * tsamp_scr * 1e3  # ms, centered on peak

        sl = slice(lo, hi)

        # ---- S/N of the peak at this scrunch factor ----
        # off-pulse rms of the (baseline-subtracted) scrunched I profile,
        # using the same transformed on-pulse exclusion window as above
        sigma_I_scr = off_pulse_rms(I_scr, on_lo_scr, on_hi_scr)
        peak_amp_scr = I_scr[peak_idx_scr]
        snr_scr = peak_amp_scr / sigma_I_scr if sigma_I_scr > 0 else np.nan

        # PA sub-row
        ax_pa_z.scatter(t_scr[sl][pa_mask_scr[sl]], PA_scr[sl][pa_mask_scr[sl]],
                         s=8, c="k")
        ax_pa_z.axvline(0, color="orange", lw=0.7, ls="--", alpha=0.7)
        ax_pa_z.set_ylim(-90, 90)
        ax_pa_z.tick_params(labelbottom=False, labelsize=7)
        ax_pa_z.set_ylabel("PA", fontsize=8)
        ax_pa_z.set_title(
            f"{factor}x  ({tsamp_scr*1e6:.2f} \u00b5s/samp)   S/N={snr_scr:.1f}",
            fontsize=9,
        )

        # I/L sub-row
        ax_prof_z.plot(t_scr[sl], I_scr[sl], color="k", lw=1.1, label="I")
        ax_prof_z.plot(0.0, peak_amp_scr, marker="v", color="darkorange",
                       ms=6, zorder=5)
        ax_prof_z.plot(t_scr[sl], L_scr[sl], color="crimson", lw=1.1, label="L")
        if V_scr is not None:
            ax_prof_z.plot(t_scr[sl], V_scr[sl], color="royalblue", lw=0.8,
                            alpha=0.7, label="V")
        ax_prof_z.axhline(0, color="gray", lw=0.5)
        ax_prof_z.axvline(0, color="orange", lw=0.7, ls="--", alpha=0.7)
        ax_prof_z.set_ylabel("Flux", fontsize=8)
        ax_prof_z.set_xlabel("Time from peak (ms)", fontsize=8)
        ax_prof_z.tick_params(labelsize=7)
        if row == 0:
            ax_prof_z.legend(loc="upper right", fontsize=7, frameon=False)

    out_path = args.out or args.npz_file.rsplit(".", 1)[0] + "_profile.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
