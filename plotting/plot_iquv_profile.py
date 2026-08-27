#!/usr/bin/env python3
"""
Plot dynamic spectrum + polarimetric pulse profile (I, debiased L, PA) for an
IQUV candidate .npz file, plus a stack of tscrunch zoom-in panels (each with
its own PA + I/L sub-panels) placed alongside the main plot. The peak is
found ONCE on the native-resolution profile and reused (as a scaled index)
for every zoom panel -- it is never re-found after scrunching.

Usage:
    # single file
    python plot_iquv_profile.py cand1_61226_999999182_dm56_67_iquv.npz [--out out.png]

    # replot all .npz files in a directory
    python plot_iquv_profile.py --npz-dir cutouts/

Can also be imported and called directly, e.g. from extract_cands.py:
    from plot_iquv_profile import generate_profile_plot
    generate_profile_plot(npz_path, out_path)
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# Config (defaults; all overridable via generate_profile_plot() kwargs or
# the CLI flags below)
# ----------------------------------------------------------------------
PA_SIGMA_THRESH = 2.0          # only plot PA where L/sigma_L exceeds this
ZOOM_HALF_WIDTH_NATIVE = 150   # +/- native samples shown at 1x zoom (tight!)
SCRUNCH_FACTORS = [1, 2, 4, 6, 8, 10]#, 12, 16, 20]
ZOOM_NCOLS = 2                 # number of columns in the zoom-panel grid
DSPEC_INTERPOLATION = "gaussian"  # matplotlib imshow interpolation for the dspec


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


def calculate_pa(Q, U, pa_mask, unwrap=False):
    """
    Calculate PA in degrees, with PA defined modulo 180 deg.

    When unwrap=True, only significant PA samples are used for unwrapping.
    Unwrapping is performed independently for each contiguous region of
    significant samples, preventing low-S/N samples or gaps from introducing
    artificial 180-deg offsets.
    """
    pa = 0.5 * np.degrees(np.arctan2(U, Q))
    pa_plot = np.full_like(pa, np.nan, dtype=float)

    valid_idx = np.flatnonzero(pa_mask)

    if valid_idx.size == 0:
        return pa_plot

    if not unwrap:
        pa_plot[valid_idx] = pa[valid_idx]
        return pa_plot

    # Find contiguous runs of significant samples.
    breaks = np.where(np.diff(valid_idx) > 1)[0] + 1
    runs = np.split(valid_idx, breaks)

    for run in runs:
        if run.size == 0:
            continue

        pa_rad = np.radians(pa[run])
        pa_unwrapped = np.unwrap(pa_rad, period=np.pi)
        pa_plot[run] = np.degrees(pa_unwrapped)

    return pa_plot

def plot_pa_segments(ax, t, pa, mask, **kwargs):
    """Plot PA as separate lines for each contiguous significant region."""
    idx = np.flatnonzero(mask)

    if idx.size == 0:
        return

    breaks = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, breaks)

    for run in runs:
        if run.size >= 2:
            ax.plot(t[run], pa[run], **kwargs)


# ------------------------------------------------------------------
# On / off-pulse mask helpers (boxcar-based)
# ------------------------------------------------------------------
def boxcar_width(profile, frac=0.95):
    prof = np.nan_to_num(np.squeeze(profile))
    n = len(prof)
    target_flux = frac * np.sum(prof)
    cumsum = np.cumsum(prof)
    min_width = n
    best_start, best_end = 0, n - 1
    for start in range(n):
        start_flux = cumsum[start - 1] if start > 0 else 0
        target_end_flux = start_flux + target_flux
        end_indices = np.where(cumsum >= target_end_flux)[0]
        if len(end_indices) > 0:
            end = end_indices[0]
            width = end - start + 1
            if width < min_width:
                min_width = width
                best_start, best_end = start, end
    return best_start, best_end


def make_onpulse_mask(n_time, left, right):
    on_mask = np.zeros(int(n_time), dtype=bool)
    l = max(0, int(left))
    r = min(int(n_time) - 1, int(right))
    if r >= l:
        on_mask[l:r + 1] = True
    return on_mask


def make_offpulse_mask(n_time, left, right, buffer_bins=0):
    n = int(n_time)
    l_on = max(0, int(left))
    r_on = min(n - 1, int(right))
    buf = max(0, int(buffer_bins))
    l_excl = max(0, l_on - buf)
    r_excl = min(n - 1, r_on + buf)
    off_mask = np.ones(n, dtype=bool)
    if r_excl >= l_excl:
        off_mask[l_excl:r_excl + 1] = False
    return off_mask


def on_off_pulse_masks_from_profile(profile, intrinsic_width_bins,
                                    frac=0.95, buffer_frac=None):
    prof = np.asarray(profile, dtype=float)
    left, right = boxcar_width(prof, frac=frac)
    buffer_bins = (int(float(buffer_frac) * intrinsic_width_bins)
                   if buffer_frac is not None else 0)
    on_mask = make_onpulse_mask(prof.size, left, right)
    off_mask = np.zeros(prof.size, dtype=bool)
    end_off = max(0, left - buffer_bins - 1)
    if end_off >= 0:
        off_mask[0:end_off + 1] = True
    return on_mask, off_mask, (left, right)

def _running_median(x, win):
    win = max(int(win), 1)
    if win % 2 == 0:
        win += 1
    if win <= 1 or x.size < win:
        return x.copy()
    half = win // 2
    xpad = np.pad(x, (half, half), mode="edge")
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        out[i] = np.median(xpad[i:i + win])
    return out


def _self_noise_bins(I_on, sigma_N2, intrinsic_width_bins, trend_frac=0.15,
                      n_bins_max=30, min_bin_count=10):
    """Detrend I_on (on-pulse, off-pulse-subtracted) and bin residual
    variance by local trend value. Works at any time resolution -- caller
    passes intrinsic_width_bins already converted to that resolution's
    sample units.

    Returns dict with bin_means, bin_vars, bin_counts, ratio (possibly
    empty arrays if there isn't enough data), plus sigma_N2 passed through.
    """
    out = dict(bin_means=np.array([]), bin_vars=np.array([]),
               bin_counts=np.array([]), ratio=np.array([]), sigma_N2=sigma_N2)

    if I_on.size <= 50 or sigma_N2 <= 0:
        return out

    trend_win = max(int(round(intrinsic_width_bins * trend_frac)), 3)
    trend = _running_median(I_on, trend_win)
    resid = I_on - trend

    lo_p, hi_p = np.percentile(trend, [1, 99])
    n_bins = min(n_bins_max, max(5, I_on.size // 20))
    edges = np.linspace(lo_p, hi_p, n_bins + 1)

    bin_means, bin_vars, bin_counts = [], [], []
    for j in range(n_bins):
        in_bin = (trend >= edges[j]) & (trend < edges[j + 1])
        cnt = int(in_bin.sum())
        if cnt < min_bin_count:
            continue
        bin_means.append(float(np.mean(trend[in_bin])))
        bin_vars.append(float(np.var(resid[in_bin])))
        bin_counts.append(cnt)

    bin_means = np.array(bin_means)
    bin_vars = np.array(bin_vars)
    bin_counts = np.array(bin_counts)
    ratio = bin_vars / sigma_N2 if bin_means.size else np.array([])

    out.update(bin_means=bin_means, bin_vars=bin_vars,
               bin_counts=bin_counts, ratio=ratio)
    return out


def _self_noise_slope_fit(I_on, sigma_N2, intrinsic_width_bins, n_boot=200,
                           rng=None, **kw):
    """Weighted linear fit of variance-ratio vs local-trend mean, with a
    bootstrap uncertainty on the slope. This is the primary self-noise
    sweep metric -- it uses every bin (weighted by count) instead of a
    single brightest-bin ratio, which is too noisy on its own to decide
    whether self-noise has actually gone away at a given time resolution.

    Returns dict with slope, slope_err, n_bins. slope/slope_err are nan
    if there isn't enough data to fit (fewer than 2 populated bins).
    """
    rng = rng or np.random.default_rng()
    sn = _self_noise_bins(I_on, sigma_N2, intrinsic_width_bins, **kw)
    bm, bv, cnt = sn["bin_means"], sn["bin_vars"], sn["bin_counts"]

    out = dict(slope=np.nan, slope_err=np.nan, n_bins=int(bm.size),
                bins=sn)
    if bm.size < 2 or sigma_N2 <= 0:
        return out

    ratio = bv / sigma_N2

    def _weighted_slope(y):
        w = cnt.astype(float)
        A = np.vstack([np.ones_like(bm), bm]).T
        W = np.diag(w)
        try:
            coef, *_ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
            return float(coef[1])
        except np.linalg.LinAlgError:
            return np.nan

    slope = _weighted_slope(ratio)

    # Bootstrap: resample on-pulse residuals within each bin (using the
    # trend/resid arrays recomputed inside _self_noise_bins is not
    # directly exposed, so instead we bootstrap at the bin level by
    # resampling bin variances via a chi-square approximation: for a
    # bin with cnt samples, Var estimate has relative std ~ sqrt(2/cnt).
    # This is a standard variance-of-variance approximation and avoids
    # needing to re-run the full detrend on every bootstrap draw.
    boot_slopes = np.empty(n_boot)
    rel_std = np.sqrt(2.0 / np.maximum(cnt.astype(float), 2))
    for b in range(n_boot):
        ratio_b = ratio * (1.0 + rel_std * rng.standard_normal(ratio.size))
        boot_slopes[b] = _weighted_slope(ratio_b)
    slope_err = float(np.nanstd(boot_slopes))

    out.update(slope=slope, slope_err=slope_err)
    return out


def generate_profile_plot(npz_file, out=None, pa_thresh=PA_SIGMA_THRESH,
                           zoom_half_width=ZOOM_HALF_WIDTH_NATIVE,
                           zoom_ncols=ZOOM_NCOLS,
                           dspec_interp=DSPEC_INTERPOLATION,
                           scrunch_factors=None, title_suffix='', dpi=150,
                           unwrap_pa=False, self_noise_sig_thresh=2.0):
    """Build the full diagnostic figure for one candidate .npz and save it.

    npz_file: path to a candidate _iquv.npz written by extract_cands.py
    out: output PNG path (default: npz_file with _iquv.npz -> _profile.png)
    title_suffix: appended to the plot title (e.g. calibration status)
    self_noise_ratio_thresh: variance-ratio (sigma^2/sigma_N^2) threshold
        below which a tscrunch factor is considered radiometer-limited
        (i.e. "safe" from self-noise bias). Used only to annotate/report
        the coarsest safe factor among scrunch_factors; does not affect
        what gets plotted.

    Returns the output path.
    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scrunch_factors = scrunch_factors or SCRUNCH_FACTORS

    d = np.load(str(npz_file), allow_pickle=True)

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
    calib_applied = bool(d.get("calib_applied", False))
    tres_label = Path(str(npz_file)).parent.name

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

    # ------------------------------------------------------------------
    # Frequency-averaged (band-integrated) time series, native resolution
    # ------------------------------------------------------------------
    I_prof = I.sum(axis=0)
    Q_prof = Q.sum(axis=0)
    U_prof = U.sum(axis=0)
    V_prof = V.sum(axis=0) if V is not None else None

    time_native = np.arange(nsamp) * tsamp_s * 1e3  # ms

    # ------------------------------------------------------------------
    # Peak finding — ONCE, on the native-resolution I profile. This
    # sample index is reused (rescaled) everywhere below; it is never
    # re-derived after scrunching.
    # ------------------------------------------------------------------
    peak_idx_native = int(np.argmax(I_prof))

    # On-pulse window (native samples) around the peak, for off-pulse stats
    width_ms = float(d["cand_width_ms"]) if "cand_width_ms" in d and d["cand_width_ms"] is not None else 0.0
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
    sigma_I = off_pulse_rms(I_prof, on_lo, on_hi)

    L_prof = debias_L(Q_prof, U_prof, sigma_QU)

    L_meas = np.sqrt(Q_prof**2 + U_prof**2)
    L_sig = np.divide(
        L_meas,
        sigma_QU,
        out=np.zeros_like(L_meas),
        where=sigma_QU > 0,
    )
    I_sig = np.divide(
        I_prof,
        sigma_I,
        out=np.zeros_like(I_prof),
        where=sigma_I > 0,
    )

    # Only use statistically significant samples for PA.
    pa_mask = (L_sig > pa_thresh) & (I_sig > 3.0)

    # Calculate / unwrap PA only after masking.
    PA_prof = calculate_pa(Q_prof, U_prof, pa_mask, unwrap=unwrap_pa)

    # PA error bars: sigma_PA = 0.5 * sigma_QU / L (in degrees)
    sigma_PA_deg = np.full(nsamp, 90.0)
    valid = L_meas > 0
    sigma_PA_deg[valid] = 0.5 * (180.0 / np.pi) * sigma_QU / L_meas[valid]

    # ==================================================================
    # FIGURE LAYOUT
    #   Left column : PA / I+L / dynamic-spectrum (native resolution, full window)
    #   Right side  : grid of landscape zoom panels (2 columns), one per
    #                 tscrunch factor, each with its own mini PA row +
    #                 I/L row, all centered on the SAME peak.
    # ==================================================================
    n_zoom = len(scrunch_factors)
    ncols = max(zoom_ncols, 1)
    nrows = int(np.ceil(n_zoom / ncols))

    row_h = 2.6                       # inches per zoom row
    fig_h = max(row_h * nrows, 8.0)
    fig_w = 9 + 5.5 * ncols
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_outer = GridSpec(
        1, 2, width_ratios=[1.0, 1.15 * ncols], wspace=0.16,
        figure=fig, top=0.95, bottom=0.06, left=0.055, right=0.98,
    )

    # ---- Left column: main overview plot ----
    gs_left = gs_outer[0, 0].subgridspec(
        2, 1, height_ratios=[1.0, 3.0], hspace=0.25
    )
    gs_sn = gs_left[0, 0].subgridspec(1, 2, wspace=0.35)
    ax_sn_var = fig.add_subplot(gs_sn[0, 0])
    ax_sn_ratio = fig.add_subplot(gs_sn[0, 1])

    gs_prof_dspec = gs_left[1, 0].subgridspec(
        2, 1, height_ratios=[1.3, 1.6], hspace=0.0
    )
    ax_prof = fig.add_subplot(gs_prof_dspec[0, 0])
    ax_dspec = fig.add_subplot(gs_prof_dspec[1, 0], sharex=ax_prof)

    # ---- Self-noise panel (native resolution) ----
    intrinsic_width_bins = max(int(float(d["cand_width_ms"]) * 1e-3 / tsamp_s), 1) if width_ms > 0 else max(nsamp // 40, 1)
    on_mask_sn, off_mask_sn, (bl, br) = on_off_pulse_masks_from_profile(
        I_prof, intrinsic_width_bins, frac=0.95, buffer_frac=2.0)

    I_on = I_prof[on_mask_sn] - np.median(I_prof[off_mask_sn])
    sigma_N2 = float(np.var(I_prof[off_mask_sn]))
    
    sn_native = _self_noise_bins(I_on, sigma_N2, intrinsic_width_bins)
    sn_native_fit = _self_noise_slope_fit(I_on, sigma_N2, intrinsic_width_bins)
    bin_means, bin_vars, bin_counts, ratio = (
        sn_native["bin_means"], sn_native["bin_vars"],
        sn_native["bin_counts"], sn_native["ratio"])

    if bin_means.size > 0:
        sz = np.clip(bin_counts / bin_counts.max() * 60, 10, 60)
        ax_sn_var.scatter(bin_means, bin_vars, s=sz, c="steelblue",
                          edgecolors="k", linewidths=0.4, alpha=0.85, zorder=3)
        ax_sn_var.axhline(sigma_N2, color="0.5", ls="--", lw=1,
                          label=rf"$\sigma_N^2$={sigma_N2:.1e}")
        ax_sn_var.set_xlabel("Local trend (off-pulse subtracted)", fontsize=7)
        ax_sn_var.set_ylabel(r"Var(residual)", fontsize=7)
        ax_sn_var.set_title("Self-noise (detrended)", fontsize=8)
        ax_sn_var.legend(fontsize=6, frameon=False)

        ax_sn_ratio.scatter(bin_means, ratio, s=sz, c="darkorange",
                            edgecolors="k", linewidths=0.4, alpha=0.85, zorder=3)
        ax_sn_ratio.axhline(1.0, color="0.5", ls="--", lw=1, label="radiometer")
        ax_sn_ratio.set_xlabel("Local trend (off-pulse subtracted)", fontsize=7)
        ax_sn_ratio.set_ylabel(r"$\sigma^2/\sigma_N^2$", fontsize=7)
        ax_sn_ratio.set_title("Variance ratio", fontsize=8)
        ax_sn_ratio.legend(fontsize=6, frameon=False)
    elif I_on.size > 50 and sigma_N2 > 0:
        ax_sn_var.text(0.5, 0.5, "too few populated bins", transform=ax_sn_var.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
        ax_sn_ratio.text(0.5, 0.5, "too few populated bins", transform=ax_sn_ratio.transAxes,
                          ha="center", va="center", fontsize=8, color="gray")
    else:
        ax_sn_var.text(0.5, 0.5, "insufficient on-pulse", transform=ax_sn_var.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
        ax_sn_ratio.text(0.5, 0.5, "insufficient on-pulse", transform=ax_sn_ratio.transAxes,
                          ha="center", va="center", fontsize=8, color="gray")

    for ax_sn in (ax_sn_var, ax_sn_ratio):
        ax_sn.tick_params(labelsize=6)

    title = f"Cand {cand_id}  MJD {cand_mjd:.6f}  DM {cand_dm:.2f}  SNR {cand_snr:.1f}  {tres_label}"
    if calib_applied:
        title += "  [cal]"
    if title_suffix:
        title += f"  {title_suffix}"
    ax_sn_var.set_title(f"Self-noise  |  Cand {cand_id}", fontsize=8)

    # ---- I / L profile panel ----
    on_t_lo = time_native[on_lo]
    on_t_hi = time_native[min(on_hi, nsamp - 1)]
    ax_prof.axvspan(on_t_lo, on_t_hi, color="royalblue", alpha=0.12, zorder=0)
    ax_prof.plot(time_native, I_prof, color="k", lw=0.8, label="I")
    ax_prof.plot(time_native, L_prof, color="crimson", lw=0.8, label="L (debiased)")
    if V_prof is not None:
        ax_prof.plot(time_native, V_prof, color="royalblue", lw=0.6, alpha=0.7, label="V")
    ax_prof.axhline(0, color="gray", lw=0.5)
    ax_prof.set_ylabel("Flux (a.u.)")
    ax_prof.legend(loc="upper right", fontsize=8, frameon=False)
    ax_prof.set_title(title, fontsize=9)
    ax_prof.tick_params(labelbottom=False)

    # ---- Dynamic spectrum panel (matplotlib-interpolated for display only) ----
    # NOTE on frequency orientation: freqs[0] corresponds to array row 0 and
    # equals fch1_mhz (foff_mhz is typically negative, so freqs descends).
    # origin='upper' places row 0 at the TOP of the image, so we must give
    # the extent's top value as freqs[0] (not fmax blindly).
    extent = [time_native[0], time_native[-1], freqs[-1], freqs[0]]
    vmax = np.percentile(I, 99.5)
    vmin = np.percentile(I, 5)
    ax_dspec.imshow(
        I, aspect="auto", origin="upper", extent=extent,
        cmap="viridis", vmin=vmin, vmax=vmax,
        interpolation=dspec_interp,
    )
    ax_dspec.set_ylabel("Freq (MHz)")
    ax_dspec.set_xlabel("Time (ms)")

    peak_t_ms = time_native[peak_idx_native]
    ax_prof.axvline(peak_t_ms, color="orange", lw=0.6, ls="--", alpha=0.7)
    ax_dspec.axvline(peak_t_ms, color="orange", lw=0.6, ls="--", alpha=0.7)

    # ==================================================================
    # RIGHT SIDE: grid of landscape zoom panels (tscrunch series), laid
    # out in `ncols` columns. Each zoom "cell" is itself split into a PA
    # sub-row and an I/L sub-row, sharing the x-axis. The peak index is
    # transformed by integer division only -- never re-found after
    # scrunching.
    #
    # Self-noise sweep: at each factor we also run the same detrend+bin
    # diagnostic on that factor's on-pulse I_scr, so we can see directly
    # (in each panel's title, and in the summary printed after the loop)
    # at what scrunch factor the variance ratio settles to ~1, i.e. the
    # coarsest time resolution at which the polarimetry error bars are
    # no longer significantly biased by self-noise.
    # ==================================================================
    gs_zoom_grid = gs_outer[0, 1].subgridspec(
        nrows, ncols, hspace=0.55, wspace=0.22
    )

    self_noise_sweep = []  # (factor, tsamp_scr, ratio_top)

    for i, factor in enumerate(scrunch_factors):
        row, col = divmod(i, ncols)
        gs_cell = gs_zoom_grid[row, col].subgridspec(
            2, 1, height_ratios=[1.0, 1.6], hspace=0.0
        )
        ax_pa_z = fig.add_subplot(gs_cell[0, 0])
        ax_prof_z = fig.add_subplot(gs_cell[1, 0], sharex=ax_pa_z)

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
        sigma_I_scr = off_pulse_rms(I_scr, on_lo_scr, on_hi_scr)

        L_scr = debias_L(Q_scr, U_scr, sigma_QU_scr)

        L_meas_scr = np.sqrt(Q_scr**2 + U_scr**2)
        L_sig_scr = np.divide(
            L_meas_scr,
            sigma_QU_scr,
            out=np.zeros_like(L_meas_scr),
            where=sigma_QU_scr > 0,
        )
        I_sig_scr = np.divide(
            I_scr,
            sigma_I_scr,
            out=np.zeros_like(I_scr),
            where=sigma_I_scr > 0,
        )

        pa_mask_scr = (L_sig_scr > pa_thresh) & (I_sig_scr > 3.0)

        PA_scr = calculate_pa(
            Q_scr,
            U_scr,
            pa_mask_scr,
            unwrap=unwrap_pa,
        )

        # PA error bars for zoom panel
        sigma_PA_scr = np.full(I_scr.shape[0], 90.0)
        valid_scr = L_meas_scr > 0
        sigma_PA_scr[valid_scr] = 0.5 * (180.0 / np.pi) * sigma_QU_scr / L_meas_scr[valid_scr]

        # zoom window, in scrunched-sample units, always centered on peak_idx_scr
        half_width_scr = max(zoom_half_width // factor, 5)
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

        # ---- Self-noise check at this scrunch factor ----
        # Reuse the same on/off masks (transformed to this resolution) so
        # the on-pulse window tracks the same physical pulse extent across
        # factors. intrinsic_width_bins is converted into this factor's
        # sample units (min 1 bin) so the trend-window fraction stays
        # physically consistent.
        intrinsic_width_bins_scr = max(intrinsic_width_bins // factor, 1)
        on_mask_scr, off_mask_scr, _ = on_off_pulse_masks_from_profile(
            I_scr, intrinsic_width_bins_scr, frac=0.95, buffer_frac=2.0)
        I_on_scr = I_scr[on_mask_scr] - np.median(I_scr[off_mask_scr])
        sigma_N2_scr = float(np.var(I_scr[off_mask_scr]))
        sn_fit_scr = _self_noise_slope_fit(
            I_on_scr, sigma_N2_scr, intrinsic_width_bins_scr)
        self_noise_sweep.append((factor, tsamp_scr, sn_fit_scr["slope"],
                                  sn_fit_scr["slope_err"], sn_fit_scr["n_bins"]))

        # PA sub-row
        ax_pa_z.errorbar(t_scr[sl][pa_mask_scr[sl]], PA_scr[sl][pa_mask_scr[sl]],
                         yerr=sigma_PA_scr[sl][pa_mask_scr[sl]], fmt='none',
                         ecolor='gray', elinewidth=0.5, capsize=2, zorder=1)
        ax_pa_z.scatter(t_scr[sl][pa_mask_scr[sl]], PA_scr[sl][pa_mask_scr[sl]],
                         s=8, c="k", zorder=2)
        plot_pa_segments(
            ax_pa_z,
            t_scr[sl],
            PA_scr[sl],
            pa_mask_scr[sl],
            color="k",
            lw=0.8,
            alpha=0.5,
        )
        ax_pa_z.axvline(0, color="orange", lw=0.7, ls="--", alpha=0.7)
        if not unwrap_pa:
            ax_pa_z.set_ylim(-90, 90)
        ax_pa_z.tick_params(labelbottom=False, labelsize=7)
        ax_pa_z.set_ylabel("PA", fontsize=8)

        slope_scr, slope_err_scr = sn_fit_scr["slope"], sn_fit_scr["slope_err"]
        if np.isfinite(slope_scr) and np.isfinite(slope_err_scr) and slope_err_scr > 0:
            sig_scr = abs(slope_scr) / slope_err_scr
            slope_str = f"{slope_scr:+.2e}\u00b1{slope_err_scr:.1e} ({sig_scr:.1f}\u03c3)"
            flag = "" if sig_scr < 2.0 else "  \u26a0"
        else:
            slope_str = "n/a"
            flag = ""
        ax_pa_z.set_title(
            f"{factor}x  ({tsamp_scr*1e6:.2f} \u00b5s)   S/N={snr_scr:.1f}"
            f"   slope={slope_str}{flag}",
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
        if i == 0:
            ax_prof_z.legend(loc="upper right", fontsize=7, frameon=False)

    # ------------------------------------------------------------------
    # Self-noise sweep summary: use the weighted-slope fit (not a
    # single-bin ratio) as the criterion, since ratio_top is dominated by
    # per-bin sampling noise and bounces around non-monotonically with
    # scrunch factor even when a real trend is present. "Safe" here means
    # the slope is statistically consistent with zero (|slope| < 2*slope_err),
    # i.e. no resolvable rise of variance with intensity -- not merely that
    # some single bin happened to land near sigma_N^2.
    # ------------------------------------------------------------------
    full_sweep = [(1, tsamp_s, sn_native_fit["slope"], sn_native_fit["slope_err"],
                   sn_native_fit["n_bins"])] + self_noise_sweep

    def _is_safe(slope, slope_err, n_bins, sig_thresh=self_noise_sig_thresh, min_bins=3):
        if not (np.isfinite(slope) and np.isfinite(slope_err)):
            return False
        if n_bins < min_bins:
            return False
        if slope_err <= 0:
            return False
        return abs(slope) / slope_err < sig_thresh

    safe = [(f, ts, sl, se) for (f, ts, sl, se, nb) in full_sweep
            if _is_safe(sl, se, nb)]

    if safe:
        # report the FINEST (smallest tsamp) safe factor -- the best
        # available time resolution at which self-noise is not resolvable
        best_factor, best_tsamp, best_slope, best_err = min(safe, key=lambda x: x[1])
        sweep_msg = (f"self-noise consistent with zero at >= {best_factor}x "
                     f"({best_tsamp*1e6:.2f} \u00b5s, "
                     f"slope={best_slope:+.2e}\u00b1{best_err:.1e})")
    elif full_sweep:
        sweep_msg = "self-noise slope not consistent with zero at any tested factor"
    else:
        sweep_msg = ""

    if sweep_msg:
        fig.text(0.005, 0.005, sweep_msg, fontsize=7, color="firebrick",
                  ha="left", va="bottom")
        print(f"[{cand_id}] {sweep_msg}")
        for f, ts, sl, se, nb in full_sweep:
            if np.isfinite(sl) and np.isfinite(se) and se > 0:
                sig = abs(sl) / se
                print(f"    {f:>4d}x  {ts*1e6:8.2f} us  n_bins={nb:2d}  "
                      f"slope={sl:+.3e}\u00b1{se:.2e}  ({sig:.1f}\u03c3)"
                      f"{'  SAFE' if sig < 2.0 else ''}")
            else:
                print(f"    {f:>4d}x  {ts*1e6:8.2f} us  n_bins={nb:2d}  slope=n/a")

    out_path = out or str(npz_file).rsplit(".", 1)[0] + "_profile.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_file", nargs='?', default=None,
                    help="single .npz file to plot")
    ap.add_argument("--npz-dir", default=None,
                    help="directory of *_iquv.npz files to replot")
    ap.add_argument("--out", default=None, help="output PNG path (single-file mode)")
    ap.add_argument("--pa-thresh", type=float, default=PA_SIGMA_THRESH,
                     help="L/sigma_L threshold to show PA points")
    ap.add_argument("--zoom-half-width", type=int, default=ZOOM_HALF_WIDTH_NATIVE,
                     help="+/- native samples shown in the 1x zoom panel")
    ap.add_argument("--zoom-ncols", type=int, default=ZOOM_NCOLS,
                     help="number of columns in the zoom-panel grid")
    ap.add_argument("--dspec-interp", default=DSPEC_INTERPOLATION,
                     help="matplotlib imshow interpolation for the dspec "
                          "(e.g. gaussian, bilinear, bicubic, none)")
    ap.add_argument("--unwrap-pa", action="store_true",
                     help="unwrap PA to remove ±90° discontinuities")
    args = ap.parse_args()

    if args.npz_dir:
        npz_files = sorted(Path(args.npz_dir).rglob('*_iquv.npz'))
        if not npz_files:
            print(f"No *_iquv.npz files found in {args.npz_dir}")
            return
        print(f"Found {len(npz_files)} .npz files in {args.npz_dir}")
        for i, npz_path in enumerate(npz_files, 1):
            print(f"[{i}/{len(npz_files)}] {npz_path.name}")
            try:
                out_path = generate_profile_plot(
                    npz_path, pa_thresh=args.pa_thresh,
                    zoom_half_width=args.zoom_half_width,
                    zoom_ncols=args.zoom_ncols,
                    dspec_interp=args.dspec_interp,
                    unwrap_pa=args.unwrap_pa,
                )
                print(f"  -> {out_path}")
            except Exception as e:
                print(f"  FAILED: {e}")
    elif args.npz_file:
        out_path = generate_profile_plot(
            args.npz_file, out=args.out, pa_thresh=args.pa_thresh,
            zoom_half_width=args.zoom_half_width, zoom_ncols=args.zoom_ncols,
            dspec_interp=args.dspec_interp, unwrap_pa=args.unwrap_pa,
        )
        print(f"Saved: {out_path}")
    else:
        ap.error("provide either a positional npz_file or --npz-dir")


if __name__ == "__main__":
    main()