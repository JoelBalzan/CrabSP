#!/usr/bin/env python3
"""
Plot a dynamic spectrum (waterfall) + frequency-summed pulse profile for a
filterbank file, zoomed around a region of interest (e.g. a saturating GP).

Usage:
    # Full file, auto-zoom around the brightest sample:
    python3 plot_dspec.py FILE.fil --out dspec.png

    # Zoom to a specific time window (seconds from start of file):
    python3 plot_dspec.py FILE.fil --out dspec.png --tmin 8.3 --tmax 8.7

    # Zoom to a specific sample range instead of seconds:
    python3 plot_dspec.py FILE.fil --out dspec.png --smin 8300000 --smax 8700000

    # Also mark saturated (clipped) samples on the waterfall:
    python3 plot_dspec.py FILE.fil --out dspec.png --sat-value 255
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sigpyproc.readers import FilReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filfile")
    ap.add_argument("--out", default="dspec.png")
    ap.add_argument("--tmin", type=float, default=None, help="zoom start (s)")
    ap.add_argument("--tmax", type=float, default=None, help="zoom end (s)")
    ap.add_argument("--smin", type=int, default=None, help="zoom start (sample)")
    ap.add_argument("--smax", type=int, default=None, help="zoom end (sample)")
    ap.add_argument("--full", action="store_true",
                     help="plot the full file instead of auto-zooming")
    ap.add_argument("--pad", type=float, default=0.05,
                     help="seconds of padding around auto-detected peak")
    ap.add_argument("--sat-value", type=float, default=None,
                     help="mark samples equal to this value as saturated (e.g. 255)")
    ap.add_argument("--dm", type=float, default=None,
                     help="if set, dedisperse to this DM before plotting")
    args = ap.parse_args()

    fr = FilReader(args.filfile)
    hdr = fr.header
    nsamp = hdr.nsamples
    tsamp = hdr.tsamp
    nchans = hdr.nchans

    print(f"Reading {args.filfile}: nsamples={nsamp}, tsamp={tsamp}, nchans={nchans}")
    block = fr.read_block(0, nsamp)
    arr = np.asarray(block.data, dtype=np.float32)  # shape (nchans, nsamp)

    if args.dm is not None:
        print(f"Dedispersing to DM={args.dm} ...")
        block_dd = block.dedisperse(args.dm)
        arr = np.asarray(block_dd.data, dtype=np.float32)

    tim = arr.sum(axis=0)

    # Work out the zoom window
    if args.full:
        # Downsample to keep under ~10k pixels wide
        max_pix = 10000
        skip = max(1, nsamp // max_pix)
        print(f"--full: downsampling by {skip}x ({nsamp} -> {(nsamp + skip - 1) // skip} pixels)")
        sub = arr[:, ::skip]
        sub_tim = tim[::skip]
        t_axis = np.arange(sub.shape[1]) * tsamp * skip
    else:
        if args.smin is not None and args.smax is not None:
            s0, s1 = args.smin, args.smax
        elif args.tmin is not None and args.tmax is not None:
            s0 = int(args.tmin / tsamp)
            s1 = int(args.tmax / tsamp)
        else:
            # auto: zoom around the brightest sample in the profile
            peak = int(np.argmax(tim))
            pad_samples = int(args.pad / tsamp)
            s0 = max(0, peak - pad_samples)
            s1 = min(nsamp, peak + pad_samples)
            print(f"Auto-zoom around peak sample {peak} "
                  f"(t={peak*tsamp:.4f} s) +/- {args.pad} s")

        s0 = max(0, s0)
        s1 = min(nsamp, s1)
        if s1 <= s0:
            raise ValueError(f"Empty zoom window: samples {s0} to {s1}")

        sub = arr[:, s0:s1]
        sub_tim = tim[s0:s1]
        t_axis = np.arange(s0, s1) * tsamp

    fig, (ax_prof, ax_wf) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # --- profile panel ---
    ax_prof.plot(t_axis, sub_tim, lw=0.8, color="k")
    ax_prof.set_ylabel("Summed power")
    ax_prof.set_title(f"{args.filfile}"
                       f"\n({t_axis[0]:.4f}-{t_axis[-1]:.4f} s)")
    ax_prof.grid(alpha=0.3)

    # --- dynamic spectrum panel ---
    extent = [t_axis[0], t_axis[-1], hdr.fch1 + nchans * hdr.foff, hdr.fch1]
    # NOTE: adjust extent orientation if your foff sign convention differs
    im = ax_wf.imshow(
        sub, aspect="auto", origin="upper",
        extent=extent, cmap="viridis",
    )
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Frequency (MHz)")
    cbar = fig.colorbar(im, ax=ax_wf, orientation="horizontal", pad=0.12,
                         fraction=0.04)
    cbar.set_label("Power (raw units)")

    if args.sat_value is not None:
        sat_mask = sub == args.sat_value
        if sat_mask.any():
            n_sat = int(sat_mask.sum())
            print(f"{n_sat} saturated samples in zoom window "
                  f"({100*n_sat/sat_mask.size:.3f}%)")
            # overlay saturation as red dots on the waterfall
            ys, xs = np.where(sat_mask)
            # convert array indices back to plot coords
            t_vals = t_axis[xs]
            freq_vals = hdr.fch1 + ys * hdr.foff
            ax_wf.scatter(t_vals, freq_vals, s=1, c="red", alpha=0.3,
                           label="saturated")
            ax_wf.legend(loc="upper right", markerscale=5)
        else:
            print("No saturated samples found in zoom window.")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
