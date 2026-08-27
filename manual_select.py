#!/usr/bin/env python3
"""Interactive peak selector with pre-plot tscrunch.

Wraps the `select_peaks_manual` function you pasted, adding `--tscrunch`
so 20M-sample files don't freeze matplotlib.

Usage:
    # Interactive on a filterbank (profile + dspec), downsample 2000x for display
    python manual_select.py FILE.fil --tscrunch 2000

    # Profile only, different scrunch
    python manual_select.py FILE.fil --tscrunch 500 --no-dspec

    # Just print selected sample ranges (no blanking)
    python manual_select.py FILE.fil --tscrunch 2000 --print-only

    # Select then blank those intervals to 128 (uses blank_fil.py logic)
    python manual_select.py FILE.fil --tscrunch 2000 --blank

The selector itself still shows Time [s] on x (same units as plot_dspec.py:83
t_axis = arange * tsamp) — tscrunch is applied *before* plotting, so clicks
are mapped back to full-resolution sample indices.
"""
import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sigpyproc.readers import FilReader

try:
    from pub_figsize import pub_figsize  # type: ignore
except ImportError:
    def pub_figsize(height_ratio=0.9):
        return (12, 6)


def select_peaks_manual(
    time_axis: np.ndarray,
    profile: np.ndarray,
    *,
    title: str = "Click start/end bounds for each peak (close window when done)",
    x_label: str = "Time [s]",
    y_label: str = r"S [arb.]",
    exclusive_end: bool = True,
    dspec: Optional[np.ndarray] = None,
    freq_axis: Optional[np.ndarray] = None,
    tscrunch: int = 1,
    first_is_noise: bool = False,
) -> List[Tuple[int, int]]:
    """Interactively select peak regions from a 1D profile.

    Parameters
    ----------
    time_axis : 1D array of time values, ascending
    profile   : 1D profile (e.g. collapsed time series)
    dspec     : optional 2D dynamic spectrum (n_freq, n_time).
    freq_axis : 1D frequency array, required if dspec is provided.
    tscrunch  : int, pre-plot time averaging factor (default 1 = no scrunch).
                Displayed arrays are `mean`-scrunched by this factor; returned
                sample indices are mapped back to full resolution.
    first_is_noise : if True, the first completed region is drawn blue and
                intended as an off-pulse noise reference; subsequent regions
                are orange (zap). Returned list is unchanged (first entry is
                still the noise window).
    """
    time_axis = np.asarray(time_axis, float)
    profile = np.asarray(profile, float)

    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1D, got shape={time_axis.shape}")
    if profile.ndim != 1:
        raise ValueError(f"profile must be 1D, got shape={profile.shape}")
    if time_axis.size != profile.size:
        raise ValueError(
            f"time_axis length ({time_axis.size}) does not match profile length ({profile.size})"
        )

    if dspec is not None:
        dspec = np.asarray(dspec, float)
        if dspec.ndim != 2:
            raise ValueError(f"dspec must be 2D, got shape={dspec.shape}")
        if dspec.shape[1] != time_axis.size:
            raise ValueError(
                f"dspec time dimension ({dspec.shape[1]}) does not match time_axis ({time_axis.size})"
            )
        if freq_axis is None:
            raise ValueError("freq_axis is required when dspec is provided")
        freq_axis = np.asarray(freq_axis, float)
        if freq_axis.ndim != 1:
            raise ValueError(f"freq_axis must be 1D, got shape={freq_axis.shape}")
        if freq_axis.size != dspec.shape[0]:
            raise ValueError(
                f"freq_axis length ({freq_axis.size}) does not match dspec n_freq ({dspec.shape[0]})"
            )

    # --- tscrunch pre-plot (mean-average) ---
    if tscrunch < 1:
        raise ValueError("tscrunch must be >=1")
    tscrunch = int(tscrunch)
    if tscrunch > 1:
        n_out = time_axis.size // tscrunch
        if n_out == 0:
            raise ValueError(f"tscrunch {tscrunch} larger than data {time_axis.size}")
        # trim + mean; time_axis is regularly sampled so averaging is fine
        time_axis_disp = time_axis[: n_out * tscrunch].reshape(n_out, tscrunch).mean(axis=1)
        profile_disp = profile[: n_out * tscrunch].reshape(n_out, tscrunch).mean(axis=1)
        if dspec is not None:
            nch = dspec.shape[0]
            dspec_disp = dspec[:, : n_out * tscrunch].reshape(nch, n_out, tscrunch).mean(axis=2)
        else:
            dspec_disp = None
        print(f"tscrunch {tscrunch}x: {time_axis.size} -> {n_out} points for display")
    else:
        time_axis_disp = time_axis
        profile_disp = profile
        dspec_disp = dspec

    clicks: List[float] = []

    if dspec_disp is not None:
        fig, (ax, ax_spec) = plt.subplots(
            2, 1, figsize=pub_figsize(height_ratio=0.9),
            sharex=True,
            gridspec_kw={'height_ratios': [1, 2], 'hspace': 0},
        )
        suffix = ""
        if tscrunch > 1:
            suffix += f"  [tscrunch={tscrunch}x]"
        if first_is_noise:
            suffix += "  — first window = noise (blue)"
        ax.plot(time_axis_disp, profile_disp, color='k', linewidth=1)
        ax.set_title(title + suffix)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        extent = [time_axis_disp[0], time_axis_disp[-1], freq_axis[0], freq_axis[-1]]
        ax_spec.imshow(dspec_disp, aspect='auto', extent=extent, origin='lower', cmap='plasma')
        ax_spec.set_xlabel(x_label)
        ax_spec.set_ylabel('Frequency [MHz]')
        ax_spec.grid(True, alpha=0.3)

        cursor_line = ax.axvline(time_axis_disp[0] if time_axis_disp.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)
        cursor_line_spec = ax_spec.axvline(time_axis_disp[0] if time_axis_disp.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)

        def on_move(event):
            if event.inaxes is None or event.xdata is None:
                return
            x = float(event.xdata)
            cursor_line.set_xdata([x, x])
            cursor_line_spec.set_xdata([x, x])
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is None or event.xdata is None:
                return
            x = float(event.xdata)
            clicks.append(x)
            for a in (ax, ax_spec):
                a.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
            if len(clicks) % 2 == 0:
                start_t, end_t = sorted((clicks[-2], clicks[-1]))
                is_first = first_is_noise and len(clicks) == 2
                color = 'tab:blue' if is_first else 'tab:orange'
                label = 'noise' if is_first else 'zap'
                for a in (ax, ax_spec):
                    a.axvspan(start_t, end_t, color=color, alpha=0.25, label=label)
                # update legend for first noise window
                if is_first:
                    ax.legend(loc='upper right', fontsize=8)
                    ax_spec.legend(loc='upper right', fontsize=8)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('button_press_event', on_click)
    else:
        fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.55))
        ax.plot(time_axis_disp, profile_disp, color='k', linewidth=1)
        suffix = ""
        if tscrunch > 1:
            suffix += f"  [tscrunch={tscrunch}x]"
        if first_is_noise:
            suffix += "  — first window = noise (blue)"
        ax.set_title(title + suffix)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        cursor_line = ax.axvline(time_axis_disp[0] if time_axis_disp.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)

        def on_move(event):
            if event.inaxes != ax or event.xdata is None:
                return
            cursor_line.set_xdata([event.xdata, event.xdata])
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            x = float(event.xdata)
            clicks.append(x)
            ax.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
            if len(clicks) % 2 == 0:
                start_t, end_t = sorted((clicks[-2], clicks[-1]))
                is_first = first_is_noise and len(clicks) == 2
                color = 'tab:blue' if is_first else 'tab:orange'
                ax.axvspan(start_t, end_t, color=color, alpha=0.25, label='noise' if is_first else 'zap')
                if is_first:
                    ax.legend(loc='upper right', fontsize=8)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    if not clicks:
        # full range in original samples
        return [(0, time_axis.size)]

    if len(clicks) % 2 != 0:
        clicks = clicks[:-1]

    regions: List[Tuple[int, int]] = []
    for i in range(0, len(clicks), 2):
        start_t, end_t = sorted((clicks[i], clicks[i + 1]))
        # map from displayed (scrunched) time to original sample indices
        # use time_axis_disp for lookup, then expand by tscrunch
        start_idx_disp = int(np.argmin(np.abs(time_axis_disp - start_t)))
        end_idx_disp = int(np.argmin(np.abs(time_axis_disp - end_t)))
        start_d = min(start_idx_disp, end_idx_disp)
        stop_d = max(start_idx_disp, end_idx_disp) + (1 if exclusive_end else 0)
        stop_d = min(time_axis_disp.size, stop_d)
        # expand to original
        start = start_d * tscrunch
        stop = stop_d * tscrunch
        stop = min(time_axis.size, stop)
        if stop <= start:
            stop = min(time_axis.size, start + tscrunch)
        regions.append((start, stop))
    print(f"Parsed {len(regions)} peak regions (full-res samples): {regions}")

    return regions if regions else [(0, time_axis.size)]


def main():
    ap = argparse.ArgumentParser(description="Interactive peak selector with tscrunch")
    ap.add_argument("filfile", help="filterbank file")
    ap.add_argument("--tscrunch", type=int, default=2000,
                    help="time averaging factor before plot (default 2000 for 0.5 us files -> ~1 ms pixels)")
    ap.add_argument("--no-dspec", action="store_true", help="only show profile, no dspec panel")
    ap.add_argument("--tmin", type=float, default=None, help="start seconds (default 0)")
    ap.add_argument("--tmax", type=float, default=None, help="end seconds (default EOF)")
    ap.add_argument("--blank", action="store_true", help="blank selected intervals to 128 after selection")
    ap.add_argument("--print-only", action="store_true", help="don't blank, just print ranges")
    args = ap.parse_args()

    fr = FilReader(args.filfile)
    hdr = fr.header
    nsamp = hdr.nsamples
    tsamp = hdr.tsamp
    nchans = hdr.nchans

    s0 = int((args.tmin / tsamp)) if args.tmin is not None else 0
    s1 = int((args.tmax / tsamp)) if args.tmax is not None else nsamp
    s0 = max(0, s0); s1 = min(nsamp, s1)
    print(f"{args.filfile}: nsamp={nsamp} tsamp={tsamp} s [{s0}:{s1}] tscrunch={args.tscrunch}")

    block = fr.read_block(s0, s1 - s0)
    arr = np.asarray(block.data, dtype=np.float32)  # (nchans, nsamp_window)
    t_axis = (np.arange(s0, s1) * tsamp)
    profile = arr.sum(axis=0)
    freq_axis = hdr.fch1 + np.arange(nchans) * hdr.foff
    dspec = None if args.no_dspec else arr

    regions = select_peaks_manual(
        t_axis, profile,
        title=f"{args.filfile}  [{s0*tsamp:.4f}-{s1*tsamp:.4f} s]",
        x_label="Time [s]", y_label="Summed power",
        dspec=dspec, freq_axis=freq_axis,
        tscrunch=args.tscrunch,
    )

    # print as seconds for blank_fil.py
    for s, e in regions:
        t_s, t_e = s * tsamp, e * tsamp
        print(f"  {s}-{e}  ({t_s:.6f}-{t_e:.6f} s)")

    if args.blank and not args.print_only:
        # reuse blank_fil.py logic: write 128s
        from pathlib import Path
        p = Path(args.filfile)
        raw = p.read_bytes()
        h_end = raw.find(b"HEADER_END") + len(b"HEADER_END")
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(raw)
            print(f"backup -> {bak}")
        ba = bytearray(raw)
        for s, e in regions:
            b0 = h_end + s * nchans
            b1 = h_end + e * nchans
            ba[b0:b1] = bytes([128]) * (b1 - b0)
            print(f"blanked {s}-{e}")
        p.write_bytes(ba)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
