#!/usr/bin/env python3
"""Blank time ranges in a SIGPROC filterbank to median (or given value).

Keeps the header/MJD/tsamp intact so transientx --cont stays contiguous;
only the data bytes in the requested intervals are overwritten.

Usage:
    # Interactive: open dspec/profile, click ranges to zap (tscrunched for display)
    python blank_fil.py FILE.fil
    python blank_fil.py FILE.fil --tscrunch 2000
    python blank_fil.py FILE.fil --tscrunch 500 --no-dspec

    # Non-interactive: give ranges explicitly
    python blank_fil.py FILE.fil 6.5-7.5
    python blank_fil.py FILE.fil 6.5-7.5 8.0-8.5 9.0-9.6

    # Blank entire file
    python blank_fil.py FILE.fil --all

    # Custom fill / restore
    python blank_fil.py FILE.fil 6.5-7.5 --value 0
    python blank_fil.py FILE.fil --restore   # mv FILE.fil.bak -> FILE.fil
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from sigpyproc.readers import FilReader

try:
    from manual_select import select_peaks_manual
except ImportError:
    from rfi.manual_select import select_peaks_manual


def parse_ranges(strs):
    ranges = []
    for s in strs:
        if "-" not in s:
            raise ValueError(f"range must be START-END, got {s!r}")
        a, b = s.split("-", 1)
        t0, t1 = float(a), float(b)
        if t1 <= t0:
            raise ValueError(f"empty range {s!r}")
        ranges.append((t0, t1))
    ranges.sort()
    # merge overlaps
    merged = []
    for t0, t1 in ranges:
        if merged and t0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t1))
        else:
            merged.append((t0, t1))
    return merged


def main():
    ap = argparse.ArgumentParser(description="Blank time ranges in a .fil to a constant")
    ap.add_argument("filfile", help="filterbank file")
    ap.add_argument("ranges", nargs="*", help="time ranges as START-END (seconds)")
    ap.add_argument("--all", action="store_true", help="blank the entire file")
    ap.add_argument("--value", type=int, default=128, help="fill byte 0-255 (default 128/median)")
    ap.add_argument("--no-backup", action="store_true", help="do not write .bak")
    ap.add_argument("--restore", action="store_true", help="restore from .fil.bak and exit")
    ap.add_argument("--dry-run", action="store_true", help="print what would be done, don't write")
    ap.add_argument("--tscrunch", type=int, default=2000,
                    help="interactive only: time averaging before plot (default 2000, ~1 ms pixels at 0.5 us)")
    ap.add_argument("--no-dspec", action="store_true", help="interactive only: profile panel only")
    ap.add_argument("--interactive", action="store_true",
                    help="force interactive selector even if ranges given")
    args = ap.parse_args()

    p = Path(args.filfile)
    bak = p.with_suffix(p.suffix + ".bak")  # .fil.bak

    if args.restore:
        if not bak.exists():
            print(f"no backup {bak}", file=sys.stderr)
            sys.exit(1)
        bak.replace(p)
        print(f"restored {p} from {bak}")
        return

    interactive = args.interactive or (not args.ranges and not args.all)
    if interactive and not args.all:
        # load window for display and run interactive selector
        hdr = FilReader(str(p)).header
        nsamp = hdr.nsamples
        tsamp = hdr.tsamp
        nchans = hdr.nchans
        print(f"{p.name}: nsamp={nsamp} tsamp={tsamp*1e6:.2f} us dur={nsamp*tsamp:.4f} s nchans={nchans}")
        block = FilReader(str(p)).read_block(0, nsamp)
        arr = np.asarray(block.data, dtype=np.float32)  # (nchans, nsamp)
        t_axis = np.arange(nsamp) * tsamp
        profile = arr.sum(axis=0)
        freq_axis = hdr.fch1 + np.arange(nchans) * hdr.foff
        dspec = None if args.no_dspec else arr
        samp_ranges = select_peaks_manual(
            t_axis, profile,
            title=f"{p.name} — first=offpulse (blue), rest=zap",
            x_label="Time [s]", y_label="Summed power",
            dspec=dspec, freq_axis=freq_axis,
            tscrunch=args.tscrunch,
            first_is_noise=True,
        )
        # select_peaks_manual returns full-res sample (start, stop)
        # filter out full-file sentinel when user closed without clicks
        if len(samp_ranges) == 1 and samp_ranges[0] == (0, t_axis.size):
            print("no ranges selected, nothing to do")
            return
        if len(samp_ranges) < 2:
            print("need at least 2 ranges: first=offpulse (blue, noise stats), rest=zap", file=sys.stderr)
            print("  e.g. click a clean offpulse region first, then each RFI burst", file=sys.stderr)
            return
        noise_s, noise_e = samp_ranges[0]
        zap_ranges = samp_ranges[1:]
        # per-channel noise stats from the blue offpulse window
        noise_data = arr[:, noise_s:noise_e].astype(np.float64)
        # guard against tiny window
        if noise_data.shape[1] < 10:
            print(f"warning: noise window only {noise_data.shape[1]} samples, stats noisy", file=sys.stderr)
        noise_mean = noise_data.mean(axis=1)
        noise_std = noise_data.std(axis=1)
        # clamp std floor like sigpyproc does (avoid zero-variance channels)
        noise_std = np.maximum(noise_std, 1.0)
        print(f"  noise window {noise_s*tsamp:.4f}-{(noise_e)*tsamp:.4f} s ({noise_e-noise_s} samp)")
        for ch in range(nchans):
            print(f"    ch{ch:2d}: mean={noise_mean[ch]:.1f} std={noise_std[ch]:.1f}")
        # stash for random fill
        _noise_mean = noise_mean
        _noise_std = noise_std
        _is_random = True
        hdr2 = hdr
        tsamp2 = tsamp
        nsamp2 = nsamp
        nchans2 = nchans
        total_s = nsamp2 * tsamp2
        # need raw header split for byte offsets
        raw_tmp = p.read_bytes()
        h_end_tmp = raw_tmp.find(b"HEADER_END") + len(b"HEADER_END")
        # build s_ranges/t_ranges from zap_ranges only (noise window is not blanked)
        s_ranges = zap_ranges
        t_ranges = [(s * tsamp2, e * tsamp2) for s, e in zap_ranges]
        # proceed to write path below — stash for reuse
        hdr = hdr2
        # trick: skip the normal ranges parse and jump to write
        ranges = None  # signal we already have s_ranges
        # stash and proceed directly to write
        _interactive_raw = raw_tmp
        _interactive_h_end = h_end_tmp
        hdr = hdr2
        tsamp = tsamp2
        nchans = nchans2
        nsamp = nsamp2
        total_s = total_s
        raw = _interactive_raw
        h_end = _interactive_h_end
        header, data = raw[:h_end], raw[h_end:]
        # s_ranges/t_ranges already set, skip normal parse
    elif args.all:
        ranges = None  # whole file
        # Need tsamp/nchans/nsamples and header length via raw HEADER_END offset
        hdr = FilReader(str(p)).header
        tsamp = hdr.tsamp
        nchans = hdr.nchans
        nsamp = hdr.nsamples
        total_s = nsamp * tsamp
        print(f"{p.name}: nsamp={nsamp} tsamp={tsamp*1e6:.2f} us dur={total_s:.4f} s nchans={nchans}")
        raw = p.read_bytes()
        h_end = raw.find(b"HEADER_END") + len(b"HEADER_END")
        header, data = raw[:h_end], raw[h_end:]
        assert len(data) == nsamp * nchans * (hdr.nbits // 8), f"data len mismatch {len(data)} vs {nsamp*nchans}"
        s_ranges = [(0, nsamp)]
        t_ranges = [(0.0, total_s)]
    else:
        if not args.ranges:
            ap.error("give at least one START-END range or --all")
        ranges = parse_ranges(args.ranges)
        hdr = FilReader(str(p)).header
        tsamp = hdr.tsamp
        nchans = hdr.nchans
        nsamp = hdr.nsamples
        total_s = nsamp * tsamp
        print(f"{p.name}: nsamp={nsamp} tsamp={tsamp*1e6:.2f} us dur={total_s:.4f} s nchans={nchans}")
        raw = p.read_bytes()
        h_end = raw.find(b"HEADER_END") + len(b"HEADER_END")
        header, data = raw[:h_end], raw[h_end:]
        assert len(data) == nsamp * nchans * (hdr.nbits // 8), f"data len mismatch {len(data)} vs {nsamp*nchans}"
        s_ranges = []
        t_ranges = []
        for t0, t1 in ranges:
            s0 = int(max(0, t0 / tsamp))
            s1 = int(min(nsamp, t1 / tsamp))
            if s1 <= s0:
                print(f"  skip empty/out-of-bounds {t0}-{t1} s -> samples {s0}-{s1}", file=sys.stderr)
                continue
            s_ranges.append((s0, s1))
            t_ranges.append((s0 * tsamp, s1 * tsamp))
        if not s_ranges:
            print("nothing to blank", file=sys.stderr)
            sys.exit(1)

    # are we in random-noise mode? set in interactive branch above
    is_random = locals().get("_is_random", False)

    if is_random:
        for (t0, t1), (s0, s1) in zip(t_ranges, s_ranges):
            print(f"  zap {t0:.4f}-{t1:.4f} s  -> samples {s0}-{s1} ({s1-s0} samp, {(s1-s0)*nchans} bytes) [random]")
            print(f"       noise stats per ch mean~{float(_noise_mean.mean()):.1f} std~{float(_noise_std.mean()):.1f}")
    else:
        fill = int(args.value) & 0xFF
        for (t0, t1), (s0, s1) in zip(t_ranges, s_ranges):
            print(f"  blank {t0:.4f}-{t1:.4f} s  -> samples {s0}-{s1} ({s1-s0} samp, {(s1-s0)*nchans} bytes) [fill={fill}]")

    if args.dry_run:
        print("dry-run, not writing")
        return

    if not args.no_backup and not bak.exists():
        bak.write_bytes(raw)
        print(f"  backup -> {bak}")
    else:
        if bak.exists():
            print(f"  backup {bak} already exists, not overwriting")

    # Blank in-place on a bytearray
    ba = bytearray(raw)
    if is_random:
        rng = np.random.default_rng()
        for s0, s1 in s_ranges:
            L = s1 - s0
            # per-channel Gaussian, clip to 0-255
            noise = np.empty((nchans, L), dtype=np.uint8)
            for ch in range(nchans):
                vals = rng.normal(loc=_noise_mean[ch], scale=_noise_std[ch], size=L)
                vals = np.clip(vals, 0, 255).astype(np.uint8)
                noise[ch] = vals
            # interleave time-major: (nchans, L) -> (L, nchans) -> flat bytes
            blob = noise.T.tobytes()
            b0 = h_end + s0 * nchans
            ba[b0:b0+len(blob)] = blob
    else:
        fill = int(args.value) & 0xFF
        for s0, s1 in s_ranges:
            b0 = h_end + s0 * nchans
            b1 = h_end + s1 * nchans
            ba[b0:b1] = bytes([fill]) * (b1 - b0)

    p.write_bytes(ba)
    print(f"  wrote {p} ({'random' if is_random else f'fill={fill}'})")


if __name__ == "__main__":
    main()
