import argparse

import matplotlib.pyplot as plt
import numpy as np
from sigpyproc.readers import FilReader

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument(
    "--ts",
    type=int,
    default=1,
    help="Number of time samples to average together (default: 1)"
)
args = parser.parse_args()

fil = FilReader(args.filename)

# Read first million samples (or whole file if shorter)
start = 0
nsamples = min(1_000_000, fil.header.nsamples)

block = fil.read_block(start, nsamples)
arr = block.data

# Ensure array is (time, frequency)
if arr.shape[0] == fil.header.nchans:
    arr = arr.T

# Remove bandpass
arr = arr - np.median(arr, axis=0, keepdims=True)

# Time scrunch
tscrunch = args.ts
if tscrunch > 1:
    ntime = arr.shape[0] // tscrunch
    arr = arr[:ntime * tscrunch]

    arr = arr.reshape(
        ntime,
        tscrunch,
        arr.shape[1]
    ).mean(axis=1)

    print(f"Applied time scrunch: {tscrunch}")
    print(f"New shape: {arr.shape}")

# Pulse profile
profile = arr.sum(axis=1)

# Plot
fig, (ax_prof, ax_ds) = plt.subplots(
    2, 1,
    figsize=(12, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 4]}
)

# Time axis in original samples
time_axis = start + np.arange(arr.shape[0]) * tscrunch

# Pulse profile
ax_prof.plot(time_axis, profile, lw=0.8, c='k')
ax_prof.set_ylabel("Mean Power")
ax_prof.set_xticks([])

# Dynamic spectrum
extent = [
    time_axis[0],
    time_axis[-1],
    fil.header.fch1 + (fil.header.nchans - 1) * fil.header.foff,
    fil.header.fch1,
]

vmin, vmax = np.percentile(arr, [5, 99.5])

im = ax_ds.imshow(
    arr.T,
    aspect="auto",
    origin="lower",
    extent=extent,
    vmin=vmin,
    vmax=vmax,
)

ax_ds.set_xlabel("Sample")
ax_ds.set_ylabel("Frequency (MHz)")

plt.tight_layout()
plt.show()
