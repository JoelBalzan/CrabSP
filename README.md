# CrabSP — Crab pulsar single-pulse search and extraction

A pipeline for finding **single pulses from the Crab pulsar** in baseband data and
extracting full-Stokes cutouts of every detected burst.

The workflow has two stages:

1. **Search** (`tx.sh`) — run [transientX](https://github.com/ypmen/TransientX)
   boxcar searches at multiple time resolutions over dedispersed filterbanks.
2. **Extract** (`extract_cands.py`) — match the resulting candidates to the raw
   `.dada` baseband fragments, form full-Stokes cutouts on the fly with
   `digifil` (from [dspsr](http://dspsr.sourceforge.net/)), convert the
   coherency products to IQUV, and save each pulse as a `.npz` cube.

## Requirements

- [transientX](https://github.com/ypmen/TransientX) (`transientx_fil`)
- [dspsr](http://dspsr.sourceforge.net/) (`digifil`)
- Python ≥ 3.6 with:
  - [`sigpyproc3`](https://github.com/FRBers/sigpyproc3)
  - `numpy`
  - `matplotlib` (only for `--plot` and `plot_fil.py`)

## Stage 1 — single-pulse search (`tx.sh`)

Search every `*.fil` in the current directory at a chosen time resolution:

```bash
./tx.sh 4us        # single resolution: 1us | 4us | 5us | 10us | 20us | 40us
./tx.sh all        # run all resolutions
```

Each resolution is searched with a boxcar length tuned to the Crab's pulse
width at that sampling, dedispersed at the pulsar's dispersion measure
(`--dms 56.65`, `--ddm 0.005`, 30 DM trials), with a S/N threshold of 7.

Output: candidate lists written to `cands/<res>/crab_<res>*.cands`, one per
filterbank. Each line holds:

```
<beam> <cand_id> <MJD> <DM> <width_ms> <SNR> <source_fil>
```

## Stage 2 — candidate extraction (`extract_cands.py`)

Match each candidate to the correct raw `.dada` fragment and save a
full-Stokes cutout of the pulse.

```bash
python3 extract_cands.py \
    --cand-dir cands \
    --workdir /path/to/raw/dada/dir \
    --outdir cutouts \
    --min-snr 5 \
    --plot
```

### How it works

1. **Fragment index** — every `<file>.dada.fil` in `--workdir` is read with
   sigpyproc to get its `tstart` / `tsamp` / `nsamples`. Each fragment's MJD
   span is computed and the fragments are sorted chronologically.
2. **MJD matching** — each candidate's MJD is matched to the fragment that
   contains it (with a 10 ms tolerance).
3. **Event clustering** — candidates are grouped into events on the MJD axis
   (gap `--cluster-gap-ms`, default 3 ms ≈ the Crab main-pulse window; rotations
   are 33.4 ms apart). Each event yields one cutout, centred on the highest-SNR
   detection, with a fixed `--window-s` (default 6 ms).
4. **digifil extraction** — forming a filterbank on the fly from voltages
   (with `-F`, see [time resolution](#time-resolution-digifil-fft)) discards
   FFT settle/edge regions, so very short requests return
   zero samples. To work around this a block of at least `--digifil-min-block`
   (default 6 s) is requested centred on the pulse, with full coherency
   products (`-d 4`), coherent dedispersion at the candidate DM (`-D ... -K`),
   float output (`-b -32`), and no rescaling (`-I 0`).
5. **Coherency → Stokes** — the cutout (AA, BB, CR, CI) is converted to
   **I, Q, U, V** via `I=AA+BB, Q=AA−BB, U=2CR, V=2CI`.
6. **Trimming** — the large block is trimmed client-side to the desired window
   around the candidate MJD.
7. **Save** — the trimmed cube is written as `cand<id>_<mjd>_dm<dm>_iquv.npz`
   along with the timing/frequency axes and candidate metadata.

### Output

Each `.npz` contains:

| Key              | Description                                          |
|------------------|------------------------------------------------------|
| `stokes`         | float32 cube, shape `(4, nchan, nsamp)`, order I,Q,U,V |
| `pol_order`      | `['I','Q','U','V']`                                   |
| `cand_id`, `cand_mjd`, `cand_dm`, `cand_width_ms`, `cand_snr` | candidate parameters |
| `source_dada`    | raw `.dada` fragment the pulse came from              |
| `tstart_mjd`     | MJD of the first sample of the trimmed cube           |
| `tsamp_s`        | sampling interval (s)                                 |
| `nsamp`          | samples along the time axis                           |
| `fch1_mhz`, `foff_mhz`, `nchan` | frequency axis                    |
| `window_s`       | requested cutout window (s)                           |
| `digifil_seek_s`, `digifil_dur_s` | digifil block actually requested   |

`--plot` additionally writes a diagnostic PNG per candidate (full-Stokes profile,
bandpass-subtracted Stokes-I dynamic spectrum plus Q/U/V time series), scrunched
to the search resolution of the candidate file. `--plot-no-tscrunch` keeps the
native extracted resolution instead.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cand-dir DIR` | — | recursively find `*.cands` under DIR (alternative to `--cand_files`) |
| `--cand_files F...` | — | explicit `.cands` files |
| `--workdir DIR` | `.` | directory containing `*.dada` + matching `*.dada.fil` |
| `--outdir DIR` | `cutouts` | output directory (per-resolution subdirs created) |
| `--min-snr S` | `0.0` | drop candidates below this S/N |
| `--cluster-gap-ms M` | `3.0` | MJD gap that separates one pulse event from the next (ms) |
| `--window-s S` | `0.006` | fixed cutout window (s), centred on the burst |
| `--digifil-min-block S` | `6.0` | min duration to request from digifil (see note above) |
| `--digifil-bin` | `digifil` | path to the digifil binary |
| `--digifil-fft` | `32` | digifil `-F`, number of channels; sets cutout time resolution (see below) |
| `--plot-tscrunch-us US` | search res | override the diagnostic-PNG time scrunch (µs) |
| `--plot-no-tscrunch` | off | plot PNGs at the native extracted resolution |
| `--keep-fil` | off | keep the intermediate `.fil` cutout |
| `--plot` | off | write a diagnostic PNG per candidate |

### Time resolution (`--digifil-fft`)

The cutout time resolution is set by digifil's FFT factor `-F`. For the 32 MHz
band used in `tx.sh`, `-F 32` produces 32 × 1 MHz channels at 1 µs raw `dt`.
Increase `-F` (64, 128, …) for finer time resolution:

```bash
python3 extract_cands.py --cand-dir cands --workdir . --digifil-fft 128 --plot
```

This is independent of the transientX search resolution (which only sets the
window sizes and the default plot scrunch); the saved `.npz` metadata and the
plotting always follow the header of the actual cutout. Use
`--plot-no-tscrunch` to view the finer native resolution in the diagnostic PNGs.

## Helper — `plot_fil.py`

Quick look at a filterbank: dynamic spectrum (bandpass-subtracted) and total
power profile. --ts 4 = 

```bash
python3 plot_fil.py cutouts/cand123_..._iquv.fil --ts 4
```

## Typical workflow

```bash
# 1. Baseband -> searchable filterbanks (dspsr; per fragment, e.g.)
digifil -F 32 -d 2 -D 56.65 -K -b 8 -o crab.fil crab.dada

# 2. Search single pulses at several time resolutions
./tx.sh all

# 3. Extract full-Stokes cutouts of every detection
python3 extract_cands.py --cand-dir cands --workdir . --outdir cutouts \
    --min-snr 5 --plot
```

## Notes / caveats

- The candidate MJD is in **UTC**; the fragment `.fil` headers must use the
  same clock, otherwise candidates will not match a fragment.
- `digifil` refuses to overwrite existing files, so stale cutouts are deleted
  before extraction.
- Duplicate candidates (same source fragment and offset) are skipped.
- `extract_cands.py` performs a sanity check on the trimmed cube: the fraction
  of negative Stokes-I samples (`I = AA + BB`) should be near zero. A large
  negative fraction indicates the pol-axis reshape is wrong — inspect the cube
  by hand.

## License

MIT — see [LICENSE](LICENSE).
