# CrabSP — Crab pulsar single-pulse search and extraction

A pipeline for finding **single pulses from the Crab pulsar** in baseband data and
extracting full-Stokes cutouts of every detected burst.

The workflow has two stages:

1. **Search** (`tx.sh`) — run [transientX](https://github.com/ypmen/TransientX)
   boxcar searches at multiple time resolutions over dedispersed filterbanks.
2. **Extract** (`extract_cands.py`) — match the resulting candidates to the raw
   `.dada` baseband fragments, coherently dedisperse and **fold** each burst with
   `dspsr`, calibrate the folded archive with `pac` (flat SingleAxis model), and
   save a full-Stokes `.npz` cube plus a polarimetric PNG per candidate.

## Requirements

- [transientX](https://github.com/ypmen/TransientX) (`transientx_fil`)
- [dspsr](http://dspsr.sourceforge.net/) (`dspsr`) with psrchive (`pac`, `pam`,
  `psredit`)
- Python ≥ 3.6 with:
  - [`sigpyproc3`](https://github.com/FRBers/sigpyproc3)
  - `numpy`
  - `matplotlib` (for the diagnostic PNGs)

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

Match each candidate to the correct raw `.dada` fragment, fold it, calibrate it,
and save the burst's Stokes profile.

```bash
python3 extract_cands.py \
    --cand-dir cands \
    --workdir /path/to/raw/dada/dir \
    --outdir cutouts \
    --min-snr 5 \
    --keep-ar \
    --dm 56.65 \
    --rm -45 \
    -F 1
```

### How it works

1. **Fragment index** — every `<file>.dada.fil` in `--workdir` is read with
   sigpyproc to get its `tstart` / `tsamp` / `nsamples`. Each fragment's MJD
   span is computed and the fragments are sorted chronologically.
2. **MJD matching** — each candidate's MJD is matched to the fragment that
   contains it (with a 10 ms tolerance).
3. **Event clustering** — candidates are grouped into events on the MJD axis
   (gap `--cluster-gap-ms`, default 0 = no clustering). Each event yields one
   fold centred on the highest-SNR detection, with a fixed `--window-s`
   (default 3 ms).
4. **Fast-pac folding** — the raw fragment is cropped to a short window around
   the pulse (a pre-roll covers dspsr's overlap-save history), then `dspsr`
   coherently dedisperses (`-D -K`) and folds one spin period (`-b nbin`,
   `-F nchan`, coherency products `-d 4`) anchored so the burst lands in the
   first phase bins. `--rm` applies coherent Faraday derotation.
5. **Calibration** — `pac` is applied with a flat SingleAxis model
   (`--pac-flags -F -b -c -T`) against a grid-matched calibration database
   that is auto-generated/auto-scrunched from the native PolnCal (see
   `cal.md`).
6. **Trim + save** — the trimmed Stokes block is written as
   `cand<id>_<mjd>_dm<dm>_iquv.npz` along with the timing/frequency axes and
   candidate metadata, and (by default) a polarimetric profile PNG per
   candidate.

### Output

Each `.npz` contains:

| Key              | Description                                          |
|------------------|------------------------------------------------------|
| `stokes`         | float32 cube, shape `(4, nchan, nsamp)`, order I,Q,U,V |
| `pol_order`      | `['I','Q','U','V']`                                   |
| `cand_id`, `cand_mjd`, `cand_dm`, `cand_width_ms`, `cand_snr` | candidate parameters |
| `source_dada`    | raw `.dada` fragment the pulse came from              |
| `tstart_mjd`     | MJD of the first sample of the trimmed cube           |
| `tsamp_s`        | sampling interval (s) = fold period / nbin            |
| `nsamp`          | samples along the time axis                           |
| `fch1_mhz`, `foff_mhz`, `nchan` | frequency axis                    |
| `window_s`       | requested cutout window (s)                           |
| `fold_seek_mjd`, `fold_period`, `fold_nbin`, `fold_nchan` | fold geometry |
| `method`         | `dspsr-fast-cropped`                                  |
| `calib_applied`, `calib_file`, `pac_flags` | calibration provenance    |
| `rm`             | rotation measure used (NaN if none)                   |

A diagnostic PNG (`*_iquv_profile.png`) is written per candidate by default;
pass `--no-plot` to disable.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cand-dir DIR` | — | recursively find `*.cands` under DIR (alternative to `--cand-files`) |
| `--cand-files F...` | — | explicit `.cands` files |
| `--workdir DIR` | `.` | directory containing `*.dada` + matching `*.dada.fil` |
| `--outdir DIR` | `cutouts` | output directory (per-resolution subdirs created) |
| `--min-snr S` | `0.0` | drop candidates below this S/N |
| `--cluster-gap-ms M` | `0.0` | MJD gap that separates one pulse event from the next (ms); 0 = no clustering |
| `--window-s S` | `0.003` | fixed cutout window (s), centred on the burst |
| `-F N, --nchan N` | `8` | dspsr `-F` folding channels |
| `--fold-period P` | `0.0334` | fold period (s), passed to dspsr `-c` |
| `--rm RM` | — | rotation measure for coherent Faraday derotation |
| `--dm DM` | — | override DM (pc/cm³) for all candidates |
| `--keep-ar` | off | keep the intermediate `.ar`/`.calib` archives |
| `--no-plot` | off | disable the diagnostic PNG per candidate (plot is on by default) |
| `--calib FILE` | — | pac calibration model (`pac -A`) |
| `--calib-db FILE` | — | pac calibration database (`pac -d`) |
| `--cal-dir DIR` | `--workdir` | directory holding calibration material |
| `--cal-db-name NAME` | `database.txt` | auto-detected/generated calibration database name |
| `--no-cal` | off | disable automatic pac calibration |
| `--pac-flags FLAGS` | `-F -b -c -T` | flags passed to every pac step |

## Typical workflow

```bash
# 1. Baseband -> searchable filterbanks (digifil; per fragment, e.g.)
digifil -F 16 -d 1 -b 8 -I 0 -o crab.fil crab.dada

# 2. Search single pulses at several time resolutions
./tx.sh all

# 3. Fold + calibrate + extract full-Stokes cutouts of every detection
python3 extract_cands.py --cand-dir cands --workdir . --outdir cutouts \
    --min-snr 5 --dm 56.65 --rm -45 -F 1
```

`pipeline/run_pipeline.sh` automates all three steps for a list of observation
directories (stage toggles `DO_FIL`/`DO_SEARCH`/`DO_EXTRACT`).

## Notes / caveats

- The candidate MJD is in **UTC**; the fragment `.fil` headers must use the
  same clock, otherwise candidates will not match a fragment.
- The folded archive is anchored with `-seek`/`-cepoch` so phase 0 = the
  candidate MJD; dspsr's overlap-save filter needs ~131 ms of pre-roll history,
  which the fast-pac crop provides.
- `extract_cands.py` performs a sanity check on the trimmed cube: the fraction
  of negative Stokes-I samples should be near zero. A large negative fraction
  indicates a pol-axis/coherency reshape problem — inspect the cube by hand.

## License

MIT — see [LICENSE](LICENSE).