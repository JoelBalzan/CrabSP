# Polarization Calibration (Pcal)

---

## Stage 1 — Build the calibration database


### 1a. Pre-process cal file  (`build_cal_database`, `calibration.py:67`)
Run pazi, then

```
pam -T -e dzT uwl_*.cf.pazi
```
 - `-T` -- t-scrunch the calibrator file, with extension dzT

### 1b. Frequency-match the calibrator to the fold  (`auto_scrunch_cal`, `calibration.py:121`)
Crab data is 32MHz BW at 3968MHz, and cal obs is full UWL band so need to slice the cal.

- Load the pcal dzT file with psrchive and `remove_chan()` to cut down to just the
  sub-band overlapping the fold centre/bandwidth.
- If not already `fold_nchan` channels, `pam -f <factor>` frequency-scrunches
  it to the same channel count as the fold.

Then,

```
pac -w -u dzT
```

- `-w` — write a calibration database
- `-u dzT` — look for dzT cal file

---

## Stage 2 — Apply `pac` to each folded candidate

After `dspsr` produces the folded `.ar` for a Crab SP candidate,

### 2a. Receiver-metadata patch  (`fix_dspsr_receiver`, `archive.py:57`)
dspsr outputs wrong header info for some reason,

```
psredit -c 'rcvr:hand=-1,rcvr:sa=0,be:dcc=1' -m <cand.ar>
```

- `rcvr:hand=-1` — feeds A/B hand (must match the calibrator's convention)
- `rcvr:sa=0` — symmetry angle (deg) of the receiver
- `be:dcc=1` — backend downconversion-conjugation corrected

### 2b. Frequency-scrunch + CHAN_BW fix  (`fscrunch_to_nchan`, `archive.py:153`)

- `a.fscrunch_to_nchan(fold_nchan)` if the fold has a different channel count.
- `psredit -c 'chan_bw=<expected>' -m` patches the FITS `CHAN_BW` keyword —
  dspsr's `-x` FFT override can write a wrong channel bandwidth, which would
  break `pac`'s channel-to-channel mapping.

### 2c. Channel-grid alignment  (`fix_archive_frequencies`, `archive.py:91`, only if `*.dzT` present)

**Only needed when pac does frequency matching / per-channel mapping** (i.e.
when you *don't* pass `-F -b`, and/or you keep `-a`). With the actual call
below (`-F -b -c -T` and no `-a`), pac neither matches centre frequency/bandwidth
nor does per-channel mapping, so this step does not apply.

- `fold_a.update_centre_frequency()` — dspsr labels channels with lower-edge
  frequencies while the calibrator uses midpoints; this fixes the fold labels.
- `cal_a.set_centre_frequency(cal_centre + shift)` — shifts the calibrator to
  the fold's centre frequency, then `update_centre_frequency()` so both
  archives sit on the same grid and `pac` can match channels.

### 2d. The actual calibration  (`apply_pac`, `calibration.py:6`)

```
pac -F -b -c -T -d <db> -e calib <cand.ar>
```

- `-d <db>` — cal database
- `-F` — matching criterion: do not try to match frequencies
- `-b` — matching criterion: do not try to match bandwidths
- `-c` — matching criterion: take closest sky coordinates (no maximum distance)
- `-T` — matching criterion: do not try to match times
- (*no `-a`*) — without per-channel mode, pac frequency-averages the data
  onto the calibrator's channel grid; combined with `-FbT` this means it makes
  no frequency/grid association, so the 2c alignment is **not** required.

> **On `-a` and the 2c grid alignment.** `-a` is a separate *application
> mode*: with it pac applies a per-channel calibration model; without it pac
> frequency-averages the data to the calibrator's channel count. The 2c
> frequency-grid alignment is needed **only in the per-channel case** (when
> pac must map individual channels, i.e. with `-a` and/or without the `-Fb`
> relaxation). With `-F -b` (no frequency/bandwidth matching) and no `-a`
> (no per-channel mapping), pac does no frequency-grid association, so 2c can
> be skipped. `-c` takes the closest sky coordinates (no maximum distance),
> which with the matched Crab and UWL positions just removes a rejection
> criterion.

---


## Commands only

```
# --- build (once) ---
pac -F -b -c -T -w -k database.txt -u dzT
psrchive: remove_chan() to Crab zoom band; pam -f <factor> -e dzT -u dir <polncal>   # scrunch cal to match desired nchan

# --- apply (per Crab candidate) ---
psredit -c 'rcvr:hand=-1,rcvr:sa=0,be:dcc=1' -m cand.ar
pac -F -b -c -T -d database.txt -e calib cand.ar
#   (2c grid alignment skipped: -F -b disables freq/bandwidth matching and
#    no -a means no per-channel grid mapping)
```
