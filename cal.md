# Polarization Calibration (Pcal)

---

## Stage 1 — Build the calibration database
### 1a. Pre-process cal file 
Run pazi, then

```
pam -T -e dzT uwl_*.cf.pazi
```
 - `-T` -- t-scrunch the calibrator file, with extension dzT

### 1b. Frequency-match the calibrator to the fold 
Crab data is 32MHz zoom BW at 3968MHz, and pcal obs is full UWL band so need to slice the cal.

- Load the pcal dzT file with psrchive and `remove_chan()` to cut down to just the
  sub-band overlapping the zoom centre/bandwidth.
- If not already cal_nchan == Crab_nchan, `pam -f <factor> pcal` frequency-scrunches
  it to the same channel count requested for the single pulse data cutouts 
  (i.e., fft length. Here I'm looking at Nyquist sampling rate, so the cal is averaged to 1 chan).

Then,

```
pac -w -u dzT
```

- `-w` — write calibration database
- `-u dzT` — look for dzT cal file

---

## Stage 2 — Apply `pac` to each folded candidate

After we get Crab SP candidate .ar files,

### 2a. Receiver-metadata patch
dspsr outputs wrong header info for some reason,

```
psredit -c 'rcvr:hand=-1,rcvr:sa=0,be:dcc=1' -m <cand.ar>
```

### 2b. The calibration

```
pac -F -b -c -T -d <database> -e calib <cand.ar>
```

- `-F` — do not try to match frequencies
- `-b` — do not try to match bandwidths
- `-c` — take closest sky coordinates (no maximum distance)
- `-T` — Take closest time (no maximum interval)

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




-seek = fold-start MJD = cand_mjd − window_s/2 (margin), so phase 0 ≈ the burst.
-turns 1 -c <period> = one spin fold, args.fold_period (0.0334 s Crab).
-cepoch <seek_mjd> makes the seek epoch the phase-0 reference (burst lands in first bins).
-D <dm> -K = coherent dedispersion at candidate DM.
-F 1 = --nchan; -d 4 = PP,QQ,RePQ,ImPQ coherency products; -b <nbin> = auto-computed nbin = ceil(period × BW/nchan) (main.py:90) = 1 068 800 bins at 32 MHz Nyquist (~31 ns/bin).
-A = apply weights; -e ar -O prefix for the output archive.
-x nfft + -U MB only kick in because nbin > 32768 (folding.py:65-72): nfft = power of 2 ≥ 2·nbin = 4 194 304, -U = ~1.07 GB.
-derotate -rm only with --rm; -cont only when the fold spans >1 fragment.