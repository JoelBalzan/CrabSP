"""psrchive archive manipulation: metadata fixes, frequency alignment, reading."""
import subprocess

import numpy as np


def fix_fits_chan_bw(ar_path, expected_chan_bw, psredit_bin='psredit'):
    """Patch FITS-level CHAN_BW keyword using psredit.

    dspsr's -x FFT override can write wrong CHAN_BW; psrchive's unload()
    faithfully writes back the bad value, so psredit is needed to fix it.

    Returns True if the patch was applied (or not needed), False on failure.
    """
    if expected_chan_bw <= 0:
        return True

    try:
        import psrchive as _psr
        a = _psr.Archive.load(str(ar_path))
        freqs = list(a.get_frequencies())
        actual_cbw = (abs(freqs[1] - freqs[0])
                      if len(freqs) >= 2 else 0.0)
        nchan = a.get_nchan()
    except Exception as e:
        print(f"  WARNING: could not load archive for CHAN_BW check: {e}")
        return True

    if abs(actual_cbw - expected_chan_bw) < 0.01 and nchan > 0:
        print(f"  CHAN_BW OK ({actual_cbw:.3f} MHz)")
        return True

    cmd = [psredit_bin, '-c', f'chan_bw={expected_chan_bw:.6f}',
           '-m', str(ar_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
    except Exception as e:
        print(f"  WARNING: psredit failed: {e}")
        return False
    if r.returncode != 0:
        print(f"  WARNING: psredit returned {r.returncode}: "
              f"{r.stderr.strip()}")
        return False

    try:
        a2 = _psr.Archive.load(str(ar_path))
        freqs2 = list(a2.get_frequencies())
        new_cbw = (abs(freqs2[1] - freqs2[0])
                   if len(freqs2) >= 2 else 0.0)
        print(f"  psredit chan_bw patched: {actual_cbw:.3f} -> "
              f"{new_cbw:.3f} MHz (expected {expected_chan_bw:.3f})")
    except Exception:
        print(f"  psredit chan_bw set to {expected_chan_bw:.3f} MHz")
    return True


def fix_dspsr_receiver(ar_path, rcvr_hand=-1, rcvr_sa=0,
                        be_dcc=1, psredit_bin='psredit'):
    """Patch dspsr output archives to match native PolnCal metadata.

    Fixes rcvr:sa (symmetry angle) and be:dcc (downconversion conjugation
    corrected) to match the calibrator's native values.

    Returns True on success, False on failure.
    """
    cmd = [psredit_bin, '-c',
           f'rcvr:hand={rcvr_hand},rcvr:sa={rcvr_sa},be:dcc={be_dcc}',
           '-m', str(ar_path)]

    cmd2 = f"{psredit_bin} {ar_path} | grep -e 'rcvr:hand' -e 'rcvr:sa' -e 'be:dcc'"
    print(f"  psredit pre-patch check: {cmd2}")
    r2 = subprocess.run(cmd2, capture_output=True, text=True, shell=True)
    print(r2.stdout)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        print(f"  psredit post-patch check: {cmd2}")
        r2 = subprocess.run(cmd2, capture_output=True, text=True, shell=True)
        print(r2.stdout)
    except Exception as e:
        print(f"  WARNING: psredit receiver patch failed: {e}")
        return False
    if r.returncode != 0:
        print(f"  WARNING: psredit returned {r.returncode}: "
              f"{r.stderr.strip()}")
        return False
    print(f"  psredit receiver patch: hand={rcvr_hand}, "
          f"sa={rcvr_sa}deg, be:dcc={be_dcc}")
    return True


def fix_archive_frequencies(fold_path, cal_path, fold_nchan):
    """Align fold and calibrator channel grids so pac can match them.

    dspsr's -F channel grid uses lower-edge labels while the calibrator uses
    midpoints.  Both are fixed via update_centre_frequency() and
    set_centre_frequency() so their labels and stored centres match.
    """
    try:
        import psrchive as _psr
    except ImportError:
        print("  WARNING: psrchive not available for frequency alignment")
        return

    try:
        fold_a = _psr.Archive.load(str(fold_path))
    except Exception as e:
        print(f"  WARNING: could not load fold for freq alignment: {e}")
        return

    fold_freqs = list(fold_a.get_frequencies())
    if len(fold_freqs) < 2:
        return
    fold_cbw = fold_freqs[1] - fold_freqs[0]
    fold_mean = sum(fold_freqs) / len(fold_freqs)
    fold_centre = fold_a.get_centre_frequency()
    print(f"  fold freqs: [{fold_freqs[0]:.1f}..{fold_freqs[-1]:.1f}] "
          f"mean={fold_mean:.1f} stored_centre={fold_centre:.1f}")

    if abs(fold_mean - fold_centre) < 0.01:
        print("  fold centres already consistent, no alignment needed")
        return

    fold_a.update_centre_frequency()
    fold_new_centre = fold_a.get_centre_frequency()
    print(f"  fold update_centre_frequency: {fold_centre:.1f} -> "
          f"{fold_new_centre:.1f}")
    fold_a.unload(str(fold_path))

    try:
        cal_a = _psr.Archive.load(str(cal_path))
    except Exception as e:
        print(f"  WARNING: could not load cal for freq alignment: {e}")
        return

    cal_freqs = list(cal_a.get_frequencies())
    cal_cbw = cal_freqs[1] - cal_freqs[0] if len(cal_freqs) >= 2 else 0
    cal_centre = cal_a.get_centre_frequency()
    shift = fold_new_centre - cal_centre
    print(f"  cal freqs:   [{cal_freqs[0]:.1f}..{cal_freqs[-1]:.1f}] "
          f"centre={cal_centre:.1f}")
    print(f"  cal shift: {cal_centre:.1f} -> {cal_centre + shift:.1f} "
          f"(delta={shift:+.1f} MHz)")

    cal_a.set_centre_frequency(cal_centre + shift)
    cal_a.update_centre_frequency()
    cal_new_centre = cal_a.get_centre_frequency()
    cal_new_freqs = list(cal_a.get_frequencies())
    print(f"  cal final:   [{cal_new_freqs[0]:.1f}..{cal_new_freqs[-1]:.1f}] "
          f"centre={cal_new_centre:.1f}")
    cal_a.unload(str(cal_path))


def fscrunch_to_nchan(ar_path, target_nchan, expected_chan_bw=0.0,
                      pam_bin='pam', psredit_bin='psredit'):
    """Ensure archive has target_nchan channels with correct chan_bw.

    Frequency-scrunches if needed, then patches FITS CHAN_BW via psredit.
    Returns the (possibly updated) path.
    """
    if target_nchan <= 0:
        return ar_path

    try:
        import psrchive as _psr
    except ImportError as e:
        print(f"  WARNING: psrchive not available for fscrunch: {e}")
        return ar_path

    try:
        a = _psr.Archive.load(str(ar_path))
    except Exception as e:
        print(f"  WARNING: could not load archive for fscrunch: {e}")
        return ar_path

    nchan = a.get_nchan()
    if nchan <= 0:
        return ar_path

    freqs = list(a.get_frequencies())
    old_cbw = abs(freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0
    need_scrunch = (nchan != target_nchan)

    if need_scrunch:
        print(f"  archive nchan={nchan}, expected {target_nchan}, "
              f"chan_bw={old_cbw:.3f} MHz")
        if nchan % target_nchan == 0:
            a.fscrunch_to_nchan(target_nchan)
            a.unload(str(ar_path))
            print(f"  fscrunch_to_nchan {nchan} -> {target_nchan}")

    fix_fits_chan_bw(ar_path, expected_chan_bw, psredit_bin=psredit_bin)
    return ar_path


def log_archive_info(ar_path, label="archive"):
    """Print a one-line psredit-style summary of an archive."""
    try:
        import psrchive as _psr
        a = _psr.Archive.load(str(ar_path))
        freqs = list(a.get_frequencies())
        nchan = a.get_nchan()
        bw = a.get_bandwidth()
        centre = a.get_centre_frequency()
        cbw = abs(freqs[1] - freqs[0]) if len(freqs) >= 2 else abs(bw) / max(nchan, 1)
        state = str(a.get_state())
        nsub = a.get_nsubint()
        nbin = a.get_nbin()
        print(f"  {label}: nsub={nsub} nbin={nbin} nchan={nchan} "
              f"bw={bw:.1f} MHz centre={centre:.1f} MHz "
              f"chan_bw={cbw:.3f} MHz state={state}")
    except Exception as e:
        print(f"  WARNING: could not log archive info: {e}")


def read_ar_stokes(ar_path, cand_mjd=None, seek_mjd=None):
    """Read a folded/calibrated psrchive archive into (4, nchan, nbin) IQUV.

    Handles both Coherence-state and Stokes-state archives. When the archive
    has multiple subints (dspsr folded past the candidate to EOF), the subint
    containing the candidate is selected by MJD; otherwise subint 0 is used.
    cand_mjd is preferred, seek_mjd is the fallback. If neither is given or
    no interval contains the MJD, the closest subint is chosen.

    Returns (stokes, meta) with meta = {state, nchan, nbin, freqs_mhz,
    fch1_mhz, foff_mhz, epoch_mjd, chosen_subint}.
    """
    try:
        import psrchive
    except ImportError as e:
        raise RuntimeError(f"psrchive python bindings not available ({e}); "
                           f"the dspsr route needs them to read the .ar") from e
    a = psrchive.Archive.load(str(ar_path))
    state = str(a.get_state())
    d = a.get_data()
    if d.ndim != 4:
        raise RuntimeError(f"{ar_path}: get_data shape {d.shape} — "
                           f"expected (nsub, npol, nchan, nbin)")
    nsub, npol, nchan, nbin = d.shape
    if npol != 4:
        raise RuntimeError(f"{ar_path}: got {npol} pols (state={state}); "
                           f"need the full 4 coherency products")
    chosen = 0
    if nsub > 1:
        target_mjd = cand_mjd if cand_mjd is not None else seek_mjd
        if target_mjd is not None:
            # Use start_time + duration to find the interval containing the
            # candidate. dspsr's first subint can be truncated (short dur) and
            # subints are not necessarily stored in chronological order, so we
            # must check all.
            best_idx = None
            best_dist = None
            for idx in range(nsub):
                try:
                    integ = a.get_Integration(idx)
                    st = integ.get_start_time()
                    st_mjd = float(st.intday() + st.fracday())
                    dur_s = float(integ.get_duration())
                    end_mjd = st_mjd + dur_s / 86400.0
                    # Prefer interval containing target
                    if st_mjd <= target_mjd < end_mjd:
                        best_idx = idx
                        break
                    dist = min(abs(target_mjd - st_mjd), abs(target_mjd - end_mjd))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                except Exception:
                    continue
            if best_idx is not None:
                chosen = best_idx
                if chosen != 0:
                    print(f"    archive has {nsub} integrations; selecting subint {chosen} "
                          f"(closest to cand MJD {target_mjd:.12f}) instead of subint 0")
                else:
                    print(f"    archive has {nsub} integrations; subint 0 is the closest to cand MJD")
            else:
                print(f"    archive has {nsub} integrations (dspsr folded past the candidate to EOF); using subint 0")
                chosen = 0
        else:
            print(f"    archive has {nsub} integrations (dspsr folded past the "
                  f"candidate to EOF); using subint 0, the temporally correct "
                  f"one given the -seek anchor")
            chosen = 0
        d = d[chosen:chosen+1]
        # For meta, use the chosen integration's epoch
        try:
            m = a.get_Integration(chosen).get_epoch()
            epoch_mjd = float(m.intday() + m.fracday())
        except Exception:
            epoch_mjd = None
        # Fall through to common handling below, but we already have epoch
        # Save it for later return
        _chosen_epoch = epoch_mjd
    else:
        _chosen_epoch = None
    if 'Stokes' in state:
        stokes = d[0]
    elif 'Coherence' in state:
        PP, QQ, RPQ, IPQ = d[0, 0], d[0, 1], d[0, 2], d[0, 3]
        stokes = np.stack([PP + QQ, PP - QQ, 2 * RPQ, 2 * IPQ], axis=0)
    else:
        raise RuntimeError(f"{ar_path}: unsupported archive state '{state}'")
    freqs = np.asarray(a.get_frequencies(), dtype=float)
    if freqs.shape[0] != nchan:
        raise RuntimeError(f"{ar_path}: get_frequencies {freqs.shape} != "
                           f"nchan {nchan} — archive state inconsistent")
    foff_mhz = float(freqs[1] - freqs[0]) if nchan > 1 else 0.0
    # Prefer the chosen subint's epoch when nsub>1, otherwise archive epoch
    if nsub > 1 and _chosen_epoch is not None:
        epoch_mjd = _chosen_epoch
    else:
        try:
            m = a.get_epoch()
            # psrchive MJD has intday/fracday methods, not val in newer bindings
            if hasattr(m, 'intday'):
                epoch_mjd = float(m.intday() + m.fracday())
            else:
                epoch_mjd = float(getattr(m, 'val', m.intday + m.fracday))
        except Exception:
            epoch_mjd = None
        if nsub > 1 and _chosen_epoch is not None and epoch_mjd is None:
            epoch_mjd = _chosen_epoch
    return stokes, {'state': state, 'nchan': nchan, 'nbin': nbin,
                    'freqs_mhz': freqs, 'fch1_mhz': float(freqs[0]),
                    'foff_mhz': foff_mhz, 'epoch_mjd': epoch_mjd,
                    'chosen_subint': chosen if nsub > 1 else 0}


def trim_folded_to_window(stokes, epoch_mjd, tsamp_s, window_s):
    """Centre the extracted window on the detected peak of the folded profile.

    Robust to absolute-MJD offsets: the burst is located by peak-finding
    in the folded Stokes-I profile rather than assumed to be at bin 0.

    Returns (trimmed_stokes, trimmed_tstart_mjd).
    """
    nbin = stokes.shape[-1]
    profile = stokes[0].mean(axis=0)
    peak = int(np.argmax(profile))
    win_bins = int(round(window_s / tsamp_s))
    if win_bins >= nbin:
        return stokes, epoch_mjd
    half = win_bins // 2
    i0 = min(max(peak - half, 0), nbin - win_bins)
    i1 = i0 + win_bins
    tstart_mjd = epoch_mjd + (i0 * tsamp_s) / 86400.0
    return stokes[..., i0:i1], tstart_mjd
