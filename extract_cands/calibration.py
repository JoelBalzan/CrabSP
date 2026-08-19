"""Polarization calibration: pac database management, calibration application."""
import subprocess
from pathlib import Path


def apply_pac(ar_path, calib=None, calib_db=None, pac_bin='pac',
              pac_flags='', out_ext='calib'):
    """Calibrate a folded .ar with pac, writing `<input>.<ext>`.

    Either calib (pac -A) or calib_db (pac -d) must be set.
    Returns the calibrated archive path, or None on failure.
    """
    cmd = [pac_bin]
    if pac_flags:
        cmd += pac_flags.split()
    if calib_db:
        cmd += ['-d', str(calib_db)]
    elif calib:
        cmd += ['-A', str(calib)]
    else:
        raise ValueError('apply_pac requires --calib or --calib-db')
    cmd += ['-e', out_ext, str(ar_path)]
    print("\nRunning pac")
    print(" ".join(cmd))
    try:
        r = subprocess.run(cmd, timeout=600.0)
    except subprocess.TimeoutExpired:
        print("    pac TIMED OUT")
        return None
    if r.returncode != 0:
        print(f"    pac FAILED (rc={r.returncode}); "
              f"re-run the printed command manually to see the error")
        return None
    out = ar_path.with_suffix(f'.{out_ext}')
    if not out.exists() or out.stat().st_size == 0:
        out_p = ar_path.with_suffix(f'.{out_ext}P')
        if out_p.exists() and out_p.stat().st_size > 0:
            out = out_p
        else:
            print(f"    pac ran but wrote nothing at {out} or {out_p}")
            return None
    return out


def find_cal_extensions(cal_dir):
    """Extensions of calibration material present in cal_dir, in pac search order.

    .dzT (tscrunched, zapped cal obs) is preferred over raw .cf; avfluxcal
    and pcm are included when present.
    """
    cal_dir = Path(cal_dir)
    exts = []
    if any(cal_dir.glob('*.dzT')):
        exts.append('dzT')
    elif any(cal_dir.glob('*.cf')):
        exts.append('cf')
    if any(cal_dir.glob('*avfluxcal*')):
        exts.append('avfluxcal')
    if any(cal_dir.glob('*.pcm')):
        exts.append('pcm')
    return exts


def build_cal_database(cal_dir, db_path, pac_bin='pac', pac_flags='',
                       use_exts=None):
    """Generate a pac calibration database with `pac -w -k`.

    Returns (db_path, None) on success or (None, note) on failure.
    """
    cal_dir = Path(cal_dir).resolve()
    db_path = Path(db_path).resolve()
    if use_exts is not None:
        exts = use_exts
    else:
        exts = find_cal_extensions(cal_dir)
    if not exts:
        return None, (f"no calibration material in {cal_dir} "
                      f"(looked for .dzT/.cf, *avfluxcal*, *.pcm)")
    cmd = [pac_bin]
    if pac_flags:
        cmd += pac_flags.split()
    cmd += ['-w', '-k', str(db_path)]
    for e in exts:
        cmd += ['-u', e]
    print("\nGenerating calibration database (pac -w)")
    print(f"  cal files dir : {cal_dir}")
    print("  " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(cal_dir), timeout=600.0)
    except subprocess.TimeoutExpired:
        return None, "pac -w (database build) TIMED OUT after 600s"
    if r.returncode != 0:
        return None, ("pac -w FAILED (rc={r.returncode}); re-run the printed "
                      "command manually to see the error")
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None, f"pac -w exited 0 but wrote no database at {db_path}"
    return db_path, None


def cal_files_from_database(db_path):
    """Parse a pac database for the raw calibrator archive filenames.

    Returns a list of (filepath, type) tuples.
    """
    db_dir = Path(db_path).resolve().parent
    out = []
    for line in open(db_path):
        parts = line.split()
        if len(parts) >= 2 and not line.startswith('#') and \
           not parts[0].startswith('Pulsar::'):
            fpath = Path(parts[0])
            if not fpath.is_absolute():
                fpath = db_dir / fpath
            out.append((fpath, parts[1]))
    return out


def auto_scrunch_cal(native_db_path, fold_nchan, cal_dir,
                     fold_bw_mhz=0.0, fold_center_mhz=0.0,
                     pam_bin='pam', pac_bin='pac', pac_flags=''):
    """Build a frequency-scrunched copy of calibration material for pac.

    Extracts the sub-band matching the fold's centre/bandwidth from the native
    PolnCal archive, then scrunches to fold_nchan so pac's channel matching
    succeeds.

    Returns (new_db_path, note); (None, note) on failure.
    """
    try:
        import psrchive
    except ImportError as e:
        return None, (f"psrchive python bindings not available ({e}); "
                      f"can't auto-scrunch calibrator")

    entries = cal_files_from_database(native_db_path)
    polncal_files = [f for f, t in entries if t == 'PolnCal']
    if not polncal_files:
        return None, (f"no PolnCal entries found in {native_db_path}, "
                      f"can't auto-scrunch")

    target_dir = Path(cal_dir) / f"scrunched_{fold_nchan}ch"
    target_dir.mkdir(parents=True, exist_ok=True)

    use_subband = fold_bw_mhz > 0 and fold_center_mhz > 0

    for f, ftype in entries:
        f = Path(f)
        if not f.exists():
            continue
        if ftype == 'PolnCal':
            out_name = f.name.split('.')[0] + '.dzT'
            out_f = target_dir / out_name
            if out_f.exists():
                try:
                    _chk = psrchive.Archive.load(str(out_f))
                    _chk_nchan = _chk.get_nchan()
                    _chk_bw = _chk.get_bandwidth()
                    _stale = (_chk_nchan != fold_nchan
                              or (_chk_nchan != 0 and _chk_bw < 0))
                    if _stale:
                        print(f"  removing stale {out_f.name} "
                              f"(nchan={_chk_nchan}, bw={_chk_bw:.1f})"
                              f" != expected nchan={fold_nchan}, bw>0)")
                        out_f.unlink()
                except Exception:
                    out_f.unlink(missing_ok=True)
                if out_f.exists():
                    continue
            if use_subband:
                a = psrchive.Archive.load(str(f))
                native_nchan = a.get_nchan()
                native_freqs = a.get_frequencies()
                lo_mhz = fold_center_mhz - fold_bw_mhz / 2.0
                hi_mhz = fold_center_mhz + fold_bw_mhz / 2.0
                in_band = [i for i, freq in enumerate(native_freqs)
                           if lo_mhz <= freq <= hi_mhz]
                if not in_band:
                    return None, (f"no channels of {f.name} "
                                 f"(nchan={native_nchan}, "
                                 f"{native_freqs[0]:.1f}-"
                                 f"{native_freqs[-1]:.1f} MHz) fall in "
                                 f"fold sub-band [{lo_mhz:.1f}, "
                                 f"{hi_mhz:.1f}] MHz")
                sub_nchan = len(in_band)
                lo_idx, hi_idx = in_band[0], in_band[-1]
                if hi_idx < native_nchan - 1:
                    a.remove_chan(hi_idx + 1, native_nchan - 1)
                if lo_idx > 0:
                    a.remove_chan(0, lo_idx - 1)
                out_path = str(target_dir / out_name)
                a.unload(out_path)
                if sub_nchan != fold_nchan:
                    if sub_nchan % fold_nchan != 0:
                        return None, (
                            f"sub-band of {f.name} has {sub_nchan} ch "
                            f"which is not an integer multiple of "
                            f"fold_nchan={fold_nchan}")
                    factor = sub_nchan // fold_nchan
                    cmd = [pam_bin, '-f', str(factor), '-e', 'dzT',
                           '-u', str(target_dir), out_path]
                    print(f"\nSub-band extracted {f.name}: "
                          f"{native_nchan} -> {sub_nchan} ch, "
                          f"centre={a.get_centre_frequency():.1f} MHz")
                    print(f"  then pam -f {factor} scrunch to "
                          f"{fold_nchan} ch: {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True)
                    pam_out = str(target_dir / out_name)
                    if not Path(pam_out).exists():
                        return None, (
                            f"pam -f failed to produce {pam_out}")
                    _chk = psrchive.Archive.load(pam_out)
                    expected_bw = fold_bw_mhz / fold_nchan
                    chk_bw = _chk.get_bandwidth()
                    actual_bw = abs(chk_bw) / _chk.get_nchan()
                    if abs(actual_bw - expected_bw) > 0.01:
                        return None, (
                            f"pam -f produced {pam_out} with "
                            f"chan_bw={actual_bw:.3f} MHz, expected "
                            f"{expected_bw:.3f} MHz "
                            f"(nchan={_chk.get_nchan()}, "
                            f"bw={abs(chk_bw):.1f} MHz)")
                    if chk_bw < 0:
                        rev_cmd = [pam_bin, '--reverse_freqs', '-m',
                                   pam_out]
                        subprocess.run(rev_cmd, check=True,
                                       capture_output=True)
                        _chk2 = psrchive.Archive.load(pam_out)
                        print(f"  reversed freq axis ({chk_bw:.1f} -> "
                              f"{_chk2.get_bandwidth():.1f} MHz)")
                    print(f"  verified: {pam_out} nchan="
                          f"{_chk.get_nchan()}, chan_bw="
                          f"{actual_bw:.3f} MHz")
                else:
                    if a.get_bandwidth() < 0:
                        rev_cmd = [pam_bin, '--reverse_freqs', '-m',
                                   out_path]
                        subprocess.run(rev_cmd, check=True,
                                       capture_output=True)
                        a = psrchive.Archive.load(out_path)
                        print(f"\nSub-band extracted {f.name}: "
                              f"{native_nchan} -> {sub_nchan} ch "
                              f"(already {fold_nchan} ch), "
                              f"centre={a.get_centre_frequency():.1f} MHz"
                              f", reversed freq axis -> {out_path}")
                    else:
                        print(f"\nSub-band extracted {f.name}: "
                              f"{native_nchan} -> {sub_nchan} ch "
                              f"(already {fold_nchan} ch), "
                              f"centre={a.get_centre_frequency():.1f} MHz"
                              f" -> {out_path}")
            else:
                a = psrchive.Archive.load(str(f))
                native_nchan = a.get_nchan()
                if native_nchan % fold_nchan != 0:
                    return None, (f"{f.name} has {native_nchan} channels, "
                                  f"can't scrunch to {fold_nchan}")
                factor = native_nchan // fold_nchan
                cmd = [pam_bin, '-f', str(factor), '-e', 'dzT',
                       '-u', str(target_dir), str(f)]
                print(f"\nScrunching calibrator {f.name} by {factor}x "
                      f"({native_nchan} -> {fold_nchan} channels)")
                print(" ".join(cmd))
                subprocess.run(cmd, capture_output=True, text=True)

    for stale in target_dir.glob('*avfluxcal*'):
        if stale.is_symlink():
            stale.unlink()
    for stale in target_dir.glob('*.pcm'):
        if stale.is_symlink():
            stale.unlink()

    new_db_path = target_dir / 'database.txt'
    if not new_db_path.exists():
        new_db_path, note = build_cal_database(target_dir, new_db_path,
                                               pac_bin, pac_flags,
                                               use_exts=['dzT'])
        if new_db_path is None:
            return None, f"scrunched files written but database build failed: {note}"
    return new_db_path, f"auto-scrunched calibration to {fold_nchan} channels -> {new_db_path}"


def resolve_calibration(args, cal_dir, fold_bw_mhz=0.0,
                        fold_center_mhz=0.0):
    """Decide which pac calibration to apply, building the database if needed.

    Precedence: explicit --calib > explicit --calib-db > existing database
    > auto-generate.

    Returns (calib_file, cal_db, note); both None means no calibration.
    """
    if args.calib:
        return args.calib, None, f"using explicit calibrator model {args.calib}"
    db_path = (Path(args.calib_db) if args.calib_db
               else Path(cal_dir) / args.cal_db_name)
    if db_path.exists():
        note = f"using existing calibration database {db_path}"
        scrunched_db, scrunch_note = auto_scrunch_cal(
            db_path, args.fold_nchan, cal_dir,
            fold_bw_mhz=fold_bw_mhz, fold_center_mhz=fold_center_mhz,
            pam_bin=args.pam_bin, pac_bin=args.pac_bin,
            pac_flags=args.pac_flags)
        if scrunched_db is None:
            return (None, None,
                    f"WARNING: {scrunch_note} — saving UNCALIBRATED folds")
        return None, scrunched_db, scrunch_note
    if args.calib_db:
        return (None, None,
                f"WARNING: --calib-db {db_path} not found — saving UNCALIBRATED folds")
    db_path, note = build_cal_database(cal_dir, db_path, args.pac_bin, args.pac_flags)
    if db_path is None:
        return None, None, f"WARNING: {note} — saving UNCALIBRATED folds"
    scrunched_db, scrunch_note = auto_scrunch_cal(
        db_path, args.fold_nchan, cal_dir,
        fold_bw_mhz=fold_bw_mhz, fold_center_mhz=fold_center_mhz,
        pam_bin=args.pam_bin, pac_bin=args.pac_bin,
        pac_flags=args.pac_flags)
    if scrunched_db is None:
        return (None, None,
                f"WARNING: {scrunch_note} — saving UNCALIBRATED folds")
    return None, scrunched_db, scrunch_note
