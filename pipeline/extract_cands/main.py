"""CLI entry point for extract_cands."""
import argparse
from pathlib import Path

import numpy as np

from .headers import parse_dada_header
from .fragments import build_fragment_index, find_fragment
from .candidates import parse_cand_line, cluster_candidates, pick_representative
from .headers import parse_fil_header
from .digifil_route import (plan_extraction, trim_to_window, extract_cutout,
							read_fil_cube, coherency_to_stokes)
from .folding import plan_dspsr_fold, fold_cutout
from .archive import (fix_dspsr_receiver, fscrunch_to_nchan, log_archive_info,
					  fix_archive_frequencies, read_ar_stokes, trim_folded_to_window)
from .calibration import resolve_calibration, apply_pac
from .utils import get_tx_resolution, tx_res_us
from .headers import crop_dada_file


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cand-dir', default=None,
				help='recursively search this directory for *.cands files')
	ap.add_argument('--cand-files', nargs='+')
	ap.add_argument('--workdir', default='.')
	ap.add_argument('--outdir', default='cutouts')
	ap.add_argument('--window-s', type=float, default=0.006,
					 help='fixed cutout window length, seconds (default 6 ms), '
						  'centred on the burst.')
	ap.add_argument('--min-snr', type=float, default=0.0)
	ap.add_argument('--cluster-gap-ms', type=float, default=0.0,
					 help='merge candidates into one event when consecutive MJDs are '
						  'less than this far apart (ms). 0 = no clustering (default)')
	ap.add_argument('--digifil-bin', default='digifil')
	ap.add_argument('--digifil-min-block', '--df-min-block', type=float, default=0.5,
					 help='minimum duration (s) to request from digifil per call.')
	ap.add_argument('--keep-fil', action='store_true',
					 help='keep the intermediate .fil cutout')
	ap.add_argument('--plot', action='store_true',
					 help='save a polarimetric diagnostic PNG per candidate')

	ap.add_argument('--method', choices=['digifil', 'dspsr'], default='dspsr',
					 help='extraction backend: dspsr (default) or digifil')
	ap.add_argument('--dspsr-bin', default='dspsr')
	ap.add_argument('--pac-bin', default='pac')
	ap.add_argument('--pam-bin', default='pam')
	ap.add_argument('--psredit-bin', default='psredit')
	ap.add_argument('--calib', default=None, metavar='FILE',
					 help='pac calibration model (pac -A)')
	ap.add_argument('--calib-db', default=None, metavar='FILE',
					 help='pac calibration database (pac -d)')
	ap.add_argument('--cal-dir', default=None, metavar='DIR',
					 help='directory holding calibration material. Default: --workdir.')
	ap.add_argument('--cal-db-name', default='database.txt',
					 help='name of auto-detected/generated calibration database')
	ap.add_argument('--no-cal', action='store_true',
					 help='disable automatic pac calibration')
	ap.add_argument('--pac-flags', default='-F -b -T -a',
					 help='extra flags for the pac steps')
	ap.add_argument('--fold-cf-offset', type=float, default=0.0,
					 help='shift (MHz) of the fold centre frequency for pac alignment')
	ap.add_argument('--fold-nbin', type=int, default=None,
					 help='phase bins per period. Default: auto-computed.')
	ap.add_argument('-F', '--nchan', type=int, default=8,
					 help='number of channels for both digifil (-F) and '
						  'dspsr (-F) extraction')
	ap.add_argument('--fold-period', type=float, default=0.0334,
					 help='fold period (s)')
	ap.add_argument('--fold-parfile', default=None, metavar='FILE',
					 help='TEMPO2-style ephemeris for phase-coherent folding')
	ap.add_argument('--fold-margin-s', type=float, default=None,
					 help='seek margin before candidate MJD (default: window_s/2)')
	ap.add_argument('--fold-turns', type=int, default=1,
					 help='number of spin periods to fold per candidate')
	ap.add_argument('--keep-ar', action='store_true',
					 help='keep the intermediate .ar/.calib archives')
	ap.add_argument('--fast-pac', dest='fast_pac', action='store_true', default=True,
					 help='crop input .dada before folding for speed (default: on)')
	ap.add_argument('--no-fast-pac', dest='fast_pac', action='store_false',
					 help='disable fast-pac cropping')
	ap.add_argument('--fast-min-crop-s', type=float, default=0.35,
					 help='minimum crop duration (s) for --fast-pac')
	ap.add_argument('--mode', default='crab',
					 help='pipeline mode (crab=fast-pac + dspsr, kept for compat; default crab)')
	ap.add_argument('--rm', type=float, default=None,
					 help='rotation measure for coherent Faraday correction')
	ap.add_argument('--dm', type=float, default=None,
					 help='override DM (pc/cm3) for all candidates')

	args = ap.parse_args()
	use_dspsr = args.method == 'dspsr' or bool(args.calib or args.calib_db)
	if use_dspsr:
		print("Using dspsr folding route")
	if args.fold_turns > 1 and not args.fold_parfile:
		print("WARNING: --fold-turns > 1 without --fold-parfile — phase will "
			  "drift across turns. Strongly recommend --fold-parfile.")
	cal_dir = Path(args.cal_dir) if args.cal_dir else Path(args.workdir)

	fold_bw_mhz = 0.0
	fold_center_mhz = 0.0
	if use_dspsr:
		_dada_files = sorted(Path(args.workdir).glob('*.dada'))
		if _dada_files:
			_hdr = parse_dada_header(_dada_files[0])
			fold_bw_mhz = abs(float(_hdr.get('BW', 0)))
			fold_center_mhz = float(_hdr.get('FREQ', 0))

	calib_file = None
	cal_db = None
	if use_dspsr and (not args.no_cal or args.calib or args.calib_db):
		calib_file, cal_db, cal_note = resolve_calibration(
			args, cal_dir, fold_bw_mhz=fold_bw_mhz,
			fold_center_mhz=fold_center_mhz)
		print(f"  calibration : {cal_note}")
	else:
		print("  calibration : disabled (--no-cal)" if args.no_cal
			  else "  calibration : none requested")

	print(f"Indexing fragments in {args.workdir} ...")
	frags, stream_root = build_fragment_index(args.workdir)
	print(f"  found {len(frags)} fragments, "
		  f"MJD {frags[0]['tstart_mjd']:.9f} -> {frags[-1]['t_end_mjd']:.9f} "
		  f"(continuous-search root MJD {stream_root:.9f})")

	if use_dspsr and args.fold_nbin is None:
		dada_hdr = parse_dada_header(frags[0]['dada_path'])
		bw_mhz = abs(float(dada_hdr.get('BW', 0.0)))
		if bw_mhz > 0:
			chan_bw_hz = bw_mhz * 1e6 / args.nchan
			args.fold_nbin = int(np.ceil(args.fold_period * chan_bw_hz))
			print(f"  auto --fold-nbin: {bw_mhz:.0f} MHz / {args.nchan} "
				  f"ch = {chan_bw_hz/1e6:.1f} MHz/ch -> "
				  f"nbin = {args.fold_nbin} "
				  f"({args.fold_period / args.fold_nbin * 1e6:.2f} us/bin)")
		else:
			args.fold_nbin = 4096
			print("  WARNING: could not read BW from DADA header, "
				  "falling back to --fold-nbin 4096")

	base_outdir = Path(args.outdir)
	base_outdir.mkdir(parents=True, exist_ok=True)

	if args.cand_dir:
		cand_files = sorted(Path(args.cand_dir).rglob('*.cands'), key=tx_res_us)
	else:
		cand_files = [Path(x) for x in args.cand_files]

	for cand_file in cand_files:

		tx_res = get_tx_resolution(cand_file)

		# avoid cands/cands or cutouts/cands for merged files (unique.cands)
		if tx_res in (base_outdir.name, "cands"):
			outdir = base_outdir
		else:
			outdir = base_outdir / tx_res
		outdir.mkdir(parents=True, exist_ok=True)

		print(f"TX resolution: {tx_res}")

		cand_file = Path(cand_file)
		print(f'\n=== {cand_file} ===')
		cands = []
		for line in open(cand_file):
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			c = parse_cand_line(line)
			if c['snr'] is not None and c['snr'] < args.min_snr:
				continue
			cands.append(c)

		if args.cluster_gap_ms > 0:
			events = cluster_candidates(cands, args.cluster_gap_ms / 1000.0)
		else:
			events = [[c] for c in cands]
		n_events = len(events)
		n_cands = len(cands)
		print(f"  {n_cands} candidates -> {n_events} events "
			  f"(cluster gap {args.cluster_gap_ms:g} ms; "
			  f"{n_cands / max(n_events, 1):.1f} detections/event)")
		for i_event, event in enumerate(events):
			c = pick_representative(event)
			if args.dm is not None:
				c['dm'] = args.dm

			frag, offset_s = find_fragment(frags, stream_root, c['mjd'])

			if frag is None:
				print(f"  cand {c['cand_id']} mjd={c['mjd']:.9f}: NO fragment contains this MJD")
				continue

			if use_dspsr:
				_process_dspsr(c, event, i_event, n_events, frag, offset_s,
							   frags, stream_root, args, calib_file, cal_db,
							   base_outdir, outdir, fold_bw_mhz)
				continue

			_process_digifil(c, event, i_event, n_events, frag, offset_s,
							 frags, args, base_outdir, outdir)


def _process_dspsr(c, event, i_event, n_events, frag, offset_s,
				   frags, stream_root, args, calib_file, cal_db,
				   base_outdir, outdir, fold_bw_mhz):
	"""Handle the dspsr folding route for a single candidate event."""
	window_s = args.window_s
	period = args.fold_period
	margin_s = (args.fold_margin_s if args.fold_margin_s is not None
				else window_s / 2.0)

	print("\n==========================")
	print(f"Event {i_event+1}/{n_events} "
		  f"({len(event)} detections, cand {c['cand_id']})")
	print(f"Candidate MJD        : {c['mjd']:.12f}")
	print(f"Candidate DM         : {c['dm']}")
	print(f"Candidate Width (ms) : {c['width_ms']}")
	print(f"Candidate SNR        : {c['snr']}")

	print(f"\nFragment")
	print(f"  file              : {frag['dada_path'].name}")
	print(f"  start MJD         : {frag['tstart_mjd']:.12f}")
	print(f"  end MJD           : {frag['t_end_mjd']:.12f}")

	dada_paths, seek_mjd, turns, plan_note = plan_dspsr_fold(
		frags, c['mjd'], margin_s, period, args.fold_turns)
	if dada_paths is None:
		print(f"    SKIP cand {c['cand_id']} mjd={c['mjd']:.9f}: {plan_note}")
		return

	print(f"\nComputed")
	print(f"  seek MJD (fold start) : {seek_mjd:.12f}")
	print(f"  margin_s              : {margin_s:.6f}")
	print(f"  fold period (s)       : {period:.6f}"
		  f"{'  (ignored: using --fold-parfile)' if args.fold_parfile else ''}")
	print(f"  fold parfile          : {args.fold_parfile or '(none, using -c/-cepoch)'}")
	print(f"  fold turns            : {turns}")
	print(f"  fold bins             : {args.fold_nbin}")
	print(f"  fold nchan            : {args.nchan}")
	print(f"  fold files            : {[p.name for p in dada_paths]}")

	outname = f"cand{c['cand_id']}_{c['mjd']:.9f}_dm{c['dm']:.2f}"

	if args.fast_pac:
		print(f"  mode                  : fast (cropped input, "
			  f"real period"
			  f"{', -derotate -rm ' + str(args.rm) if args.rm is not None else ''})")
		fold_dada_paths = dada_paths
		tmp_cropped = None
		if len(dada_paths) == 1:
			# Coherent dedispersion needs history before the fold epoch.
			# dspsr's overlap-save filter is ~131 ms at 32 Msps (nfft=4194304)
			# plus ~tens of ms of dispersive sweep at low frequency, so
			# starting the cropped file exactly at seek_mjd leaves the first
			# ~100 ms of the fold corrupted / truncated (subint 0 short, burst
			# appears in the next subint and is missed when we keep subint 0).
			# Include a pre-roll before seek; clamp to the fragment start.
			_prepad_s = 0.25  # enough for 131 ms filter + DM sweep anywhere in UWL
			seek_offset_s = (seek_mjd - frag['tstart_mjd']) * 86400.0
			desired_start_s = seek_offset_s - _prepad_s
			if desired_start_s < 0.0:
				_prepad_s = seek_offset_s  # use whatever history is available
				desired_start_s = 0.0
			crop_offset_s = desired_start_s
			crop_dur_s = max(3.0 * (turns * period + 2 * margin_s),
							 args.fast_min_crop_s) + _prepad_s
			crop_dur_s = min(crop_dur_s,
							 frag['obslen_s'] - crop_offset_s)
			try:
				tmp_cropped = crop_dada_file(
					dada_paths[0], crop_offset_s, crop_dur_s,
					outdir / f".{outname}_crop.dada",
					true_obslen_s=frag['obslen_s'])
				fold_dada_paths = [tmp_cropped]
				print(f"  cropped input         : "
					  f"{crop_dur_s*1000:.1f} ms from "
					  f"{dada_paths[0].name} -> {tmp_cropped.name}")
			except RuntimeError as e:
				print(f"  WARNING: crop_dada_file failed ({e}); "
					  f"falling back to the full fragment (will be slow)")
		else:
			print("  WARNING: candidate spans multiple fragments; "
				  "skipping the input-crop optimisation for this one (will be slow)")

		ar_path = fold_cutout(
			fold_dada_paths, seek_mjd, c['dm'], period,
			args.fold_nbin, args.nchan, turns,
			outname, outdir, dspsr_bin=args.dspsr_bin,
			cf_offset_mhz=(
				args.fold_cf_offset
				if (calib_file or cal_db) else 0.0),
			parfile=args.fold_parfile,
			rm=args.rm)
		if tmp_cropped is not None:
			tmp_cropped.unlink(missing_ok=True)
		if ar_path is None:
			return

		log_archive_info(ar_path, "dspsr output")
		cal_path = ar_path
		if calib_file or cal_db:
			cal_path = _calibrate_archive(ar_path, calib_file, cal_db, args,
										  fold_bw_mhz)
			if cal_path is None:
				if not args.keep_ar:
					ar_path.unlink(missing_ok=True)
				return
		else:
			print("  (no calibration available: saving the "
				  "UNCALIBRATED fast fold)")
	else:
		ar_path = fold_cutout(dada_paths, seek_mjd, c['dm'], period,
							  args.fold_nbin, args.nchan, turns,
							  outname, outdir, dspsr_bin=args.dspsr_bin,
							  cf_offset_mhz=(
								  args.fold_cf_offset
								  if (calib_file or cal_db) else 0.0),
							  parfile=args.fold_parfile,
							  rm=args.rm)
		if ar_path is None:
			return

		log_archive_info(ar_path, "dspsr output")
		cal_path = ar_path
		if calib_file or cal_db:
			cal_path = _calibrate_archive(ar_path, calib_file, cal_db, args,
										  fold_bw_mhz)
			if cal_path is None:
				if not args.keep_ar:
					ar_path.unlink(missing_ok=True)
				return
		else:
			print("  (no calibration available: saving the UNCALIBRATED fold)")

	try:
		stokes, meta = read_ar_stokes(cal_path, cand_mjd=c['mjd'], seek_mjd=seek_mjd)
	except Exception as e:
		print(f"    FAILED to read {cal_path}: {e}")
		if not args.keep_ar:
			ar_path.unlink(missing_ok=True)
			if cal_path != ar_path:
				cal_path.unlink(missing_ok=True)
		return

	folded_period_s = period
	tsamp_s = folded_period_s / meta['nbin']
	print(f"\nFolded archive")
	print(f"  state             : {meta['state']}")
	print(f"  shape             : nsub=1, npol=4, "
		  f"nchan={meta['nchan']}, nbin={meta['nbin']}")
	print(f"  tsamp (bin dur)   : {tsamp_s*1e6:.3f} us")
	print(f"  freqs (MHz)       : {meta['fch1_mhz']:.3f} .. "
		  f"{meta['fch1_mhz'] + meta['foff_mhz']*(meta['nchan']-1):.3f} "
		  f"(df={meta['foff_mhz']:.3f})")
	epoch_str = (f"{meta['epoch_mjd']:.12f}" if meta['epoch_mjd']
				 else "n/a (using fold seek)")
	print(f"  archive epoch MJD : {epoch_str}")

	stokes, trimmed_tstart_mjd = trim_folded_to_window(
		stokes, meta['epoch_mjd'] or seek_mjd, tsamp_s, window_s)

	#labels = ["I", "Q", "U", "V"]
	#print("\nTrimmed Stokes block")
	#for i, label in enumerate(labels):
	#    arr = stokes[i]
	#    print(f"{label}")
	#    print(f"   mean = {arr.mean():.6f}")
	#    print(f"   std  = {arr.std():.6f}")
	#    print(f"   min  = {arr.min():.6f}")
	#    print(f"   max  = {arr.max():.6f}")
	profile = stokes[0].mean(axis=0)
	peak = int(np.argmax(profile))
	print(f"Trimmed start MJD  : {trimmed_tstart_mjd:.12f}")
	print(f"Trimmed duration   : {stokes.shape[-1]*tsamp_s*1000:.3f} ms")
	print(f"Peak sample        : {peak}  ({peak*tsamp_s*1000:.3f} ms)")

	neg_I_frac = float((stokes[0] < 0).mean())
	print(f"    [debug] fraction of trimmed Stokes-I samples < 0: "
		  f"{neg_I_frac:.3f}")

	npz_path = outdir / f"{outname}_iquv.npz"
	np.savez(
		npz_path,
		stokes=stokes.astype(np.float32),
		pol_order=np.array(['I', 'Q', 'U', 'V']),
		cand_id=c['cand_id'],
		cand_mjd=c['mjd'],
		cand_dm=c['dm'],
		cand_width_ms=c['width_ms'],
		cand_snr=c['snr'],
		source_dada=str(frag['dada_path']),
		tstart_mjd=trimmed_tstart_mjd,
		tsamp_s=tsamp_s,
		nsamp=stokes.shape[-1],
		fch1_mhz=meta['fch1_mhz'],
		foff_mhz=meta['foff_mhz'],
		nchan=meta['nchan'],
		window_s=window_s,
		method='dspsr-fast-cropped' if args.fast_pac else 'dspsr',
		calib_applied=bool(calib_file or cal_db),
		calib_file=str(calib_file or cal_db or ''),
		pac_flags=args.pac_flags,
		fold_seek_mjd=seek_mjd,
		fold_period=folded_period_s,
		fold_parfile=str(args.fold_parfile or ''),
		fold_turns=turns,
		fold_nbin=args.fold_nbin,
		fold_nchan=args.nchan,
		rm=(args.rm if args.rm is not None else np.nan),
	)
	print(f"    saved -> {npz_path}  shape={stokes.shape} (pol,chan,samp)")

	if args.plot:
		_generate_plot(npz_path, outname, outdir,
					   calib_file, cal_db, args)

	if not args.keep_ar:
		ar_path.unlink(missing_ok=True)
		if cal_path != ar_path:
			cal_path.unlink(missing_ok=True)


def _calibrate_archive(ar_path, calib_file, cal_db, args, fold_bw_mhz):
	"""Apply calibration to a dspsr archive. Returns calibrated path or None."""
	fix_dspsr_receiver(ar_path, psredit_bin=args.psredit_bin)
	_exp_cbw = (fold_bw_mhz / args.nchan
				if fold_bw_mhz > 0 else 0.0)
	fscrunch_to_nchan(ar_path, args.nchan,
					  expected_chan_bw=_exp_cbw,
					  pam_bin=args.pam_bin,
					  psredit_bin=args.psredit_bin)
	log_archive_info(ar_path, "after fscrunch")
	if cal_db:
		_cal_dzts = list(Path(cal_db).parent.glob('*.dzT'))
		if _cal_dzts:
			fix_archive_frequencies(ar_path, _cal_dzts[0], args.nchan)
	cal_path = apply_pac(ar_path, calib=calib_file,
						 calib_db=cal_db,
						 pac_bin=args.pac_bin,
						 pac_flags=args.pac_flags)
	return cal_path


def _process_digifil(c, event, i_event, n_events, frag, offset_s,
					 frags, args, base_outdir, outdir):
	"""Handle the digifil extraction route for a single candidate event."""
	from plotting.plot_iquv_profile import generate_profile_plot

	window_s = args.window_s
	dada_paths, digifil_seek_s, digifil_dur_s, first_frag, plan_note = plan_extraction(
		frags, frag, offset_s, min_block_s=args.digifil_min_block)
	if dada_paths is None:
		print(f"    SKIP cand {c['cand_id']} mjd={c['mjd']:.9f}: {plan_note}")
		return

	print("\n==========================")
	print(f"Event {i_event+1}/{n_events} "
		  f"({len(event)} detections, cand {c['cand_id']})")
	print(f"Candidate MJD        : {c['mjd']:.12f}")
	print(f"Candidate DM         : {c['dm']}")
	print(f"Candidate Width (ms) : {c['width_ms']}")
	print(f"Candidate SNR        : {c['snr']}")

	print(f"\nFragment")
	print(f"  file              : {frag['dada_path'].name}")
	print(f"  start MJD         : {frag['tstart_mjd']:.12f}")
	print(f"  end MJD           : {frag['t_end_mjd']:.12f}")
	print(f"  duration (s)      : {frag['obslen_s']:.6f}")

	print(f"\nComputed")
	print(f"  offset_s (local)  : {offset_s:.6f}")
	print(f"  window_s          : {window_s:.6f}")
	print(f"  digifil_seek_s    : {digifil_seek_s:.6f}")
	print(f"  digifil_dur_s     : {digifil_dur_s:.6f}")
	print(f"  digifil files     : {[p.name for p in dada_paths]}")

	outname = f"cand{c['cand_id']}_{c['mjd']:.9f}_dm{c['dm']:.2f}"
	print(f"  cand {c['cand_id']}: mjd={c['mjd']:.9f} width={c['width_ms']}ms "
		  f"dm={c['dm']:.2f} -> {frag['dada_path'].name} "
		  f"final_window={window_s*1000:.2f}ms "
		  f"digifil_seek={digifil_seek_s:.4f}s digifil_dur={digifil_dur_s:.4f}s snr={c['snr']}")

	fil_cutout = extract_cutout(dada_paths, digifil_seek_s, digifil_dur_s,
								 c['dm'], outname, outdir,
								 digifil_bin=args.digifil_bin,
								 fft=args.nchan)
	if fil_cutout is None:
		return

	try:
		cutout_hdr = parse_fil_header(fil_cutout)
		expected_start = first_frag['tstart_mjd'] + digifil_seek_s/86400.0

		print("\nCutout header")
		print(f"Header start MJD     : {cutout_hdr['tstart_mjd']:.12f}")
		print(f"Expected start MJD   : {expected_start:.12f}")
		print(f"Difference (ms)      : {(cutout_hdr['tstart_mjd']-expected_start)*86400*1000:.6f}")
		print(f"Header tsamp         : {cutout_hdr['tsamp_s']}")
		print(f"Header nsamp         : {cutout_hdr['nsamp']}")
		print(f"Header nifs          : {cutout_hdr['nifs']}")
		print(f"Header nchan         : {cutout_hdr['nchan']}")

		cube = read_fil_cube(fil_cutout)
		print("\nCube")
		print("cube shape:", cube.shape)

		stokes_block = coherency_to_stokes(cube)

		I = stokes_block[0]
		profile = I.mean(axis=0)
		peak = np.argmax(profile)

		labels = ["I", "Q", "U", "V"]
		print("\nStokes block")
		for i, label in enumerate(labels):
			arr = stokes_block[i]
			print(f"{label}")
			print(f"   mean = {arr.mean():.6f}")
			print(f"   std  = {arr.std():.6f}")
			print(f"   min  = {arr.min():.6f}")
			print(f"   max  = {arr.max():.6f}")
		stokes, trimmed_tstart_mjd = trim_to_window(
			stokes_block, cutout_hdr['tstart_mjd'], cutout_hdr['tsamp_s'],
			c['mjd'], window_s)

		print("\nTrimmed cube")
		print("shape:", stokes.shape)

		I = stokes[0]
		profile = I.mean(axis=0)
		peak = np.argmax(profile)

		print("\nTrimmed profile")
		print(f"Peak sample      : {peak}")
		print(f"Peak time (ms)   : {peak*cutout_hdr['tsamp_s']*1000:.3f}")
		print(f"Peak value       : {profile[peak]:.6f}")

		print(f"Trimmed start MJD : {trimmed_tstart_mjd:.12f}")

		new_offset = (c['mjd'] - trimmed_tstart_mjd)*86400.0

		print(f"Candidate offset after trimming : {new_offset*1000:.3f} ms")
		print(f"Trimmed duration               : {stokes.shape[-1]*cutout_hdr['tsamp_s']*1000:.3f} ms")
	except Exception as e:
		print(f"    FAILED to read/convert {fil_cutout}: {e}")
		return

	neg_I_frac = float((stokes[0] < 0).mean())
	print(f"    [debug] fraction of trimmed Stokes-I samples < 0: "
		  f"{neg_I_frac:.3f} (expect near 0.0; if not, pol-axis reshape "
		  f"likely still wrong)")

	npz_path = outdir / f"{outname}_iquv.npz"
	np.savez(
		npz_path,
		stokes=stokes.astype(np.float32),
		pol_order=np.array(['I', 'Q', 'U', 'V']),
		cand_id=c['cand_id'],
		cand_mjd=c['mjd'],
		cand_dm=c['dm'],
		cand_width_ms=c['width_ms'],
		cand_snr=c['snr'],
		source_dada=str(frag['dada_path']),
		tstart_mjd=trimmed_tstart_mjd,
		tsamp_s=cutout_hdr['tsamp_s'],
		nsamp=stokes.shape[-1],
		fch1_mhz=cutout_hdr['f1_mhz'],
		foff_mhz=cutout_hdr['bw_mhz'],
		nchan=cutout_hdr['nchan'],
		window_s=window_s,
		digifil_seek_s=digifil_seek_s,
		digifil_dur_s=digifil_dur_s,
	)
	print(f"    saved -> {npz_path}  shape={stokes.shape} (pol,chan,samp)")

	if args.plot:
		png_path = outdir / f"{outname}_profile.png"
		try:
			generate_profile_plot(npz_path, out=str(png_path),
								   title_suffix="[digifil, uncalibrated]")
			print(f"    profile plot -> {png_path}")
		except Exception as e:
			print(f"    plot FAILED: {e}")

	if not args.keep_fil:
		fil_cutout.unlink(missing_ok=True)


def _generate_plot(npz_path, outname, outdir, calib_file, cal_db, args):
	"""Generate polarimetric profile plot."""
	from plotting.plot_iquv_profile import generate_profile_plot
	import os

	# PNGs requested in cands/<mode>/ (not cutouts) per user — cd there and run
	cands_outdir = Path(str(outdir).replace("cutouts", "cands", 1)) if "cutouts" in str(outdir) else outdir
	cands_outdir.mkdir(parents=True, exist_ok=True)
	png_path = cands_outdir / f"{outname}_iquv_profile.png"
	cal_applied_now = bool(calib_file or cal_db)
	cal_tag = 'calibrated' if cal_applied_now else 'UNCALIBRATED'
	if args.fast_pac:
		cal_tag += ', fast'
	npz_abs = Path(npz_path).resolve()
	png_abs = Path(png_path).resolve()
	orig = os.getcwd()
	try:
		os.chdir(cands_outdir.resolve())
		generate_profile_plot(str(npz_abs), out=str(png_abs),
							   title_suffix=f"[{cal_tag}]")
		print(f"    profile plot -> {png_path}")
	except Exception as e:
		print(f"    plot FAILED: {e}")
	finally:
		try:
			os.chdir(orig)
		except Exception:
			pass
