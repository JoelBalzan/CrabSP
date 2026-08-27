"""Fragment indexing: scan workdir for .dada/.fil pairs and locate candidates."""
from pathlib import Path

from .headers import parse_fil_header


def build_fragment_index(workdir):
    """Scan workdir for *.dada.fil files, build a chronological fragment index.

    Returns (frags, stream_root_mjd) where stream_root_mjd is the MJD of the
    first fragment's tstart (used by transientX continuous-search timestamps).
    """
    frags = []
    for fil in sorted(Path(workdir).glob('*.dada.fil')):
        dada_path = Path(str(fil)[:-4])
        if not dada_path.exists():
            print(f"  WARNING: no raw .dada for {fil.name}, skipping")
            continue
        h = parse_fil_header(fil)
        t_end = h['tstart_mjd'] + h['obslen_s'] / 86400.0
        frags.append({'dada_path': dada_path, 'fil_path': fil, **h, 't_end_mjd': t_end})
    frags.sort(key=lambda f: f['tstart_mjd'])
    if not frags:
        return frags, None
    cum = 0.0
    for f in frags:
        f['stream_start_s'] = cum
        cum += f['obslen_s']
    return frags, frags[0]['tstart_mjd']


def find_fragment(frags, stream_root, mjd, tol_s=0.01):
    """Locate the searched fragment containing a candidate.

    Primary: continuous-stream matching. Fallback: absolute-MJD matching.
    Returns (frag, offset_within_frag_s) or (None, None).
    """
    if stream_root is not None:
        global_s = (mjd - stream_root) * 86400.0
        if global_s >= 0:
            for f in frags:
                if f['stream_start_s'] <= global_s < f['stream_start_s'] + f['obslen_s']:
                    return f, global_s - f['stream_start_s']
    tol_days = tol_s / 86400.0
    for f in frags:
        if f['tstart_mjd'] - tol_days <= mjd < f['t_end_mjd'] + tol_days:
            return f, (mjd - f['tstart_mjd']) * 86400.0
    return None, None
