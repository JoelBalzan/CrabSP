"""Candidate parsing and event clustering."""


def parse_cand_line(line):
    """Parse a whitespace-separated candidate line into a dict."""
    p = line.split()
    return {
        'beam': p[0],
        'cand_id': p[1],
        'mjd': float(p[2]),
        'dm': float(p[3]),
        'width_ms': float(p[4]) if len(p) > 4 else None,
        'snr': float(p[5]) if len(p) > 5 else None,
        'fil_path_in_cand': p[-1],
    }


def cluster_candidates(cands, gap_s):
    """Group candidates into events by MJD.

    A new event starts when consecutive (MJD-sorted) candidates are more than
    gap_s apart.  One physical Crab pulse produces one .cands row per trial DM,
    so a gap of ~3 ms merges same-pulse detections without merging rotations.
    """
    cands = sorted(cands, key=lambda c: c['mjd'])
    events = []
    for c in cands:
        if events and (c['mjd'] - events[-1][-1]['mjd']) * 86400.0 <= gap_s:
            events[-1].append(c)
        else:
            events.append([c])
    return events


def pick_representative(event):
    """The highest-SNR candidate of an event (best DM estimate + peak time)."""
    return max(event, key=lambda c: c['snr'] or 0.0)
