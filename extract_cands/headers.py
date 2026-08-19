"""Sigproc .fil and psrdada .dada header parsing, plus .dada file cropping."""
import re
from pathlib import Path

import numpy as np
from sigpyproc.readers import FilReader


def parse_fil_header(fil_path):
    """Read sigproc header fields we need via sigpyproc."""
    fil = FilReader(str(fil_path))
    h = fil.header
    nifs = getattr(h, 'nifs', 1)
    return {
        'tstart_mjd': float(h.tstart),
        'tsamp_s': float(h.tsamp),
        'nsamp': int(h.nsamples),
        'obslen_s': float(h.tsamp) * int(h.nsamples),
        'f1_mhz': float(h.fch1),
        'bw_mhz': float(h.foff),
        'nchan': int(h.nchans),
        'nifs': int(nifs),
    }


def parse_dada_header(dada_path):
    """Read the 4096-byte DADA key/value header into a dict."""
    out = {}
    try:
        with open(dada_path, 'rb') as f:
            header = f.read(4096).decode('latin-1', errors='replace')
    except OSError:
        return out
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', parts[0]):
            out[parts[0]] = parts[1]
    return out


def crop_dada_file(dada_path, offset_s, dur_s, out_path, hdr_size=4096,
                   true_obslen_s=None):
    """Write a small standalone .dada file covering [offset_s, offset_s+dur_s).

    dspsr's duration-limiting flags don't compose safely with -seek, so the
    only reliable way to bound dspsr's read time is to shrink the input file
    so EOF is naturally close by.

    Byte rate is derived from (file size - header size) / true_obslen_s when
    provided (self-consistent with the rest of the pipeline's timing math),
    falling back to the header BYTES_PER_SECOND otherwise.

    Returns out_path, or raises RuntimeError on failure.
    """
    hdr = parse_dada_header(dada_path)
    hdr_size = int(hdr.get('HDR_SIZE', hdr_size))

    if true_obslen_s:
        file_size = Path(dada_path).stat().st_size
        data_size = file_size - hdr_size
        if data_size <= 0 or true_obslen_s <= 0:
            raise RuntimeError(f"{dada_path}: can't derive byte rate from "
                               f"file_size={file_size}, "
                               f"true_obslen_s={true_obslen_s}")
        bytes_per_second = data_size / true_obslen_s
    else:
        bytes_per_second = hdr.get('BYTES_PER_SECOND')
        if not bytes_per_second:
            raise RuntimeError(f"{dada_path}: no true_obslen_s given and no "
                               f"BYTES_PER_SECOND in DADA header, can't crop "
                               f"safely")
        bytes_per_second = float(bytes_per_second)

    resolution = int(hdr.get('RESOLUTION', 1) or 1)

    byte_start = int(offset_s * bytes_per_second)
    byte_count = int(np.ceil(dur_s * bytes_per_second))
    if resolution > 1:
        byte_start -= byte_start % resolution
        rem = byte_count % resolution
        if rem:
            byte_count += resolution - rem

    orig_offset = int(hdr.get('OBS_OFFSET', 0) or 0)
    new_offset = orig_offset + byte_start

    with open(dada_path, 'rb') as f:
        header_bytes = f.read(hdr_size)
        f.seek(hdr_size + byte_start)
        data = f.read(byte_count)
    if len(data) < byte_count:
        raise RuntimeError(f"{dada_path}: requested {byte_count} bytes at "
                           f"offset {byte_start}, only {len(data)} available "
                           f"-- crop window runs past this file's end")

    header_text = header_bytes.decode('latin-1', errors='replace')
    lines = header_text.split('\n')
    out_lines = []
    replaced = False
    for line in lines:
        if line.strip().split()[:1] == ['OBS_OFFSET']:
            out_lines.append(f'OBS_OFFSET  {new_offset}')
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.insert(1, f'OBS_OFFSET  {new_offset}')
    new_header = '\n'.join(out_lines).encode('latin-1', errors='replace')
    new_header = new_header[:hdr_size].ljust(hdr_size, b'\x00')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(new_header)
        f.write(data)
    return out_path
