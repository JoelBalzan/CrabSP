"""Small utility helpers."""
from pathlib import Path


def get_tx_resolution(cand_file):
    """Extract TX search resolution label from parent folder, e.g. 4us."""
    return Path(cand_file).parent.name


def tx_res_us(cand_file):
    """Numeric resolution (us) of a .cands file's parent folder, for ordering."""
    name = get_tx_resolution(cand_file)
    try:
        return float(name.replace('us', ''))
    except ValueError:
        return float('inf')
