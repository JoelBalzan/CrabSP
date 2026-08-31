"""Small utility helpers."""
from pathlib import Path


def get_tx_resolution(cand_file):
    """Extract TX search resolution label from parent folder, e.g. 4us.

    For merged files like cands/unique.cands (parent=cands) return the
    stem (unique) so callers don't create cands/cands.
    """
    p = Path(cand_file)
    parent = p.parent.name
    # cands/unique.cands or cands/merged.cands -> use stem, not "cands"
    if parent == "cands" and p.stem in ("unique", "merged"):
        return p.stem
    # bare cands/FOO.cands where FOO is not a tres (e.g. cands/clean.cands)
    # also avoid returning "cands"
    if parent == "cands":
        # if stem looks like a tres (endswith us) keep parent, else use stem
        if p.stem.endswith("us"):
            return parent
        return p.stem
    return parent


def tx_res_us(cand_file):
    """Numeric resolution (us) of a .cands file's parent folder, for ordering."""
    name = get_tx_resolution(cand_file)
    try:
        return float(name.replace('us', ''))
    except ValueError:
        return float('inf')
