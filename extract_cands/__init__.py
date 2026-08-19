"""extract_cands — match transientX MJD candidates to raw .dada fragments,
pull full-Stokes cutouts, and save as .npz cubes.
"""
from .main import main

__all__ = ['main']
