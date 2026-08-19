#!/usr/bin/env python3
"""Backward-compatibility shim — delegates to the extract_cands package.

This file exists so that `python3 extract_cands.py` (used by run_pipeline.sh)
continues to work.  The actual code lives in extract_cands/main.py and its
submodules.
"""
from extract_cands.main import main

if __name__ == '__main__':
    main()
