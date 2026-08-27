#!/usr/bin/env python3
"""Root shim — delegates to pipeline/extract_cands (kept for backward compat)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
from pipeline.extract_cands.main import main

if __name__ == '__main__':
    main()
