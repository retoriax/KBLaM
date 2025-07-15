#!/usr/bin/env python3
"""
Command-line interface for the branch comparison tool.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from kblam.utils.branch_comparator import main

if __name__ == "__main__":
    main()