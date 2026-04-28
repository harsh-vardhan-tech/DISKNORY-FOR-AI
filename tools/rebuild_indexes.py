#!/usr/bin/env python3
"""Rebuild lexeme + prefix indexes from JSONL data."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
os.chdir(ROOT)
from memory_manager import MemoryManager
mm = MemoryManager("brain")
n = mm.rebuild_indexes()
print(f"rebuilt: {n} words indexed")
