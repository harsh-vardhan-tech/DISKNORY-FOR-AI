#!/usr/bin/env python3
"""Create a backup of the brain folder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
os.chdir(ROOT)
from memory_manager import MemoryManager
mm = MemoryManager("brain")
print("backup at:", mm.backup())
