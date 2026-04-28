#!/usr/bin/env python3
"""Run validation on all brain JSONL files."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
os.chdir(ROOT)
from validator import validate_all_files
rep = validate_all_files("brain")
print(json.dumps(rep, indent=2, ensure_ascii=False))
sys.exit(0 if rep["overall_passed"] else 1)
