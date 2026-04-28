#!/usr/bin/env python3
"""DISKNORY-FOR-AI - Brain data validator.

Ensures every word entry conforms to brain_schema_v1.json BEFORE being written.
Single change of one word never corrupts the whole brain because every line
is an independent JSONL record + a validator gate sits in front of every write.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Validator:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema: Dict[str, Any] = json.load(f)
        self.required = self.schema.get("required_fields", [])
        self.bool_flags = self.schema.get("boolean_flags", [])
        self.allowed_pos = {
            "noun", "verb", "adjective", "adverb", "pronoun",
            "preposition", "conjunction", "interjection", "phrase", "other",
            "number", "color", "determiner", "particle",
        }

    def validate_word_entry(self, entry: Dict[str, Any]) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        for field_name in self.required:
            if field_name not in entry:
                errors.append(f"missing required field: {field_name}")

        for flag in self.bool_flags:
            if flag in entry and not isinstance(entry[flag], bool):
                errors.append(f"boolean field '{flag}' must be bool")

        if "confidence_score" in entry:
            cs = entry["confidence_score"]
            if not isinstance(cs, (int, float)) or not (0.0 <= float(cs) <= 1.0):
                errors.append(f"confidence_score must be 0..1, got {cs}")

        if "word" in entry and "letters" in entry:
            if entry["letters"] != list(entry["word"]):
                errors.append("letters array does not equal list(word)")

        if "letters" in entry and "letter_count" in entry:
            if entry["letter_count"] != len(entry["letters"]):
                errors.append("letter_count != len(letters)")

        if "part_of_speech" in entry and entry["part_of_speech"] not in self.allowed_pos:
            warnings.append(f"unknown part_of_speech: {entry['part_of_speech']}")

        if "language" in entry and entry["language"] not in {"english", "hindi", "hinglish"}:
            warnings.append(f"unknown language: {entry['language']}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_jsonl_file(self, filepath: str) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        if not os.path.exists(filepath):
            return ValidationResult(False, [f"file not found: {filepath}"])
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_num}: invalid JSON - {exc}")
                    continue
                res = self.validate_word_entry(entry)
                errors.extend([f"line {line_num}: {e}" for e in res.errors])
                warnings.extend([f"line {line_num}: {w}" for w in res.warnings])
        return ValidationResult(len(errors) == 0, errors, warnings)


def validate_all_files(base_path: str = "brain") -> Dict[str, Any]:
    schema_path = os.path.join(base_path, "schema", "brain_schema_v1.json")
    v = Validator(schema_path)
    files = [
        os.path.join(base_path, "data", "english_core.jsonl"),
        os.path.join(base_path, "data", "hindi_core.jsonl"),
        os.path.join(base_path, "data", "hinglish_core.jsonl"),
    ]
    report = {
        "total_files": 0, "files_passed": 0, "files_failed": 0,
        "total_errors": 0, "total_warnings": 0,
        "overall_passed": True, "details": [],
    }
    for fp in files:
        if not os.path.exists(fp):
            continue
        report["total_files"] += 1
        res = v.validate_jsonl_file(fp)
        if res.valid:
            report["files_passed"] += 1
        else:
            report["files_failed"] += 1
            report["overall_passed"] = False
        report["total_errors"] += len(res.errors)
        report["total_warnings"] += len(res.warnings)
        report["details"].append({
            "file": fp, "valid": res.valid,
            "errors": res.errors[:5], "warnings": res.warnings[:5],
        })
    return report


if __name__ == "__main__":
    rep = validate_all_files("brain")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
