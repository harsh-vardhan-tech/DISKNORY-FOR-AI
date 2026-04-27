"""DISKNORY brain validator.

Validates schema, JSONL dictionary files, and index consistency.
Only Python stdlib is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

WORD_ID_PATTERN = re.compile(r"^[A-Z]{3}_\d{3,}_[a-z0-9\-]+$")
ISO_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BOOLEAN_FLAGS = [
    "is_noun",
    "is_verb",
    "is_adjective",
    "is_adverb",
    "is_common",
    "is_technical",
    "is_slang",
    "is_formal",
    "is_informal",
    "is_question_word",
    "is_emotion",
    "is_time_related",
    "is_place_related",
    "is_person_related",
    "is_number",
    "is_color",
    "is_action",
]

REQUIRED_FIELDS = [
    "word_id",
    "word",
    "lowercase",
    "letters",
    "letter_count",
    "hindi_meaning",
    "english_meaning",
    "part_of_speech",
    "confidence_score",
    "learned_from",
    "learned_date",
    "created_by",
    "created_timestamp",
    "version",
    "is_locked",
]


@dataclass
class ValidationError:
    category: str  # CRITICAL | ERROR | WARNING | INFO
    message: str
    file_path: str = ""
    line_number: Optional[int] = None
    entry_id: Optional[str] = None


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entry_id: str = ""


@dataclass
class ValidationReport:
    file_path: str
    total_entries: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class MasterValidationReport:
    timestamp: str
    total_files: int
    files_passed: int
    files_failed: int
    critical_errors: int
    total_errors: int
    total_warnings: int
    reports: List[ValidationReport]
    overall_passed: bool


@dataclass
class RepairReport:
    file_path: str
    changed: bool
    changes_made: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not ISO_DATE_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _is_iso_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not ISO_TS_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def _find_repo_root(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        abs_path = os.path.dirname(abs_path)
    while True:
        if os.path.exists(os.path.join(abs_path, ".git")):
            return abs_path
        parent = os.path.dirname(abs_path)
        if parent == abs_path:
            return os.path.abspath(path)
        abs_path = parent


def _resolve_brain_paths(base_path: str) -> Tuple[str, str, str]:
    base_abs = os.path.abspath(base_path)
    if os.path.isdir(os.path.join(base_abs, "schema")):
        brain_root = base_abs
    elif os.path.isdir(os.path.join(base_abs, "brain", "schema")):
        brain_root = os.path.join(base_abs, "brain")
    else:
        repo_root = _find_repo_root(base_abs)
        brain_root = os.path.join(repo_root, "brain")

    schema_path = os.path.join(brain_root, "schema", "brain_schema_v1.json")
    data_path = os.path.join(brain_root, "data", "english_core.jsonl")
    index_path = os.path.join(brain_root, "indexes", "lexeme_index.json")
    return schema_path, data_path, index_path


def validate_word_entry(entry: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Validate a single word entry against schema requirements."""
    errors: List[ValidationError] = []
    warnings: List[str] = []
    entry_id = str(entry.get("word_id", ""))

    for field_name in REQUIRED_FIELDS:
        if field_name not in entry or entry[field_name] is None:
            errors.append(
                ValidationError("ERROR", f"Field '{field_name}' is missing in entry", entry_id=entry_id)
            )

    type_rules = {
        "word_id": str,
        "word": str,
        "lowercase": str,
        "letters": list,
        "letter_count": int,
        "hindi_meaning": str,
        "english_meaning": str,
        "part_of_speech": str,
        "confidence_score": (int, float),
        "learned_from": str,
        "learned_date": str,
        "created_by": str,
        "created_timestamp": str,
        "version": int,
        "is_locked": bool,
    }
    for key, expected_type in type_rules.items():
        if key in entry and entry[key] is not None and not isinstance(entry[key], expected_type):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"Field '{key}' has invalid type {type(entry[key]).__name__}; expected {expected_type}",
                    entry_id=entry_id,
                )
            )

    for flag in BOOLEAN_FLAGS:
        value = entry.get(flag)
        if not isinstance(value, bool):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"Boolean field '{flag}' has value {value!r} (must be bool)",
                    entry_id=entry_id,
                )
            )

    if "confidence_score" in entry and isinstance(entry.get("confidence_score"), (int, float)):
        score = float(entry["confidence_score"])
        if score < 0.0 or score > 1.0:
            errors.append(
                ValidationError(
                    "WARNING",
                    f"confidence_score {score} exceeds allowed range 0.0-1.0",
                    entry_id=entry_id,
                )
            )

    word = entry.get("word")
    if isinstance(word, str):
        lowercase = entry.get("lowercase")
        if isinstance(lowercase, str) and lowercase != word.lower():
            errors.append(
                ValidationError(
                    "ERROR",
                    f"lowercase '{lowercase}' does not match word.lower() '{word.lower()}'",
                    entry_id=entry_id,
                )
            )

        letters = entry.get("letters")
        if isinstance(letters, list) and letters != list(word):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"letters array {letters!r} does not match word '{word}'",
                    entry_id=entry_id,
                )
            )

        letter_count = entry.get("letter_count")
        if isinstance(letter_count, int) and isinstance(letters, list) and letter_count != len(letters):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"letter_count {letter_count} does not match len(letters) {len(letters)}",
                    entry_id=entry_id,
                )
            )

    word_id = entry.get("word_id")
    if isinstance(word_id, str) and not WORD_ID_PATTERN.match(word_id):
        errors.append(
            ValidationError(
                "ERROR",
                f"word_id '{word_id}' does not match LANG_NNN_wordname format",
                entry_id=entry_id,
            )
        )

    if "created_timestamp" in entry and entry.get("created_timestamp") is not None:
        if not _is_iso_timestamp(entry["created_timestamp"]):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"created_timestamp '{entry['created_timestamp']}' is not valid ISO-8601 UTC",
                    entry_id=entry_id,
                )
            )

    for ts_field in ("modified_timestamp",):
        ts_val = entry.get(ts_field)
        if ts_val is not None and not _is_iso_timestamp(ts_val):
            errors.append(
                ValidationError(
                    "ERROR",
                    f"{ts_field} '{ts_val}' is not valid ISO-8601 UTC",
                    entry_id=entry_id,
                )
            )

    if "learned_date" in entry and entry.get("learned_date") is not None:
        if not _is_iso_date(entry["learned_date"]):
            errors.append(
                ValidationError(
                    "ERROR", f"learned_date '{entry['learned_date']}' is not valid YYYY-MM-DD", entry_id=entry_id
                )
            )

    if isinstance(schema, dict):
        expected_schema_version = (
            schema.get("_schema_info", {}).get("schema_version")
            if isinstance(schema.get("_schema_info"), dict)
            else schema.get("schema_version")
        )
        entry_schema_version = entry.get("schema_version")
        if expected_schema_version and entry_schema_version and entry_schema_version != expected_schema_version:
            warnings.append(
                f"entry schema_version '{entry_schema_version}' differs from schema '{expected_schema_version}'"
            )

    has_critical_or_error = any(e.category in {"CRITICAL", "ERROR"} for e in errors)
    return ValidationResult(valid=not has_critical_or_error, errors=errors, warnings=warnings, entry_id=entry_id)


def validate_jsonl_file(filepath: str, schema: Dict[str, Any]) -> ValidationReport:
    """Validate entire JSONL file and all contained entries."""
    report = ValidationReport(file_path=filepath)

    if not os.path.exists(filepath):
        report.errors.append(ValidationError("CRITICAL", "File does not exist", file_path=filepath))
        report.passed = False
        return report

    seen_ids = set()
    seen_words = set()

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip():
                    report.errors.append(
                        ValidationError("ERROR", "Empty line is not allowed in JSONL", filepath, line_no)
                    )
                    continue
                if raw.lstrip().startswith("#"):
                    continue

                report.total_entries += 1
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    report.invalid_entries += 1
                    report.errors.append(
                        ValidationError(
                            "CRITICAL",
                            f"Invalid JSON on line {line_no}: {exc.msg}",
                            filepath,
                            line_no,
                        )
                    )
                    continue

                result = validate_word_entry(entry, schema)
                for error in result.errors:
                    error.file_path = filepath
                    error.line_number = line_no
                report.errors.extend(result.errors)
                report.warnings.extend(result.warnings)

                word_id = entry.get("word_id")
                if isinstance(word_id, str):
                    if word_id in seen_ids:
                        report.errors.append(
                            ValidationError(
                                "CRITICAL",
                                f"word_id '{word_id}' is duplicate",
                                filepath,
                                line_no,
                                word_id,
                            )
                        )
                    seen_ids.add(word_id)

                lowercase = entry.get("lowercase")
                if isinstance(lowercase, str):
                    lc_key = lowercase.lower()
                    if lc_key in seen_words:
                        report.errors.append(
                            ValidationError(
                                "CRITICAL",
                                f"word '{lowercase}' is duplicate (case-insensitive)",
                                filepath,
                                line_no,
                                str(word_id) if isinstance(word_id, str) else None,
                            )
                        )
                    seen_words.add(lc_key)

                if result.valid:
                    report.valid_entries += 1
                else:
                    report.invalid_entries += 1

    except UnicodeDecodeError as exc:
        report.errors.append(ValidationError("CRITICAL", f"UTF-8 decoding failed: {exc}", filepath))

    report.passed = not any(e.category in {"CRITICAL", "ERROR"} for e in report.errors)
    return report


def validate_json_file(filepath: str) -> ValidationReport:
    """Validate generic JSON file format and basic structure."""
    report = ValidationReport(file_path=filepath, total_entries=1)
    if not os.path.exists(filepath):
        report.errors.append(ValidationError("CRITICAL", "File does not exist", filepath))
        report.passed = False
        return report

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except UnicodeDecodeError as exc:
        report.errors.append(ValidationError("CRITICAL", f"UTF-8 decoding failed: {exc}", filepath))
        report.passed = False
        return report
    except json.JSONDecodeError as exc:
        report.errors.append(
            ValidationError("CRITICAL", f"Invalid JSON: {exc.msg} at line {exc.lineno}", filepath, exc.lineno)
        )
        report.passed = False
        return report

    report.valid_entries = 1

    filename = os.path.basename(filepath)
    if filename == "brain_schema_v1.json":
        required = ["_schema_info", "entry_schema", "validation_rules"]
        for key in required:
            if key not in data:
                report.errors.append(ValidationError("ERROR", f"Missing top-level key '{key}'", filepath))
    elif filename == "lexeme_index.json":
        required = ["_index_info", "word_to_line", "word_id_to_line", "prefix_index", "category_index", "statistics"]
        for key in required:
            if key not in data:
                report.errors.append(ValidationError("ERROR", f"Missing top-level key '{key}'", filepath))

    report.passed = not any(e.category in {"CRITICAL", "ERROR"} for e in report.errors)
    return report


def validate_index_consistency(index_path: str, data_path: str) -> ValidationReport:
    """Validate index file against actual JSONL data file."""
    report = ValidationReport(file_path=index_path)

    if not os.path.exists(index_path):
        report.errors.append(ValidationError("CRITICAL", "Index file does not exist", index_path))
        return report
    if not os.path.exists(data_path):
        report.errors.append(ValidationError("CRITICAL", "Data file does not exist", data_path))
        return report

    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    except Exception as exc:  # keep broad for robustness
        report.errors.append(ValidationError("CRITICAL", f"Could not parse index JSON: {exc}", index_path))
        return report

    data_word_to_line: Dict[str, int] = {}
    data_id_to_line: Dict[str, int] = {}
    data_entries = 0

    with open(data_path, "r", encoding="utf-8") as handle:
        for physical_line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            data_entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(
                    ValidationError(
                        "CRITICAL",
                        f"Data file has invalid JSON at physical line {physical_line_no}: {exc.msg}",
                        data_path,
                        physical_line_no,
                    )
                )
                continue
            lower = entry.get("lowercase")
            word_id = entry.get("word_id")
            if isinstance(lower, str):
                data_word_to_line[lower] = data_entries
            if isinstance(word_id, str):
                data_id_to_line[word_id] = data_entries

    report.total_entries = data_entries
    report.valid_entries = data_entries

    idx_info = index.get("_index_info", {}) if isinstance(index, dict) else {}
    declared_total = idx_info.get("total_words")
    if declared_total != data_entries:
        report.errors.append(
            ValidationError(
                "CRITICAL",
                f"Index total_words ({declared_total}) does not match actual entries ({data_entries})",
                index_path,
            )
        )

    word_to_line = index.get("word_to_line", {}) if isinstance(index, dict) else {}
    word_id_to_line = index.get("word_id_to_line", {}) if isinstance(index, dict) else {}

    if not isinstance(word_to_line, dict):
        report.errors.append(ValidationError("ERROR", "word_to_line is not an object", index_path))
        word_to_line = {}
    if not isinstance(word_id_to_line, dict):
        report.errors.append(ValidationError("ERROR", "word_id_to_line is not an object", index_path))
        word_id_to_line = {}

    for word, line_no in word_to_line.items():
        if not isinstance(line_no, int) or line_no < 1:
            report.errors.append(
                ValidationError("ERROR", f"word_to_line['{word}'] has non-positive line number {line_no!r}", index_path)
            )
        actual = data_word_to_line.get(word)
        if actual is None:
            report.errors.append(ValidationError("ERROR", f"Orphan index entry for word '{word}'", index_path))
        elif actual != line_no:
            report.errors.append(
                ValidationError(
                    "ERROR",
                    f"word_to_line mismatch for '{word}': index={line_no}, data={actual}",
                    index_path,
                )
            )

    for word_id, line_no in word_id_to_line.items():
        if not isinstance(line_no, int) or line_no < 1:
            report.errors.append(
                ValidationError(
                    "ERROR", f"word_id_to_line['{word_id}'] has non-positive line number {line_no!r}", index_path
                )
            )
        actual = data_id_to_line.get(word_id)
        if actual is None:
            report.errors.append(ValidationError("ERROR", f"Orphan index entry for word_id '{word_id}'", index_path))
        elif actual != line_no:
            report.errors.append(
                ValidationError(
                    "ERROR",
                    f"word_id_to_line mismatch for '{word_id}': index={line_no}, data={actual}",
                    index_path,
                )
            )

    for word in data_word_to_line:
        if word not in word_to_line:
            report.errors.append(ValidationError("ERROR", f"Data word '{word}' missing from index", index_path))

    for word_id in data_id_to_line:
        if word_id not in word_id_to_line:
            report.errors.append(ValidationError("ERROR", f"Data word_id '{word_id}' missing from index", index_path))

    prefix_index = index.get("prefix_index", {}) if isinstance(index, dict) else {}
    if isinstance(prefix_index, dict):
        for key in prefix_index.keys():
            if not isinstance(key, str) or len(key) != 1 or not key.isalpha() or key.lower() != key:
                report.errors.append(
                    ValidationError("ERROR", f"prefix_index key '{key}' must be one lowercase letter", index_path)
                )

    category_index = index.get("category_index", {}) if isinstance(index, dict) else {}
    if isinstance(category_index, dict):
        for flag in BOOLEAN_FLAGS:
            if flag not in category_index:
                report.warnings.append(f"category_index missing flag '{flag}'")

    source_file = idx_info.get("source_file") if isinstance(idx_info, dict) else None
    if isinstance(source_file, str):
        expected_suffix = os.path.join("brain", "data", "english_core.jsonl")
        if not (source_file.endswith(expected_suffix) or source_file.endswith("data/english_core.jsonl")):
            report.warnings.append(
                f"_index_info.source_file is '{source_file}', expected path ending with '{expected_suffix}'"
            )

    report.invalid_entries = len([e for e in report.errors if e.category in {"CRITICAL", "ERROR"}])
    report.passed = not any(e.category in {"CRITICAL", "ERROR"} for e in report.errors)
    return report


def validate_all_files(base_path: str) -> MasterValidationReport:
    """Validate schema, data, index and cross-file consistency."""
    schema_path, data_path, index_path = _resolve_brain_paths(base_path)

    reports: List[ValidationReport] = []
    files_passed = 0
    files_failed = 0

    schema_report = validate_json_file(schema_path)
    reports.append(schema_report)

    schema_dict: Dict[str, Any] = {}
    if schema_report.passed:
        with open(schema_path, "r", encoding="utf-8") as handle:
            schema_dict = json.load(handle)

    data_report = validate_jsonl_file(data_path, schema_dict)
    reports.append(data_report)

    index_report = validate_json_file(index_path)
    reports.append(index_report)

    index_consistency_report = validate_index_consistency(index_path, data_path)
    reports.append(index_consistency_report)

    for report in reports:
        if report.passed:
            files_passed += 1
        else:
            files_failed += 1

    critical_errors = sum(1 for r in reports for e in r.errors if e.category == "CRITICAL")
    total_errors = sum(1 for r in reports for e in r.errors if e.category in {"CRITICAL", "ERROR"})
    total_warnings = sum(len(r.warnings) for r in reports) + sum(
        1 for r in reports for e in r.errors if e.category == "WARNING"
    )

    overall_passed = files_failed == 0 and critical_errors == 0 and total_errors == 0

    return MasterValidationReport(
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_files=len(reports),
        files_passed=files_passed,
        files_failed=files_failed,
        critical_errors=critical_errors,
        total_errors=total_errors,
        total_warnings=total_warnings,
        reports=reports,
        overall_passed=overall_passed,
    )


def pre_write_validation(entry: Dict[str, Any], operation: str) -> bool:
    """Quick safety check before single entry write operation."""
    allowed = {"add", "edit", "delete"}
    if operation not in allowed:
        return False

    mock_schema = {"_schema_info": {"schema_version": "1.0.0"}}
    result = validate_word_entry(entry, mock_schema)
    if not result.valid:
        return False

    if operation == "delete" and not isinstance(entry.get("is_locked"), bool):
        return False

    return True


def post_write_verification(filepath: str, line_number: int, entry: Dict[str, Any]) -> bool:
    """Verify the written line exactly matches intended JSON content."""
    if line_number < 1:
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            json_lines = [ln.strip() for ln in handle if ln.strip() and not ln.lstrip().startswith("#")]
    except (OSError, UnicodeDecodeError):
        return False

    if line_number > len(json_lines):
        return False

    try:
        read_back = json.loads(json_lines[line_number - 1])
    except json.JSONDecodeError:
        return False

    return read_back == entry


def repair_common_issues(filepath: str) -> RepairReport:
    """Auto-fix safe formatting issues only; never structural issues."""
    report = RepairReport(file_path=filepath, changed=False)
    if not os.path.exists(filepath):
        report.warnings.append("File does not exist; nothing to repair")
        return report

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            original_lines = handle.readlines()
    except UnicodeDecodeError:
        report.warnings.append("UTF-8 decoding failed; manual repair required")
        return report

    cleaned_lines: List[str] = []
    changed = False
    for line in original_lines:
        new_line = line.rstrip() + "\n"
        if new_line != line:
            changed = True
        cleaned_lines.append(new_line)

    if filepath.endswith(".jsonl"):
        compacted: List[str] = []
        for line in cleaned_lines:
            if not line.strip():
                changed = True
                continue
            compacted.append(line)
        cleaned_lines = compacted

    if changed:
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.writelines(cleaned_lines)
        report.changed = True
        report.changes_made.append("Removed trailing whitespace and compacted empty lines")

    return report


if __name__ == "__main__":
    import sys

    print("🔍 DISKNORY-FOR-AI Validator v1.0")
    print("=" * 50)

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "brain/"

    report = validate_all_files(target)

    print(f"Files checked: {report.total_files}")
    print(f"Files passed: {report.files_passed}")
    print(f"Files failed: {report.files_failed}")
    print(f"Critical errors: {report.critical_errors}")
    print(f"Total errors: {report.total_errors}")
    print(f"Warnings: {report.total_warnings}")
    print("=" * 50)

    if report.overall_passed:
        print("✅ All validations PASSED!")
        sys.exit(0)

    print("❌ Validation FAILED!")
    for file_report in report.reports:
        for error in file_report.errors:
            if error.category == "CRITICAL":
                where = f" ({error.file_path}:{error.line_number})" if error.line_number else f" ({error.file_path})"
                print(f"  CRITICAL: {error.message}{where}")
    sys.exit(1)
