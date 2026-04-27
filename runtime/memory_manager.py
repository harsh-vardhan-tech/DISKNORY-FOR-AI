"""DISKNORY memory manager.

Main control engine for safe add/edit/delete/replace/learn operations
on brain data with validation, rollback, journaling and index updates.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from runtime import validator
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    import validator  # type: ignore


@dataclass
class OperationResult:
    success: bool
    operation_type: str
    word_id: str = ""
    word: str = ""
    timestamp: str = ""
    version: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rollback_id: str = ""
    message: str = ""


@dataclass
class MergeReport:
    success: bool
    merged: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""


class MemoryManagerError(Exception):
    """Base memory manager error."""


class MMValidationError(MemoryManagerError):
    """Raised when validation fails."""


class MMIndexError(MemoryManagerError):
    """Raised when index is inconsistent."""


class LockError(MemoryManagerError):
    """Raised when locked entries are modified."""


class DuplicateError(MemoryManagerError):
    """Raised when duplicate words/ids are inserted."""


class NotFoundError(MemoryManagerError):
    """Raised when requested word does not exist."""


class CorruptionError(MemoryManagerError):
    """Raised when post-write verification fails."""


class RollbackError(MemoryManagerError):
    """Raised when rollback restoration fails."""


class MemoryManager:
    """AI brain control engine with safety-first write operations."""

    CACHE_LIMIT = 100
    CACHE_TTL_SECONDS = 300

    def __init__(self, base_path: str = "brain/"):
        self.base_path = base_path.rstrip("/")
        self.schema_path = os.path.join(self.base_path, "schema", "brain_schema_v1.json")
        self.data_path = os.path.join(self.base_path, "data", "english_core.jsonl")
        self.archive_path = os.path.join(self.base_path, "data", "archived.jsonl")
        self.index_path = os.path.join(self.base_path, "indexes", "lexeme_index.json")
        self.journal_dir = os.path.join(self.base_path, "journal")
        self.events_path = os.path.join(self.journal_dir, "events.log")
        self.rollback_path = os.path.join(self.journal_dir, "rollback_points.json")

        self._ensure_paths()

        self.schema = self._load_schema()
        self.index = self._load_index()
        self.cache: "OrderedDict[str, Tuple[datetime, Dict[str, Any]]]" = OrderedDict()
        self.journal: List[Dict[str, Any]] = []

    def _ensure_paths(self) -> None:
        os.makedirs(os.path.join(self.base_path, "schema"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "indexes"), exist_ok=True)
        os.makedirs(self.journal_dir, exist_ok=True)
        if not os.path.exists(self.events_path):
            with open(self.events_path, "w", encoding="utf-8"):
                pass
        if not os.path.exists(self.rollback_path):
            with open(self.rollback_path, "w", encoding="utf-8") as handle:
                json.dump([], handle)

    def _now_iso(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _today_iso(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _load_schema(self) -> Dict[str, Any]:
        with open(self.schema_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_index(self) -> Dict[str, Any]:
        with open(self.index_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_index(self) -> None:
        self._atomic_write_json(self.index_path, self.index)

    def _atomic_write_json(self, path: str, payload: Dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _read_jsonl_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(json.loads(line))
        return entries

    def _write_jsonl_entries(self, entries: List[Dict[str, Any]], include_header: bool = True) -> None:
        directory = os.path.dirname(self.data_path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".jsonl", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if include_header:
                    handle.write("# DISKNORY English core dictionary JSONL v1 (1000 entries)\n")
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            os.replace(tmp_path, self.data_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _load_entry(self, word_id: str) -> Dict[str, Any]:
        line = self.index.get("word_id_to_line", {}).get(word_id)
        if not isinstance(line, int):
            raise NotFoundError(f"word_id not found: {word_id}")
        entries = self._read_jsonl_entries()
        if line < 1 or line > len(entries):
            raise MMIndexError(f"Index line out of bounds for {word_id}: {line}")
        return entries[line - 1]

    def _save_entry(self, entry: Dict[str, Any], line_number: int) -> bool:
        entries = self._read_jsonl_entries()
        if line_number < 1 or line_number > len(entries):
            return False
        entries[line_number - 1] = entry
        self._write_jsonl_entries(entries)
        return True

    def _append_entry(self, entry: Dict[str, Any]) -> int:
        entries = self._read_jsonl_entries()
        entries.append(entry)
        self._write_jsonl_entries(entries)
        return len(entries)

    def _next_word_number(self) -> int:
        max_n = 0
        for word_id in self.index.get("word_id_to_line", {}).keys():
            parts = word_id.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                max_n = max(max_n, int(parts[1]))
        return max_n + 1

    def _sanitize_word_token(self, word: str) -> str:
        token = "".join(ch.lower() if ch.isalnum() else "-" for ch in word.strip())
        while "--" in token:
            token = token.replace("--", "-")
        return token.strip("-") or "word"

    def _generate_word_id(self, word: str) -> str:
        n = self._next_word_number()
        return f"ENG_{n:03d}_{self._sanitize_word_token(word)}"

    def _compute_flags(self, entry: Dict[str, Any]) -> Dict[str, bool]:
        pos = str(entry.get("part_of_speech", "")).lower()
        word = str(entry.get("lowercase", "")).lower()
        context = set(entry.get("usage_context") or [])
        semantic = set(entry.get("related_concepts") or [])

        return {
            "is_noun": pos == "noun",
            "is_verb": pos == "verb",
            "is_adjective": pos == "adjective",
            "is_adverb": pos == "adverb",
            "is_common": True,
            "is_technical": "technical" in context or "tech" in semantic,
            "is_slang": "slang" in context,
            "is_formal": "formal" in context,
            "is_informal": "informal" in context or "casual" in context,
            "is_question_word": word in {"what", "why", "when", "where", "who", "how", "which", "whom"},
            "is_emotion": "emotion" in semantic,
            "is_time_related": word in {"time", "day", "week", "month", "year", "hour", "minute", "second"},
            "is_place_related": "place" in semantic,
            "is_person_related": word in {"person", "man", "woman", "child", "people"},
            "is_number": word.isdigit() or word in {"one", "two", "three", "four", "five", "ten", "hundred"},
            "is_color": word in {"red", "blue", "green", "yellow", "black", "white", "pink", "brown"},
            "is_action": pos == "verb",
        }

    def _prepare_entry_defaults(self, entry: Dict[str, Any], created_by: str = "ai") -> Dict[str, Any]:
        e = dict(entry)
        word = e.get("word") or e.get("lowercase")
        if not isinstance(word, str) or not word.strip():
            raise MMValidationError("word is required")
        word = word.strip()
        lowercase = word.lower()

        e.setdefault("word_id", self._generate_word_id(lowercase))
        e["word"] = word
        e["lowercase"] = lowercase
        e["letters"] = list(word)
        e["letter_count"] = len(e["letters"])

        e.setdefault("hindi_meaning", f"{word} (हिंदी अर्थ जोड़ना बाकी)")
        e.setdefault("english_meaning", f"Core meaning of {word}")
        e.setdefault("part_of_speech", "noun")
        e.setdefault("confidence_score", 1.0)
        e.setdefault("learned_from", "dictionary")
        e.setdefault("learned_date", self._today_iso())
        e.setdefault("created_by", created_by)
        e.setdefault("created_timestamp", self._now_iso())
        e.setdefault("modified_by", created_by)
        e.setdefault("modified_timestamp", self._now_iso())
        e.setdefault("version", 1)
        e.setdefault("is_locked", False)

        optional_defaults = {
            "syllables": None,
            "phonetic_ipa": None,
            "phonetic_simple": None,
            "audio_available": False,
            "grammar_role": None,
            "root_word": lowercase,
            "word_family": [lowercase],
            "plural_form": None,
            "tense_forms": None,
            "comparative": None,
            "superlative": None,
            "definition_simple": e["english_meaning"],
            "definition_detailed": e["english_meaning"],
            "synonyms": [],
            "antonyms": [],
            "example_sentence_en": f"I used the word {word} in a sentence.",
            "example_sentence_hi": f"मैंने {word} शब्द का वाक्य में उपयोग किया।",
            "usage_context": ["general"],
            "common_phrases": [],
            "times_used": 0,
            "last_used": None,
            "user_corrections": [],
            "alternative_meanings": [],
            "related_concepts": [],
        }
        for key, val in optional_defaults.items():
            e.setdefault(key, val)

        e.update(self._compute_flags(e))
        return e

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        item = self.cache.get(key)
        if not item:
            return None
        last_seen, entry = item
        if datetime.utcnow() - last_seen > timedelta(seconds=self.CACHE_TTL_SECONDS):
            self.cache.pop(key, None)
            return None
        self.cache.move_to_end(key)
        self.cache[key] = (datetime.utcnow(), entry)
        return dict(entry)

    def _cache_set(self, key: str, entry: Dict[str, Any]) -> None:
        self.cache[key] = (datetime.utcnow(), dict(entry))
        self.cache.move_to_end(key)
        while len(self.cache) > self.CACHE_LIMIT:
            self.cache.popitem(last=False)

    def _cache_invalidate(self, entry: Dict[str, Any]) -> None:
        for key in (entry.get("word_id"), entry.get("lowercase")):
            if isinstance(key, str):
                self.cache.pop(key, None)

    def _update_index(self, entry: Dict[str, Any], operation: str, old_entry: Optional[Dict[str, Any]] = None, line_number: Optional[int] = None) -> None:
        word_to_line = self.index.setdefault("word_to_line", {})
        word_id_to_line = self.index.setdefault("word_id_to_line", {})
        prefix_index = self.index.setdefault("prefix_index", {})
        category_index = self.index.setdefault("category_index", {})
        deleted_lines = self.index.setdefault("deleted_lines", [])

        if operation == "add":
            assert line_number is not None
            word_to_line[entry["lowercase"]] = line_number
            word_id_to_line[entry["word_id"]] = line_number
        elif operation in {"edit", "replace"}:
            assert line_number is not None
            if old_entry and old_entry.get("lowercase") != entry.get("lowercase"):
                word_to_line.pop(old_entry["lowercase"], None)
                word_to_line[entry["lowercase"]] = line_number
            if old_entry and old_entry.get("word_id") != entry.get("word_id"):
                word_id_to_line.pop(old_entry["word_id"], None)
            word_id_to_line[entry["word_id"]] = line_number
        elif operation == "delete":
            if line_number is not None and line_number not in deleted_lines:
                deleted_lines.append(line_number)

        for k in list(prefix_index.keys()):
            prefix_index[k] = [wid for wid in prefix_index[k] if isinstance(wid, str) and wid in word_id_to_line]

        for letter in "abcdefghijklmnopqrstuvwxyz":
            prefix_index.setdefault(letter, [])

        for wid, ln in word_id_to_line.items():
            if ln in deleted_lines:
                continue
            word = None
            for w, wln in word_to_line.items():
                if wln == ln:
                    word = w
                    break
            if word and word[0].isalpha() and wid not in prefix_index[word[0]]:
                prefix_index[word[0]].append(wid)

        for flag in validator.BOOLEAN_FLAGS:
            category_index[flag] = []

        entries = self._read_jsonl_entries()
        for idx, item in enumerate(entries, start=1):
            if idx in deleted_lines:
                continue
            wid = item.get("word_id")
            if not isinstance(wid, str):
                continue
            for flag in validator.BOOLEAN_FLAGS:
                if bool(item.get(flag)):
                    category_index[flag].append(wid)

        stats = self.index.setdefault("statistics", {})
        active_entries = [e for i, e in enumerate(entries, start=1) if i not in deleted_lines]
        stats["total_nouns"] = sum(1 for e in active_entries if e.get("is_noun") is True)
        stats["total_verbs"] = sum(1 for e in active_entries if e.get("is_verb") is True)
        stats["total_adjectives"] = sum(1 for e in active_entries if e.get("is_adjective") is True)
        stats["total_adverbs"] = sum(1 for e in active_entries if e.get("is_adverb") is True)
        stats["total_common"] = sum(1 for e in active_entries if e.get("is_common") is True)
        stats["total_technical"] = sum(1 for e in active_entries if e.get("is_technical") is True)
        if active_entries:
            lengths = [len(str(e.get("lowercase", ""))) for e in active_entries]
            scores = [float(e.get("confidence_score", 0.0)) for e in active_entries]
            stats["avg_word_length"] = round(sum(lengths) / len(lengths), 2)
            stats["min_confidence"] = min(scores)
            stats["max_confidence"] = max(scores)
            stats["avg_confidence"] = round(sum(scores) / len(scores), 4)

        info = self.index.setdefault("_index_info", {})
        info["total_words"] = len(active_entries)
        info["last_updated"] = self._today_iso()

        self._save_index()

    def _log_operation(self, operation: Dict[str, Any]) -> None:
        self.journal.append(operation)
        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _create_rollback_point(self) -> str:
        rollback_id = f"rollback_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        rb_dir = os.path.join(self.journal_dir, "rollback_store", rollback_id)
        os.makedirs(rb_dir, exist_ok=True)

        snapshot = {
            "rollback_id": rollback_id,
            "timestamp": self._now_iso(),
            "files": {
                "data": os.path.join(rb_dir, "english_core.jsonl"),
                "index": os.path.join(rb_dir, "lexeme_index.json"),
                "archive": os.path.join(rb_dir, "archived.jsonl"),
            },
        }
        shutil.copy2(self.data_path, snapshot["files"]["data"])
        shutil.copy2(self.index_path, snapshot["files"]["index"])
        if os.path.exists(self.archive_path):
            shutil.copy2(self.archive_path, snapshot["files"]["archive"])

        points: List[Dict[str, Any]] = []
        with open(self.rollback_path, "r", encoding="utf-8") as handle:
            points = json.load(handle)
        points.append(snapshot)
        self._atomic_write_json(self.rollback_path, points)
        return rollback_id

    def _rollback(self, rollback_id: str) -> bool:
        try:
            with open(self.rollback_path, "r", encoding="utf-8") as handle:
                points = json.load(handle)
        except Exception as exc:
            raise RollbackError(f"Unable to read rollback points: {exc}")

        point = next((p for p in points if p.get("rollback_id") == rollback_id), None)
        if not point:
            raise RollbackError(f"Rollback point not found: {rollback_id}")

        files = point.get("files", {})
        try:
            shutil.copy2(files["data"], self.data_path)
            shutil.copy2(files["index"], self.index_path)
            if os.path.exists(files.get("archive", "")):
                shutil.copy2(files["archive"], self.archive_path)
        except Exception as exc:
            raise RollbackError(f"Rollback failed: {exc}")

        self.index = self._load_index()
        self.cache.clear()
        return True

    def _line_for_word_id(self, word_id: str) -> int:
        line = self.index.get("word_id_to_line", {}).get(word_id)
        if not isinstance(line, int):
            raise NotFoundError(f"word_id not found: {word_id}")
        return line

    def _line_for_word(self, word: str) -> int:
        line = self.index.get("word_to_line", {}).get(word.lower())
        if not isinstance(line, int):
            raise NotFoundError(f"word not found: {word}")
        return line

    def _result(self, success: bool, op: str, entry: Optional[Dict[str, Any]] = None, rollback_id: str = "", msg: str = "", errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None) -> OperationResult:
        e = entry or {}
        return OperationResult(
            success=success,
            operation_type=op,
            word_id=str(e.get("word_id", "")),
            word=str(e.get("word", "")),
            timestamp=self._now_iso(),
            version=int(e.get("version", 0) or 0),
            errors=errors or [],
            warnings=warnings or [],
            rollback_id=rollback_id,
            message=msg,
        )

    def add_word(self, entry: Dict[str, Any]) -> OperationResult:
        try:
            prepared = self._prepare_entry_defaults(entry)
            word_lc = prepared["lowercase"]
            if word_lc in self.index.get("word_to_line", {}):
                raise DuplicateError(f"Word already exists: {word_lc}")
            if prepared["word_id"] in self.index.get("word_id_to_line", {}):
                prepared["word_id"] = self._generate_word_id(word_lc)

            if not validator.pre_write_validation(prepared, "add"):
                raise MMValidationError("Pre-write validation failed")

            rollback_id = self._create_rollback_point()
            new_line = self._append_entry(prepared)
            self._update_index(prepared, operation="add", line_number=new_line)

            if not validator.post_write_verification(self.data_path, new_line, prepared):
                self._rollback(rollback_id)
                raise CorruptionError("Post-write verification failed")

            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": self._now_iso(),
                "operation": "add",
                "word_id": prepared["word_id"],
                "word": prepared["word"],
                "old_version": None,
                "new_version": prepared["version"],
                "changes": {"created": True},
                "performed_by": "ai",
                "rollback_id": rollback_id,
                "status": "success",
            }
            self._log_operation(event)
            self._cache_set(prepared["word_id"], prepared)
            self._cache_set(prepared["lowercase"], prepared)
            return self._result(True, "add", prepared, rollback_id, "Word added successfully")
        except Exception as exc:
            return self._result(False, "add", entry, msg="Add failed", errors=[str(exc)])

    def edit_word(self, word_id: str, updates: Dict[str, Any]) -> OperationResult:
        try:
            line_no = self._line_for_word_id(word_id)
            old_entry = self._load_entry(word_id)
            if old_entry.get("is_locked"):
                raise LockError(f"Word is locked: {word_id}")

            new_entry = dict(old_entry)
            new_entry.update(updates)
            new_entry["word_id"] = old_entry["word_id"]
            new_entry["created_timestamp"] = old_entry.get("created_timestamp")
            new_entry["created_by"] = old_entry.get("created_by", "system")
            new_entry["version"] = int(old_entry.get("version", 1)) + 1
            new_entry["modified_by"] = "ai"
            new_entry["modified_timestamp"] = self._now_iso()
            new_entry = self._prepare_entry_defaults(new_entry, created_by=old_entry.get("created_by", "system"))
            new_entry["word_id"] = old_entry["word_id"]
            new_entry["version"] = int(old_entry.get("version", 1)) + 1

            if not validator.pre_write_validation(new_entry, "edit"):
                raise MMValidationError("Pre-write validation failed")

            rollback_id = self._create_rollback_point()
            if not self._save_entry(new_entry, line_no):
                raise CorruptionError("Could not write updated entry")

            self._update_index(new_entry, operation="edit", old_entry=old_entry, line_number=line_no)
            if not validator.post_write_verification(self.data_path, line_no, new_entry):
                self._rollback(rollback_id)
                raise CorruptionError("Post-write verification failed")

            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": self._now_iso(),
                "operation": "edit",
                "word_id": new_entry["word_id"],
                "word": new_entry["word"],
                "old_version": old_entry.get("version"),
                "new_version": new_entry.get("version"),
                "changes": updates,
                "performed_by": "ai",
                "rollback_id": rollback_id,
                "status": "success",
            }
            self._log_operation(event)
            self._cache_invalidate(old_entry)
            self._cache_set(new_entry["word_id"], new_entry)
            self._cache_set(new_entry["lowercase"], new_entry)
            return self._result(True, "edit", new_entry, rollback_id, "Word edited successfully")
        except Exception as exc:
            return self._result(False, "edit", {"word_id": word_id}, msg="Edit failed", errors=[str(exc)])

    def delete_word(self, word_id: str, reason: str = "") -> OperationResult:
        try:
            line_no = self._line_for_word_id(word_id)
            entry = self._load_entry(word_id)
            rollback_id = self._create_rollback_point()

            entry["is_locked"] = True
            entry["modified_by"] = "ai"
            entry["modified_timestamp"] = self._now_iso()
            entry["version"] = int(entry.get("version", 1)) + 1

            if not validator.pre_write_validation(entry, "delete"):
                raise MMValidationError("Pre-write validation failed")

            with open(self.archive_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

            if not self._save_entry(entry, line_no):
                raise CorruptionError("Could not mark entry locked for delete")

            self._update_index(entry, operation="delete", line_number=line_no)
            if not validator.post_write_verification(self.data_path, line_no, entry):
                self._rollback(rollback_id)
                raise CorruptionError("Post-write verification failed")

            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": self._now_iso(),
                "operation": "delete",
                "word_id": entry["word_id"],
                "word": entry["word"],
                "old_version": int(entry["version"]) - 1,
                "new_version": int(entry["version"]),
                "changes": {"is_locked": True, "reason": reason},
                "performed_by": "ai",
                "rollback_id": rollback_id,
                "status": "success",
            }
            self._log_operation(event)
            self._cache_invalidate(entry)
            return self._result(True, "delete", entry, rollback_id, "Word archived and marked deleted")
        except Exception as exc:
            return self._result(False, "delete", {"word_id": word_id}, msg="Delete failed", errors=[str(exc)])

    def replace_word(self, word_id: str, new_entry: Dict[str, Any]) -> OperationResult:
        try:
            line_no = self._line_for_word_id(word_id)
            old_entry = self._load_entry(word_id)
            if old_entry.get("is_locked"):
                raise LockError(f"Word is locked: {word_id}")

            replacement = self._prepare_entry_defaults(new_entry)
            replacement["word_id"] = old_entry["word_id"]
            replacement["created_timestamp"] = old_entry.get("created_timestamp")
            replacement["created_by"] = old_entry.get("created_by", "system")
            replacement["version"] = int(old_entry.get("version", 1)) + 1
            replacement["modified_by"] = "ai"
            replacement["modified_timestamp"] = self._now_iso()

            if not validator.pre_write_validation(replacement, "edit"):
                raise MMValidationError("Pre-write validation failed")

            rollback_id = self._create_rollback_point()
            if not self._save_entry(replacement, line_no):
                raise CorruptionError("Could not write replacement entry")

            self._update_index(replacement, operation="replace", old_entry=old_entry, line_number=line_no)
            if not validator.post_write_verification(self.data_path, line_no, replacement):
                self._rollback(rollback_id)
                raise CorruptionError("Post-write verification failed")

            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": self._now_iso(),
                "operation": "replace",
                "word_id": replacement["word_id"],
                "word": replacement["word"],
                "old_version": old_entry.get("version"),
                "new_version": replacement.get("version"),
                "changes": {"replace": True},
                "performed_by": "ai",
                "rollback_id": rollback_id,
                "status": "success",
            }
            self._log_operation(event)
            self._cache_invalidate(old_entry)
            self._cache_set(replacement["word_id"], replacement)
            self._cache_set(replacement["lowercase"], replacement)
            return self._result(True, "replace", replacement, rollback_id, "Word replaced successfully")
        except Exception as exc:
            return self._result(False, "replace", {"word_id": word_id}, msg="Replace failed", errors=[str(exc)])

    def learn_meaning(self, word: str, new_meaning: Dict[str, Any]) -> OperationResult:
        try:
            entry = self.get_word(word)
            if not entry:
                raise NotFoundError(f"word not found: {word}")
            word_id = entry["word_id"]
            line_no = self._line_for_word_id(word_id)

            old_entry = dict(entry)
            alternatives = list(entry.get("alternative_meanings") or [])
            for item in new_meaning.get("alternative_meanings", []):
                if item not in alternatives:
                    alternatives.append(item)
            entry["alternative_meanings"] = alternatives

            concepts = list(entry.get("related_concepts") or [])
            for concept in new_meaning.get("related_concepts", []):
                if concept not in concepts:
                    concepts.append(concept)
            entry["related_concepts"] = concepts

            incoming_conf = new_meaning.get("confidence_score")
            if isinstance(incoming_conf, (int, float)):
                base = float(entry.get("confidence_score", 0.5))
                entry["confidence_score"] = max(0.0, min(1.0, round((base + float(incoming_conf)) / 2.0, 4)))

            entry["version"] = int(entry.get("version", 1)) + 1
            entry["modified_by"] = "ai"
            entry["modified_timestamp"] = self._now_iso()
            entry = self._prepare_entry_defaults(entry)
            entry["word_id"] = old_entry["word_id"]

            if not validator.pre_write_validation(entry, "edit"):
                raise MMValidationError("Pre-write validation failed")

            rollback_id = self._create_rollback_point()
            if not self._save_entry(entry, line_no):
                raise CorruptionError("Could not write learning update")

            self._update_index(entry, operation="edit", old_entry=old_entry, line_number=line_no)
            if not validator.post_write_verification(self.data_path, line_no, entry):
                self._rollback(rollback_id)
                raise CorruptionError("Post-write verification failed")

            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": self._now_iso(),
                "operation": "learn",
                "word_id": entry["word_id"],
                "word": entry["word"],
                "old_version": old_entry.get("version"),
                "new_version": entry.get("version"),
                "changes": new_meaning,
                "performed_by": "ai",
                "rollback_id": rollback_id,
                "status": "success",
            }
            self._log_operation(event)
            self._cache_set(entry["word_id"], entry)
            self._cache_set(entry["lowercase"], entry)
            return self._result(True, "learn", entry, rollback_id, "Meaning learned successfully")
        except Exception as exc:
            return self._result(False, "learn", {"word": word}, msg="Learn failed", errors=[str(exc)])

    def merge_user_data(self, user_memory_path: str) -> MergeReport:
        report = MergeReport(success=False, timestamp=self._now_iso())
        if not os.path.exists(user_memory_path):
            report.errors.append(f"user_memory_path does not exist: {user_memory_path}")
            return report

        try:
            with open(user_memory_path, "r", encoding="utf-8") as handle:
                user_entries = [json.loads(line) for line in handle if line.strip() and not line.lstrip().startswith("#")]
        except Exception as exc:
            report.errors.append(f"Failed to read user memory: {exc}")
            return report

        for user_entry in user_entries:
            word = str(user_entry.get("word", "")).strip()
            if not word:
                report.skipped += 1
                report.warnings.append("Skipped user entry without word")
                continue

            existing = self.get_word(word)
            if existing:
                updates = dict(user_entry)
                updates.pop("word_id", None)
                res = self.edit_word(existing["word_id"], updates)
            else:
                user_entry.setdefault("created_by", "user")
                user_entry.setdefault("learned_from", "user")
                res = self.add_word(user_entry)

            if res.success:
                report.merged += 1
            else:
                report.skipped += 1
                report.errors.extend(res.errors)

        report.success = len(report.errors) == 0
        event = {
            "event_id": uuid.uuid4().hex,
            "timestamp": self._now_iso(),
            "operation": "merge",
            "word_id": "",
            "word": "",
            "old_version": None,
            "new_version": None,
            "changes": {"merged": report.merged, "skipped": report.skipped},
            "performed_by": "ai",
            "rollback_id": "",
            "status": "success" if report.success else "failed",
        }
        self._log_operation(event)
        return report

    def get_word(self, word: str) -> Dict[str, Any]:
        key = word.lower()
        cached = self._cache_get(key)
        if cached:
            return cached

        line_no = self.index.get("word_to_line", {}).get(key)
        if not isinstance(line_no, int):
            return {}
        deleted = set(self.index.get("deleted_lines", []))
        if line_no in deleted:
            return {}

        entries = self._read_jsonl_entries()
        if line_no < 1 or line_no > len(entries):
            return {}

        entry = entries[line_no - 1]
        self._cache_set(key, entry)
        if isinstance(entry.get("word_id"), str):
            self._cache_set(entry["word_id"], entry)
        return entry

    def search_words(self, prefix: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = self._read_jsonl_entries()
        deleted = set(self.index.get("deleted_lines", []))
        result: List[Dict[str, Any]] = []

        allowed_ids: Optional[set] = None
        if category:
            ids = self.index.get("category_index", {}).get(category)
            if isinstance(ids, list):
                allowed_ids = set(ids)
            else:
                return []

        prefix_lc = prefix.lower() if isinstance(prefix, str) else None

        for line_no, entry in enumerate(entries, start=1):
            if line_no in deleted:
                continue
            wid = entry.get("word_id")
            word = str(entry.get("lowercase", ""))

            if allowed_ids is not None and wid not in allowed_ids:
                continue
            if prefix_lc and not word.startswith(prefix_lc):
                continue
            result.append(entry)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        info = self.index.get("_index_info", {})
        stats = self.index.get("statistics", {})
        return {
            "total_words": info.get("total_words", 0),
            "last_updated": info.get("last_updated"),
            **stats,
            "cache_size": len(self.cache),
        }

    def validate_brain(self) -> validator.MasterValidationReport:
        return validator.validate_all_files(self.base_path)

    def backup_brain(self, backup_path: str) -> bool:
        try:
            os.makedirs(backup_path, exist_ok=True)
            for src in [self.schema_path, self.data_path, self.index_path, self.archive_path, self.events_path, self.rollback_path]:
                if os.path.exists(src):
                    rel = os.path.relpath(src, self.base_path)
                    dest = os.path.join(backup_path, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)
            return True
        except Exception:
            return False

    def restore_brain(self, backup_path: str) -> bool:
        try:
            for rel in [
                os.path.join("schema", "brain_schema_v1.json"),
                os.path.join("data", "english_core.jsonl"),
                os.path.join("indexes", "lexeme_index.json"),
                os.path.join("data", "archived.jsonl"),
                os.path.join("journal", "events.log"),
                os.path.join("journal", "rollback_points.json"),
            ]:
                src = os.path.join(backup_path, rel)
                if os.path.exists(src):
                    dest = os.path.join(self.base_path, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)
            self.index = self._load_index()
            self.cache.clear()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Basic smoke check.
    manager = MemoryManager("brain/")
    stats = manager.get_statistics()
    print("DISKNORY MemoryManager ready")
    print(f"Total words: {stats.get('total_words')}")
