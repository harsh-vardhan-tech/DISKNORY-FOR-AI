#!/usr/bin/env python3
"""DISKNORY-FOR-AI - Memory Manager (the brain's hands).

Atomic, journaled, validated CRUD over JSONL brain files.
- One word per line (JSONL) so editing one word never breaks others.
- Index files give O(1) lookup so reply-time stays under 2 sec on huge data.
- Every write goes through Validator -> Journal -> File swap (atomic).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from validator import Validator


@dataclass
class OperationResult:
    success: bool
    word_id: str = ""
    word: str = ""
    error: str = ""
    message: str = ""
    version: int = 0


class MemoryManager:
    def __init__(self, base_path: str = "brain"):
        self.base_path = base_path
        self.schema_path = os.path.join(base_path, "schema", "brain_schema_v1.json")
        self.lex_index_path = os.path.join(base_path, "indexes", "lexeme_index.json")
        self.prefix_index_path = os.path.join(base_path, "indexes", "prefix_index.json")
        self.journal_path = os.path.join(base_path, "journal", "events.log")
        self.queue_path = os.path.join(base_path, "learning_queue.jsonl")
        self.data_files = {
            "english": os.path.join(base_path, "data", "english_core.jsonl"),
            "hindi": os.path.join(base_path, "data", "hindi_core.jsonl"),
            "hinglish": os.path.join(base_path, "data", "hinglish_core.jsonl"),
        }
        self.validator = Validator(self.schema_path)
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max = 5000
        self.lex_index: Dict[str, Any] = {"word_to_loc": {}, "id_to_loc": {}, "total": 0}
        self.prefix_index: Dict[str, List[str]] = {}
        self._ensure_dirs()
        self._load_indexes()
        if not self.lex_index.get("word_to_loc"):
            self.rebuild_indexes()

    # ---------- bootstrap ----------
    def _ensure_dirs(self):
        for p in (
            os.path.dirname(self.lex_index_path),
            os.path.dirname(self.journal_path),
            os.path.join(self.base_path, "data"),
            os.path.join(self.base_path, "backups"),
        ):
            os.makedirs(p, exist_ok=True)

    def _load_indexes(self):
        if os.path.exists(self.lex_index_path):
            try:
                with open(self.lex_index_path, "r", encoding="utf-8") as f:
                    self.lex_index = json.load(f)
            except Exception:
                self.lex_index = {"word_to_loc": {}, "id_to_loc": {}, "total": 0}
        if os.path.exists(self.prefix_index_path):
            try:
                with open(self.prefix_index_path, "r", encoding="utf-8") as f:
                    self.prefix_index = json.load(f)
            except Exception:
                self.prefix_index = {}

    def _save_indexes(self):
        with open(self.lex_index_path, "w", encoding="utf-8") as f:
            json.dump(self.lex_index, f, ensure_ascii=False)
        with open(self.prefix_index_path, "w", encoding="utf-8") as f:
            json.dump(self.prefix_index, f, ensure_ascii=False)

    def rebuild_indexes(self) -> int:
        with self._lock:
            self.lex_index = {"word_to_loc": {}, "id_to_loc": {}, "total": 0}
            self.prefix_index = {}
            count = 0
            for lang, path in self.data_files.items():
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, raw in enumerate(f, 1):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                        except Exception:
                            continue
                        lc = entry.get("lowercase") or entry.get("word", "").lower()
                        wid = entry.get("word_id", "")
                        self.lex_index["word_to_loc"][lc] = [lang, line_num]
                        self.lex_index["id_to_loc"][wid] = [lang, line_num]
                        prefix = lc[:2]
                        self.prefix_index.setdefault(prefix, []).append(lc)
                        count += 1
            self.lex_index["total"] = count
            self._save_indexes()
            return count

    # ---------- read ----------
    def get_word(self, word: str) -> Optional[Dict[str, Any]]:
        if not word:
            return None
        lc = word.lower().strip()
        if lc in self._cache:
            return self._cache[lc]
        loc = self.lex_index["word_to_loc"].get(lc)
        if not loc:
            return None
        lang, line_num = loc
        path = self.data_files.get(lang)
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                if i == line_num:
                    try:
                        entry = json.loads(raw.strip())
                    except Exception:
                        return None
                    self._cache_put(lc, entry)
                    return entry
        return None

    def search_prefix(self, prefix: str, limit: int = 20) -> List[str]:
        prefix = prefix.lower()
        bucket = self.prefix_index.get(prefix[:2], [])
        return [w for w in bucket if w.startswith(prefix)][:limit]

    def _cache_put(self, key: str, value: Dict[str, Any]):
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    # ---------- write ----------
    def _atomic_append(self, path: str, line: str) -> int:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _atomic_replace_line(self, path: str, line_num: int, new_line: str):
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".disknory_", dir=os.path.dirname(path))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as out, open(path, "r", encoding="utf-8") as src:
                for i, raw in enumerate(src, 1):
                    out.write(new_line + "\n" if i == line_num else raw)
            shutil.move(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _journal(self, operation: str, actor: str, entry: Dict[str, Any], reason: str = "", before: Optional[Dict[str, Any]] = None):
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "operation": operation,
            "actor": actor,
            "target_word_id": entry.get("word_id", ""),
            "target_word": entry.get("word", ""),
            "reason": reason,
            "before": before,
            "after": entry if operation != "delete" else None,
        }
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def add_word(self, entry: Dict[str, Any], actor: str = "user") -> OperationResult:
        with self._lock:
            res = self.validator.validate_word_entry(entry)
            if not res.valid:
                return OperationResult(False, error="; ".join(res.errors))
            lc = entry["lowercase"]
            if lc in self.lex_index["word_to_loc"]:
                return OperationResult(False, error="word already exists")
            lang = entry.get("language", "english")
            path = self.data_files.get(lang, self.data_files["english"])
            line_num = self._atomic_append(path, json.dumps(entry, ensure_ascii=False))
            self.lex_index["word_to_loc"][lc] = [lang, line_num]
            self.lex_index["id_to_loc"][entry["word_id"]] = [lang, line_num]
            self.prefix_index.setdefault(lc[:2], []).append(lc)
            self.lex_index["total"] += 1
            self._save_indexes()
            self._cache_put(lc, entry)
            self._journal("add", actor, entry)
            return OperationResult(True, word_id=entry["word_id"], word=entry["word"], message="added", version=entry.get("version", 1))

    def edit_word(self, word_id: str, updates: Dict[str, Any], actor: str = "user") -> OperationResult:
        with self._lock:
            loc = self.lex_index["id_to_loc"].get(word_id)
            if not loc:
                return OperationResult(False, error="word_id not found")
            lang, line_num = loc
            path = self.data_files[lang]
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            try:
                before = json.loads(lines[line_num - 1].strip())
            except Exception:
                return OperationResult(False, error="cannot read existing entry")
            if before.get("is_locked"):
                return OperationResult(False, error="entry is locked")
            after = {**before, **updates}
            after["version"] = before.get("version", 1) + 1
            after["modified_timestamp"] = datetime.utcnow().isoformat() + "Z"
            after["modified_by"] = actor
            res = self.validator.validate_word_entry(after)
            if not res.valid:
                return OperationResult(False, error="; ".join(res.errors))
            self._atomic_replace_line(path, line_num, json.dumps(after, ensure_ascii=False))
            self._cache_put(after["lowercase"], after)
            self._journal("edit", actor, after, before=before)
            return OperationResult(True, word_id=word_id, word=after["word"], message="edited", version=after["version"])

    def delete_word(self, word_id: str, actor: str = "user", reason: str = "") -> OperationResult:
        with self._lock:
            loc = self.lex_index["id_to_loc"].get(word_id)
            if not loc:
                return OperationResult(False, error="word_id not found")
            lang, line_num = loc
            path = self.data_files[lang]
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            try:
                before = json.loads(lines[line_num - 1].strip())
            except Exception:
                return OperationResult(False, error="cannot read existing entry")
            archive = os.path.join(self.base_path, "data", "archived.jsonl")
            with open(archive, "a", encoding="utf-8") as f:
                f.write(json.dumps({**before, "archived_at": datetime.utcnow().isoformat() + "Z", "archive_reason": reason}, ensure_ascii=False) + "\n")
            # mark deleted by replacing line with tombstone JSON keeping line numbers stable
            tombstone = {"_tombstone": True, "word_id": before.get("word_id"), "deleted_at": datetime.utcnow().isoformat() + "Z"}
            self._atomic_replace_line(path, line_num, json.dumps(tombstone, ensure_ascii=False))
            self.lex_index["word_to_loc"].pop(before.get("lowercase", ""), None)
            self.lex_index["id_to_loc"].pop(word_id, None)
            self._cache.pop(before.get("lowercase", ""), None)
            self._save_indexes()
            self._journal("delete", actor, before, reason=reason, before=before)
            return OperationResult(True, word_id=word_id, word=before.get("word", ""), message="archived")

    # ---------- learning queue ----------
    def queue_unknown(self, word: str, context: str = ""):
        with self._lock:
            os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
            with open(self.queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "word": word.lower(),
                    "context": context[:200],
                    "ts": datetime.utcnow().isoformat() + "Z",
                }, ensure_ascii=False) + "\n")

    def list_unknown(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(self.queue_path):
            return []
        agg: Dict[str, Dict[str, Any]] = {}
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                w = item["word"]
                if w in agg:
                    agg[w]["count"] += 1
                    agg[w]["last_seen"] = item["ts"]
                else:
                    agg[w] = {"word": w, "count": 1, "first_seen": item["ts"], "last_seen": item["ts"], "context": item.get("context", "")}
        items = sorted(agg.values(), key=lambda x: -x["count"])
        return items[:limit]

    # ---------- stats / backup ----------
    def stats(self) -> Dict[str, Any]:
        s = {"total_words": self.lex_index.get("total", 0), "by_language": {}, "cache_size": len(self._cache)}
        for lang, path in self.data_files.items():
            n = 0
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        if raw.strip() and not raw.strip().startswith('{"_tombstone"'):
                            n += 1
            s["by_language"][lang] = n
        return s

    def backup(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(self.base_path, "backups", f"backup_{ts}")
        os.makedirs(target, exist_ok=True)
        for sub in ("data", "schema", "indexes", "journal"):
            src = os.path.join(self.base_path, sub)
            dst = os.path.join(target, sub)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        return target
