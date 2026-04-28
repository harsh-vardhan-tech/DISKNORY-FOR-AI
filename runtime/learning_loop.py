#!/usr/bin/env python3
"""DISKNORY-FOR-AI - Self-learning loop.

When AI sees an unknown word, it goes to learning_queue.jsonl.
User can teach via `learn <word> | <hindi> | <english> | <example>`.
AI also self-infers from related/co-occurring words (basic heuristic).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory_manager import MemoryManager, OperationResult


def _has_devanagari(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", s))


def make_word_entry(
    word: str,
    hindi_meaning: str = "",
    english_meaning: str = "",
    example_en: str = "",
    example_hi: str = "",
    part_of_speech: str = "noun",
    learned_from: str = "user",
    rank: int = 0,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    if language is None:
        if _has_devanagari(word):
            language = "hindi"
        elif re.fullmatch(r"[a-zA-Z\-']+", word or ""):
            language = "english"
        else:
            language = "hinglish"
    prefix = {"english": "ENG", "hindi": "HIN", "hinglish": "HNG"}[language]
    word_id = f"{prefix}_{rank:06d}_{word.lower()}"
    now = datetime.utcnow().isoformat() + "Z"
    pos = part_of_speech.lower()
    flags = {f"is_{k}": False for k in [
        "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
        "conjunction", "interjection", "common", "technical", "slang",
        "formal", "informal", "question_word", "emotion", "time_related",
        "place_related", "person_related", "number", "color", "action", "abstract",
    ]}
    if pos in flags or pos in {"noun", "verb", "adjective", "adverb"}:
        flags[f"is_{pos}"] = True
    flags["is_common"] = True
    flags["is_informal"] = True
    return {
        "word_id": word_id,
        "word": word,
        "lowercase": word.lower(),
        "letters": list(word),
        "letter_count": len(word),
        "syllables": max(1, len(re.findall(r"[aeiouAEIOU]", word)) or 1),
        "phonetic_ipa": "",
        "phonetic_simple": word.lower(),
        "language": language,
        "part_of_speech": pos,
        "grammar_role": "",
        "root_word": word.lower(),
        "word_family": [word.lower()],
        "plural_form": None,
        "tense_forms": None,
        "comparative": None,
        "superlative": None,
        "hindi_meaning": hindi_meaning,
        "english_meaning": english_meaning,
        "definition_simple": english_meaning or hindi_meaning,
        "definition_detailed": "",
        "synonyms": [],
        "antonyms": [],
        "example_sentence_en": example_en,
        "example_sentence_hi": example_hi,
        "usage_context": ["informal"],
        "common_phrases": [],
        "alternative_meanings": [],
        "related_concepts": [],
        "emotion_weight": 0.0,
        "intent_type": "",
        "response_priority": "medium",
        "topic_domain": "general",
        "semantic_tags": [],
        "sentence_position_role": "",
        "confidence_score": 0.85 if learned_from == "dictionary" else 0.6,
        "learned_from": learned_from,
        "learned_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "times_used": 0,
        "last_used": None,
        "user_corrections": [],
        "created_by": "system" if learned_from == "dictionary" else "user",
        "created_timestamp": now,
        "modified_by": "system",
        "modified_timestamp": now,
        "version": 1,
        "is_locked": False,
        **flags,
    }


class LearningLoop:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def teach(self, word: str, hindi: str = "", english: str = "", example: str = "", pos: str = "noun") -> OperationResult:
        rank = self.memory.lex_index.get("total", 0) + 1
        entry = make_word_entry(word, hindi, english, example, "", pos, learned_from="user", rank=rank)
        return self.memory.add_word(entry, actor="user")

    def correct(self, word_id: str, hindi: str = None, english: str = None, example: str = None) -> OperationResult:
        updates: Dict[str, Any] = {}
        if hindi is not None:
            updates["hindi_meaning"] = hindi
        if english is not None:
            updates["english_meaning"] = english
        if example is not None:
            updates["example_sentence_en"] = example
        return self.memory.edit_word(word_id, updates, actor="user")

    def reinforce(self, word: str):
        entry = self.memory.get_word(word)
        if not entry:
            return
        new_score = min(1.0, float(entry.get("confidence_score", 0.5)) + 0.02)
        self.memory.edit_word(entry["word_id"], {
            "confidence_score": new_score,
            "times_used": entry.get("times_used", 0) + 1,
            "last_used": datetime.utcnow().isoformat() + "Z",
        }, actor="ai")
