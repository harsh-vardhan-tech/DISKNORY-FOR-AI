#!/usr/bin/env python3
"""DISKNORY-FOR-AI - Natural Chat Engine.

Talks like a human friend (no commands needed).
- Detects language: english / hindi / hinglish
- Looks up known words in brain
- Asks the user when it sees an unknown word
- Remembers user name + last topic within a session
- Generates varied replies so it never sounds robotic
- Auto-queues unknown words to learning_queue.jsonl
"""
from __future__ import annotations

import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory_manager import MemoryManager


@dataclass
class ChatResponse:
    reply_text: str
    reply_type: str = "chat"
    confidence: float = 0.7
    language: str = "english"
    source_words: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    learning_triggered: bool = False
    elapsed_ms: float = 0.0


GREETINGS_IN = {
    "hi", "hello", "hey", "hii", "hiii", "namaste", "namaskar", "salam",
    "yo", "oye", "sup", "helo", "helo",
}
FAREWELL_IN = {"bye", "byee", "goodbye", "alvida", "tata", "ttyl", "cya"}
QWORDS = {
    "what", "why", "how", "when", "where", "who", "which",
    "kya", "kaise", "kyu", "kyun", "kahan", "kahaan", "kab", "kaun", "konsa", "kitna",
}
MEAN_TRIGGERS = {"matlab", "meaning", "mtlb", "arth", "definition", "define", "bata", "explain"}
NAME_PATTERNS = [
    re.compile(r"\bmera\s+naam\s+([A-Za-z\u0900-\u097F]+)", re.IGNORECASE),
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z\u0900-\u097F]+)", re.IGNORECASE),
    re.compile(r"\bi\s*am\s+([A-Za-z\u0900-\u097F]+)", re.IGNORECASE),
    re.compile(r"\bmain\s+([A-Za-z\u0900-\u097F]+)\s+hu\b", re.IGNORECASE),
    re.compile(r"\bmujhe\s+([A-Za-z\u0900-\u097F]+)\s+bolo\b", re.IGNORECASE),
]
POSITIVE_WORDS = {
    "good", "great", "love", "happy", "awesome", "thanks", "thank", "nice",
    "acha", "achha", "badhiya", "mast", "khush", "shukriya", "wah", "wow",
}
NEGATIVE_WORDS = {
    "bad", "sad", "hate", "angry", "worse", "terrible", "pain", "tired",
    "udas", "bura", "dukh", "thaka", "thak", "bura",
}

# Stop-words that should not be treated as unknown words
STOP_WORDS = {
    "the", "a", "an", "is", "are", "am", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "our", "their", "this", "that", "these", "those", "and", "or",
    "but", "so", "if", "in", "on", "at", "to", "for", "of", "with",
    "not", "no", "yes", "ok", "okay", "about", "tell", "from", "by",
    "up", "out", "as", "into", "through", "then", "than", "too",
    "meaning", "means", "mean", "define", "definition",
    "kya", "hai", "nahi", "haan", "tu", "tum", "main", "mera", "tera",
    "yaar", "bhai", "acha", "kar", "ho", "raha", "rahi", "hu", "hun",
    "ka", "ki", "ke", "se", "me", "ne", "ko", "hota", "hoti",
}


class ChatEngine:
    MAX_HISTORY = 50
    CACHE_LIMIT = 200

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self._reply_cache: "OrderedDict[str, str]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, message: str) -> ChatResponse:
        """Alias kept for backward compatibility."""
        return self.process_message(message)

    def process_message(self, message: str) -> ChatResponse:
        t0 = time.time()
        msg = (message or "").strip()
        if not msg:
            return ChatResponse(reply_text="Haan bol, sun raha hu.", elapsed_ms=0.0)

        language = self._detect_language(msg)
        self._capture_name(msg)
        tokens = self._tokenize(msg)
        known, unknown = self._lookup_tokens(tokens)
        intent = self._detect_intent(msg, tokens)
        sentiment = self._detect_sentiment(tokens)

        # Self-learning: queue every unknown word automatically
        for w in unknown:
            self.memory.queue_unknown(w, context=msg)

        reply = self._compose(msg, intent, language, sentiment, known, unknown, tokens)

        elapsed_ms = (time.time() - t0) * 1000.0
        resp = ChatResponse(
            reply_text=reply,
            reply_type=intent,
            confidence=0.85 if known else 0.45,
            language=language,
            source_words=[e["word"] for e in known],
            unknown_words=unknown,
            learning_triggered=bool(unknown),
            elapsed_ms=elapsed_ms,
        )
        self._remember(msg, reply, intent, language)
        # Reinforce used words (boost confidence score slightly)
        for entry in known[:5]:
            self._reinforce(entry)
        return resp

    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.history[-limit:]

    def clear_context(self):
        self.context.clear()

    def clear(self):
        self.clear_context()

    # ------------------------------------------------------------------ #
    # Language & tokenisation
    # ------------------------------------------------------------------ #
    def _detect_language(self, text: str) -> str:
        hindi_chars = re.findall(r"[\u0900-\u097F]", text)
        latin_chars = re.findall(r"[A-Za-z]", text)
        if hindi_chars and latin_chars:
            return "hinglish"
        if hindi_chars:
            return "hindi"
        # Roman-Hindi heuristic
        low = text.lower()
        roman_hits = sum(
            1 for w in ["hai", "kya", "kaise", "tu", "tum", "main",
                         "yaar", "bhai", "acha", "nahi", "haan",
                         "kar", "ho", "raha", "rahi", "hu", "hun"]
            if re.search(rf"\b{w}\b", low)
        )
        return "hinglish" if roman_hits >= 1 else "english"

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s\u0900-\u097F']", " ", text)
        return [t for t in cleaned.lower().split() if t]

    def _lookup_tokens(self, tokens: List[str]):
        known: List[Dict[str, Any]] = []
        unknown: List[str] = []
        seen: set = set()
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            entry = self.memory.get_word(tok)
            if entry:
                known.append(entry)
            elif (
                len(tok) > 2
                and tok not in QWORDS
                and tok not in GREETINGS_IN
                and tok not in FAREWELL_IN
                and tok not in STOP_WORDS
            ):
                unknown.append(tok)
        return known, unknown

    # ------------------------------------------------------------------ #
    # Intent & sentiment
    # ------------------------------------------------------------------ #
    def _detect_intent(self, msg: str, tokens: List[str]) -> str:
        toks = set(tokens)
        if toks & GREETINGS_IN:
            return "greeting"
        if toks & FAREWELL_IN:
            return "farewell"
        if toks & MEAN_TRIGGERS:
            return "ask_meaning"
        if msg.strip().endswith("?") or (toks & QWORDS):
            return "question"
        return "statement"

    def _detect_sentiment(self, tokens: List[str]) -> str:
        score = sum(1 for t in tokens if t in POSITIVE_WORDS) \
              - sum(1 for t in tokens if t in NEGATIVE_WORDS)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # Words too common to be someone's name
    _NOT_A_NAME = {
        "feeling", "doing", "happy", "sad", "good", "bad", "fine", "okay", "ok",
        "tired", "hungry", "ready", "bored", "excited", "student", "teacher",
        "here", "there", "going", "coming", "trying", "learning", "working",
        "acha", "theek", "thakaa", "taiyaar",
    }

    def _capture_name(self, msg: str):
        for pat in NAME_PATTERNS:
            m = pat.search(msg)
            if m:
                name = m.group(1).strip()
                if name and len(name) > 1 and len(name) <= 20 and name.lower() not in self._NOT_A_NAME:
                    self.context["user_name"] = name.title()
                    return

    # ------------------------------------------------------------------ #
    # Reply composition
    # ------------------------------------------------------------------ #
    def _name_suffix(self) -> str:
        n = self.context.get("user_name")
        return f" {n}" if n else ""

    @staticmethod
    def _pick(items: List[str]) -> str:
        return random.choice(items)

    def _compose(
        self,
        msg: str,
        intent: str,
        lang: str,
        sentiment: str,
        known: List[Dict[str, Any]],
        unknown: List[str],
        tokens: List[str],
    ) -> str:
        nm = self._name_suffix()

        # --- Greeting ---
        if intent == "greeting":
            opts = {
                "english": [
                    f"Hey{nm}! How are you?",
                    f"Hi{nm}, how's it going?",
                    f"Hello{nm}! What's up?",
                ],
                "hindi": [
                    f"नमस्ते{nm}! कैसे हो?",
                    f"हेलो{nm}! क्या हाल है?",
                    f"अरे{nm}! क्या चल रहा है?",
                ],
                "hinglish": [
                    f"Hey{nm}! Kaise hai?",
                    f"Hello{nm} bhai, kya scene hai?",
                    f"Arre{nm}! Sab badhiya?",
                ],
            }
            return self._pick(opts.get(lang, opts["hinglish"]))

        # --- Farewell ---
        if intent == "farewell":
            opts = {
                "english": [f"Bye{nm}! Take care.", f"See you{nm}!", "Catch you later!"],
                "hindi": [f"अलविदा{nm}! ध्यान रखना।", "फिर मिलेंगे!"],
                "hinglish": [f"Bye{nm}! Khayal rakhna.", "Phir milte hain!", "Tata!"],
            }
            return self._pick(opts.get(lang, opts["hinglish"]))

        # --- Meaning / Question ---
        if intent in ("ask_meaning", "question"):
            target = self._pick_target(tokens, known)
            if target:
                return self._explain(target, lang)
            return self._ask_to_teach(msg, unknown, lang)

        # --- Statement ---
        if known:
            content = self._best_content_word(known)
            if content:
                return self._statement_reply(content, sentiment, lang, unknown)
            # All known words are function words; treat unknowns or give generic reply
            if unknown:
                return self._ask_to_teach(msg, unknown, lang)
            fallback_s = {
                "english": [f"Tell me more{nm}.", "Go on, I'm listening."],
                "hindi": [f"बताओ{nm}, सुन रहा हूँ।"],
                "hinglish": [f"Bata{nm}, sun raha hu.", "Aur bol yaar."],
            }
            return self._pick(fallback_s.get(lang, fallback_s["hinglish"]))

        if unknown:
            return self._ask_to_teach(msg, unknown, lang)

        # Fallback
        fallback = {
            "english": [f"Tell me more{nm}.", "Interesting, go on...", "Hmm, and then?"],
            "hindi": [f"और बताओ{nm}।", "रोचक है, आगे बताओ...", "अच्छा, फिर?"],
            "hinglish": [f"Aur bata{nm}.", "Acha, phir kya hua?", "Hmm interesting, aage bol."],
        }
        return self._pick(fallback.get(lang, fallback["hinglish"]))

    # Function / grammar words that are poor "topic" targets in meaning questions
    _FUNCTION_WORDS = {
        "is", "am", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had",
        "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
        "this", "that", "these", "those",
        "ka", "ki", "ke", "kya", "hai", "hain", "tha", "thi",
        "mean", "means", "meaning", "define",
    }

    def _pick_target(self, tokens: List[str], known: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        skip = QWORDS | MEAN_TRIGGERS | self._FUNCTION_WORDS
        # Pattern: "meaning of X" / "matlab X" → target is the word right after the trigger
        for i, t in enumerate(tokens):
            if t in MEAN_TRIGGERS and i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                entry = self.memory.get_word(next_tok)
                if entry and next_tok not in skip:
                    return entry
        # General: first non-skip content word that is known
        for t in tokens:
            if t in skip:
                continue
            entry = self.memory.get_word(t)
            if entry:
                return entry
        # Fall back: first known word that isn't a function/question word
        for entry in known:
            w = entry.get("word", "").lower()
            if w not in self._FUNCTION_WORDS and w not in QWORDS:
                return entry
        # All known words are function/grammar words — return None so caller asks user
        return None

    def _best_content_word(self, known: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the first known entry that is a content word (not a function word)."""
        skip = self._FUNCTION_WORDS | QWORDS | STOP_WORDS
        for entry in known:
            if entry.get("word", "").lower() not in skip:
                return entry
        return None

    def _explain(self, entry: Dict[str, Any], lang: str) -> str:
        word = entry.get("word", "")
        en = entry.get("english_meaning", "") or entry.get("definition_simple", "")
        hi = entry.get("hindi_meaning", "")
        ex = entry.get("example_sentence_en", "")

        if lang == "hindi":
            base = f"'{word}' का मतलब है: {hi or en}."
            if ex:
                base += f" उदाहरण: {ex}"
            base += " और कुछ पूछना है?"
            return base

        if lang == "hinglish":
            base = f"'{word}' ka matlab hai: {hi}"
            if en and en != hi:
                base += f" / {en}"
            if ex:
                base += f". Example: {ex}"
            base += ". Aur kuch?"
            return base

        # English
        base = f"'{word}' means: {en}"
        if hi:
            base += f" (Hindi: {hi})"
        if ex:
            base += f". Example: {ex}"
        base += ". Anything else?"
        return base

    def _ask_to_teach(self, msg: str, unknown: List[str], lang: str) -> str:
        nm = self._name_suffix()
        if unknown:
            w = unknown[0]
            opts = {
                "english": [
                    f"Hmm{nm}, I don't know '{w}' yet. Can you tell me what it means?",
                    f"'{w}' is new to me{nm}. What does it mean?",
                    f"I haven't learned '{w}' yet. Teach me{nm}?",
                ],
                "hindi": [
                    f"मुझे '{w}' नहीं पता{nm}. क्या तुम बताओगे इसका मतलब?",
                    f"'{w}' नया शब्द है मेरे लिए{nm}. समझाओगे?",
                ],
                "hinglish": [
                    f"Yaar{nm}, '{w}' mujhe nahi pata. Iska matlab kya hai?",
                    f"'{w}' new word hai mere liye{nm}. Tu sikhayega?",
                    f"Bhai{nm}, '{w}' ka matlab bata na, main yaad rakh lunga.",
                ],
            }
            return self._pick(opts.get(lang, opts["hinglish"]))

        # Generic don't-know
        opts = {
            "english": [
                f"I'm not sure{nm}. Can you explain a bit more?",
                "Hmm, I need a little more context.",
            ],
            "hindi": [f"मुझे ठीक से समझ नहीं आया{nm}. थोड़ा और बताओ।"],
            "hinglish": [
                f"Pata nahi yaar{nm}, thoda aur detail de.",
                "Hmm, samjha nahi — phir se bata?",
            ],
        }
        return self._pick(opts.get(lang, opts["hinglish"]))

    def _statement_reply(
        self,
        topic: Dict[str, Any],
        sentiment: str,
        lang: str,
        unknown: List[str],
    ) -> str:
        word = topic.get("word", "")
        nm = self._name_suffix()
        self.context["last_topic"] = word

        if sentiment == "positive":
            opts = {
                "english": [
                    f"Nice{nm}! Tell me more about {word}.",
                    f"That's great! What else about {word}?",
                ],
                "hindi": [
                    f"वाह{nm}! {word} के बारे में और बताओ।",
                    f"बढ़िया! फिर {word} के बारे में क्या?",
                ],
                "hinglish": [
                    f"Wah{nm}! {word} ke baare me aur bata.",
                    f"Mast{nm}! Phir {word} ka kya scene?",
                ],
            }
        elif sentiment == "negative":
            opts = {
                "english": [
                    f"Oh{nm}, that sounds tough. What happened with {word}?",
                    "I'm sorry to hear that. Want to talk about it?",
                ],
                "hindi": [
                    f"अरे{nm}, मुश्किल लग रहा है। {word} के साथ क्या हुआ?",
                    f"समझा{nm}, बात करना चाहो तो बताओ।",
                ],
                "hinglish": [
                    f"Arre{nm}, tough lag raha hai. {word} me kya hua?",
                    f"Samjha{nm}. Tension mat le, baat kar.",
                ],
            }
        else:
            opts = {
                "english": [
                    f"Got it{nm}, you mentioned '{word}'. Tell me more.",
                    f"Okay, '{word}' — what about it?",
                    f"Interesting{nm}, can you say more about {word}?",
                ],
                "hindi": [
                    f"समझा{nm}, '{word}' की बात है। और बताओ।",
                    f"अच्छा, {word} — क्या हुआ इसके साथ?",
                ],
                "hinglish": [
                    f"Samjha{nm}, '{word}' ki baat hai. Aur bata.",
                    f"Acha, {word} — iske baare me kya?",
                    f"Got it{nm}, '{word}' — phir kya hua?",
                ],
            }

        text = self._pick(opts.get(lang, opts["hinglish"]))
        if unknown:
            text += f" (BTW '{unknown[0]}' new word lag raha mujhe — baad me sikhana.)"
        return text

    # ------------------------------------------------------------------ #
    # Memory bookkeeping
    # ------------------------------------------------------------------ #
    def _reinforce(self, entry: Dict[str, Any]):
        wid = entry.get("word_id")
        if not wid:
            return
        try:
            new_score = min(1.0, float(entry.get("confidence_score", 0.5)) + 0.01)
            self.memory.edit_word(wid, {
                "confidence_score": new_score,
                "times_used": int(entry.get("times_used", 0)) + 1,
                "last_used": datetime.utcnow().isoformat() + "Z",
            }, actor="ai")
        except Exception:
            pass

    def _remember(self, user_msg: str, ai_msg: str, intent: str, lang: str):
        self.history.append({
            "user": user_msg,
            "ai": ai_msg,
            "intent": intent,
            "lang": lang,
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
