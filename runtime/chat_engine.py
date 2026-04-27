"""DISKNORY chat engine.

Conversation layer that parses user input, detects intent, retrieves word
knowledge from brain storage, builds short replies, and triggers learning.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import re
import random
from typing import Any, Dict, List, Optional

try:
    from runtime.memory_manager import MemoryManager
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from memory_manager import MemoryManager  # type: ignore


@dataclass
class Token:
    word: str
    lowercase: str
    position: int
    is_known: bool
    entry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedMessage:
    original: str
    words: List[str]
    tokens: List[Token]
    sentence_type: str
    language: str
    punctuation: List[str]
    word_count: int


@dataclass
class UnderstoodMessage:
    intent: str
    confidence: float
    main_subject: str
    main_action: str
    entities: List[Dict[str, Any]]
    sentiment: str
    unknown_words: List[str]
    known_words: List[str]
    context_needed: bool
    suggested_reply_type: str


@dataclass
class ChatResponse:
    reply_text: str
    reply_type: str
    confidence: float
    source_words: List[str]
    language: str
    includes_hindi: bool
    includes_examples: bool
    follow_up_suggested: bool
    learning_triggered: bool
    timestamp: str


class ChatEngine:
    """Conversation brain that connects parser, understanding and reply logic."""

    MAX_HISTORY = 10
    RESPONSE_CACHE_LIMIT = 100

    def __init__(self, memory_manager: MemoryManager):
        self.mm = memory_manager
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        self.unknown_words: List[str] = []
        self.response_cache: "OrderedDict[str, ChatResponse]" = OrderedDict()

    def _now_iso(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def process_message(self, user_message: str) -> ChatResponse:
        """Process user message end-to-end and return chat response."""
        message_key = user_message.strip().lower()
        cached = self._cache_get(message_key)
        if cached is not None:
            self._save_to_history(user_message, cached)
            return cached

        parsed = self._parse_message(user_message)
        understood = self._understand_message(parsed)
        reply = self._build_reply(understood)
        self._learn_from_interaction(parsed, understood, reply)
        self._save_to_history(user_message, reply)
        self._cache_set(message_key, reply)
        return reply

    def _parse_message(self, message: str) -> ParsedMessage:
        """Parse raw user message into structured tokens and metadata."""
        punctuation = re.findall(r"[?!.,;:]", message)
        words = re.findall(r"[\w']+", message, flags=re.UNICODE)

        if any(ch in message for ch in "?？"):
            sentence_type = "question"
        elif any(ch in message for ch in "!！"):
            sentence_type = "exclamation"
        elif words and words[0].lower() in {"please", "show", "tell", "define", "explain", "give"}:
            sentence_type = "command"
        else:
            sentence_type = "statement"

        language = self._detect_language(message)
        tokens: List[Token] = []
        for i, word in enumerate(words):
            lw = word.lower()
            entry = self.mm.get_word(lw)
            tokens.append(Token(word=word, lowercase=lw, position=i, is_known=bool(entry), entry=entry))

        return ParsedMessage(
            original=message,
            words=words,
            tokens=tokens,
            sentence_type=sentence_type,
            language=language,
            punctuation=punctuation,
            word_count=len(words),
        )

    def _detect_language(self, text: str) -> str:
        """Best-effort language detection for english/hindi/hinglish."""
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        ascii_words = re.findall(r"[A-Za-z]+", text)

        if has_devanagari and ascii_words:
            return "hinglish"
        if has_devanagari:
            return "hindi"
        return "english"

    def _understand_message(self, parsed: ParsedMessage) -> UnderstoodMessage:
        """Understand message intent, entities, sentiment and known/unknown words."""
        known_words: List[str] = []
        unknown_words: List[str] = []
        entities: List[Dict[str, Any]] = []

        for token in parsed.tokens:
            if token.is_known:
                known_words.append(token.lowercase)
                entry = token.entry
                entities.append(
                    {
                        "word": token.lowercase,
                        "part_of_speech": entry.get("part_of_speech"),
                        "is_noun": entry.get("is_noun", False),
                        "is_verb": entry.get("is_verb", False),
                    }
                )
            else:
                unknown_words.append(token.lowercase)

        intent = self._detect_intent(parsed)
        sentiment = self._detect_sentiment(parsed.words)

        main_subject = ""
        main_action = ""
        for ent in entities:
            if not main_subject and ent.get("is_noun"):
                main_subject = ent["word"]
            if not main_action and ent.get("is_verb"):
                main_action = ent["word"]

        if not main_subject and known_words:
            main_subject = known_words[0]

        confidence = 0.3
        if parsed.word_count > 0:
            confidence = min(1.0, max(0.1, len(known_words) / parsed.word_count))

        context_needed = intent in {"question", "unknown"} and len(known_words) <= 1
        suggested_reply_type = {
            "greeting": "greeting",
            "question": "answer",
            "statement": "acknowledgement",
            "command": "confirmation",
            "farewell": "farewell",
            "unknown": "clarification",
        }[intent]

        return UnderstoodMessage(
            intent=intent,
            confidence=round(confidence, 3),
            main_subject=main_subject,
            main_action=main_action,
            entities=entities,
            sentiment=sentiment,
            unknown_words=unknown_words,
            known_words=known_words,
            context_needed=context_needed,
            suggested_reply_type=suggested_reply_type,
        )

    def _detect_intent(self, parsed: ParsedMessage) -> str:
        """Detect high-level intent from message shape and keywords."""
        words = {w.lower() for w in parsed.words}
        joined = parsed.original.lower()

        if words & {"hello", "hi", "hey", "namaste", "hola"}:
            return "greeting"
        if words & {"bye", "goodbye", "farewell", "see", "later"} and "bye" in words:
            return "farewell"
        if parsed.sentence_type == "question" or words & {"what", "why", "how", "when", "where", "who", "which"}:
            return "question"
        if parsed.sentence_type == "command" or words & {"tell", "define", "show", "explain", "give"}:
            return "command"
        if joined.strip():
            return "statement"
        return "unknown"

    def _detect_sentiment(self, words: List[str]) -> str:
        """Tiny rule-based sentiment detector."""
        positive = {"good", "great", "love", "happy", "awesome", "thanks", "thank"}
        negative = {"bad", "sad", "hate", "angry", "worse", "terrible", "pain"}
        score = 0
        for word in words:
            lw = word.lower()
            if lw in positive:
                score += 1
            if lw in negative:
                score -= 1
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def _build_reply(self, understood: UnderstoodMessage) -> ChatResponse:
        """Build reply text from interpreted message state with natural conversation flow."""
        reply_type = understood.suggested_reply_type
        includes_hindi = False
        includes_examples = False
        follow_up = False

        if understood.intent == "greeting":
            text = self._compose_greeting(understood)

        elif understood.intent == "farewell":
            text = random.choice([
                "Goodbye! Take care 👋",
                "Phir milenge! Bye 👋",
                "See you later! 🎉",
            ])

        elif understood.intent == "question":
            text, includes_hindi, includes_examples, follow_up = self._compose_question(understood)

        elif understood.intent == "command":
            if understood.main_action:
                text = f"Got it. I understood your request related to '{understood.main_action}'."
            else:
                text = "Sure, I can do that. Please share a little more detail."
                follow_up = True

        elif understood.intent == "statement":
            text = self._compose_statement(understood, includes_hindi)

        else:
            text = self._compose_default(understood)
            follow_up = True

        if understood.unknown_words:
            unknown_preview = ", ".join(understood.unknown_words[:3])
            text += f" I also found unknown words: {unknown_preview}."
            follow_up = True

        # Keep quick response target
        if len(text) > 500:
            text = text[:497] + "..."

        return ChatResponse(
            reply_text=text,
            reply_type=reply_type,
            confidence=max(0.2, understood.confidence),
            source_words=understood.known_words + understood.unknown_words,
            language="hinglish" if includes_hindi else understood.language,
            includes_hindi=includes_hindi,
            includes_examples=includes_examples,
            follow_up_suggested=follow_up,
            learning_triggered=bool(understood.unknown_words),
            timestamp=self._now_iso(),
        )

    def _compose_greeting(self, understood: UnderstoodMessage) -> str:
        """Natural conversation greeting responses."""
        name = self.get_context("user_name")
        
        greetings = {
            "english": [
                f"Hey! Kaise hai tu?",
                f"Hello! Kya haal chaal?",
                f"Hi! Aaj kya plan hai?",
                f"Hey there! What's up?",
                f"Hello! How can I help?",
            ],
            "hindi": [
                "नमस्ते! कैसे हो भाई?",
                "अरे! क्या हाल है?",
                "हेलो! आज क्या चल रहा है?",
                "नमस्कार! कैसे हो?",
                "सुनो! क्या चल रहा है?",
            ],
            "hinglish": [
                "Hey bhai! Kaise hai?",
                "Hello dost! Kya scene hai?",
                "Hi yaar! Sab badhiya?",
                "Namaste! Kaisa jaa raha hai?",
                "Kya chal raha hai bhai?",
            ]
        }
        
        text = random.choice(greetings.get(understood.language, greetings["english"]))
        
        if name:
            text = f"{text} {name}!"
        
        return text

    def _compose_question(self, understood: UnderstoodMessage) -> tuple:
        """Natural conversation question responses."""
        includes_hindi = False
        includes_examples = False
        follow_up = False
        
        if understood.main_subject:
            entry = self.mm.get_word(understood.main_subject)
            if entry:
                word = entry.get("word", understood.main_subject)
                meaning = entry.get("english_meaning", "")
                h_meaning = entry.get("hindi_meaning", "")
                example = entry.get("example_sentence_en", "")
                
                if understood.language == "hindi":
                    text = f"{word} ka matlab hota hai: {h_meaning or meaning}."
                    if example:
                        text += f" Udaharan: {example}"
                        includes_examples = True
                elif understood.language == "hinglish":
                    text = f"'{word}' means {meaning}"
                    if h_meaning:
                        text += f" (Hindi: {h_meaning})"
                        includes_hindi = True
                    if example:
                        text += f". Example: {example}"
                        includes_examples = True
                    text += ". Aur kuch jaanna hai?"
                else:
                    text = f"'{word}' means: {meaning}"
                    if h_meaning:
                        text += f" | Hindi: {h_meaning}"
                        includes_hindi = True
                    if example:
                        text += f". Example: {example}"
                        includes_examples = True
            else:
                if understood.language == "hindi":
                    text = f"Mujhe '{understood.main_subject}' ke baare mein pata nahi. Tu samjha sakta hai?"
                elif understood.language == "hinglish":
                    text = f"'{understood.main_subject}' mere paas nahi hai. Kya tu sikhayega?"
                else:
                    text = f"I couldn't find '{understood.main_subject}' yet. Can you teach me its meaning?"
                follow_up = True
        else:
            if understood.language == "hindi":
                text = "Mujhe thoda context chahiye. Kya tu phir se puch sakta hai?"
            elif understood.language == "hinglish":
                text = "Mujhe thoda aur detail chahiye. Phir se puch na?"
            else:
                text = "I need a little more context to answer that. Can you rephrase?"
            follow_up = True
        
        return text, includes_hindi, includes_examples, follow_up

    def _compose_statement(self, understood: UnderstoodMessage, includes_hindi: bool) -> str:
        """Natural conversation statement responses."""
        if understood.sentiment == "positive":
            responses = {
                "english": [
                    "Nice! That's awesome! Want me to explain any word from that?",
                    "Great! I understand your point. Aur bataao?",
                    "Awesome! Can I help clarify something?",
                ],
                "hindi": [
                    "वाह! बहुत अच्छा! कुछ शब्द समझाऊँ?",
                    "बढ़िया! मुझे समझ आ गया। और बताओ?",
                    "शानदार! कुछ और है?",
                ],
                "hinglish": [
                    "Wah! Bahut badhiya! Kuch samjhaoo?",
                    "Nice bhai! Aur bata.",
                    "Awesome! Kya aur bataana hai?",
                ]
            }
        elif understood.sentiment == "negative":
            responses = {
                "english": [
                    "I hear you. That sounds tough. I can help break it down.",
                    "Samjha. Ye problem hai. Hum milke dekh sakte hain.",
                    "I understand. Let's work through it together.",
                ],
                "hindi": [
                    "समझा मैंने। यह मुश्किल है। मैं मदद कर सकता हूँ।",
                    "पता चला। ये समस्या है। हम मिलकर देख सकते हैं।",
                    "समझा! आइए इसे हल करते हैं।",
                ],
                "hinglish": [
                    "Samjha bhai. Ye tough hai. Help kar sakta hoon.",
                    "Pata chal gaya. Chalo milder dekh lete hain.",
                    "Samjha! Aate hain isko solve karein.",
                ]
            }
        else:
            responses = {
                "english": [
                    "Understood! I can explain meanings or give examples from that.",
                    "Got it! Want me to break down any word?",
                    "Samjh gaya! Aur kya jaanna hai?",
                ],
                "hindi": [
                    "समझा! क्या कोई शब्द समझाऊँ?",
                    "पता चल गया! और क्या चाहिए?",
                    "ठीक है! कुछ और पूछना है?",
                ],
                "hinglish": [
                    "Samjh gaya! Kya word samjhao?",
                    "Got it! Aur kya chahiye?",
                    "Thik hai! Kuch aur puchna?",
                ]
            }
        
        return random.choice(responses.get(understood.language, responses["english"]))

    def _compose_default(self, understood: UnderstoodMessage) -> str:
        """Default fallback responses for unclear intents."""
        responses = {
            "english": [
                "I'm not fully sure what you want. Can you ask as a question or command?",
                "Hmm, I didn't quite get that. Can you clarify?",
                "Sorry, I need more details to understand you better.",
            ],
            "hindi": [
                "मुझे समझ नहीं आया। क्या तुम सवाल पूछ सकते हो?",
                "अरे! थोड़ा स्पष्ट करो न।",
                "माफ करना। मुझे और जानकारी चाहिए।",
            ],
            "hinglish": [
                "Nahi samjha bhai. Kya question form mein puch sakta hai?",
                "Hmm, samjh nahi aya. Detail de na?",
                "Sorry yaar. Thoda aur bata.",
            ]
        }
        
        return random.choice(responses.get(understood.language, responses["english"]))

    def _learn_from_interaction(self, parsed: ParsedMessage, understood: UnderstoodMessage, reply: ChatResponse) -> None:
        """Track usage stats and unknown words after each interaction."""
        for unknown in understood.unknown_words:
            if unknown not in self.unknown_words:
                self.unknown_words.append(unknown)

        now_iso = self._now_iso()
        for known in understood.known_words:
            entry = self.mm.get_word(known)
            if not entry:
                continue
            updates = {
                "times_used": int(entry.get("times_used", 0)) + 1,
                "last_used": now_iso,
            }
            # best-effort update; no hard failure on stats write
            self.mm.edit_word(entry["word_id"], updates)

        self.current_context["last_intent"] = understood.intent
        self.current_context["last_sentiment"] = understood.sentiment
        self.current_context["last_unknown_count"] = len(understood.unknown_words)
        self.current_context["last_message_length"] = parsed.word_count

        if reply.learning_triggered:
            self.current_context["needs_teaching"] = True

    def _save_to_history(self, message: str, reply: ChatResponse) -> None:
        """Save latest pair to rolling conversation history."""
        self.conversation_history.append(
            {
                "timestamp": self._now_iso(),
                "user": message,
                "ai": reply.reply_text,
                "intent": self.current_context.get("last_intent"),
            }
        )
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY :]

    def get_context(self, key: str) -> Any:
        """Get value from context store."""
        return self.current_context.get(key)

    def set_context(self, key: str, value: Any) -> None:
        """Set context key/value."""
        self.current_context[key] = value

    def clear_context(self) -> None:
        """Clear all transient context."""
        self.current_context.clear()

    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return latest N history entries (default 10)."""
        if limit <= 0:
            return []
        return self.conversation_history[-limit:]

    def _cache_get(self, key: str) -> Optional[ChatResponse]:
        item = self.response_cache.get(key)
        if not item:
            return None
        self.response_cache.move_to_end(key)
        return item

    def _cache_set(self, key: str, value: ChatResponse) -> None:
        self.response_cache[key] = value
        self.response_cache.move_to_end(key)
        while len(self.response_cache) > self.RESPONSE_CACHE_LIMIT:
            self.response_cache.popitem(last=False)


if __name__ == "__main__":
    print("🤖 DISKNORY-FOR-AI Chat Engine v1.0")
    print("Type 'exit' to stop, 'help' for commands")
    print("=" * 50)

    mm = MemoryManager("brain/")
    engine = ChatEngine(mm)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("AI: Goodbye! 👋")
            break
        if user_input.lower() == "help":
            print("Commands: exit, help, history, stats, validate")
            continue
        if user_input.lower() == "history":
            for msg in engine.get_conversation_history():
                print(f"  You: {msg['user']}")
                print(f"  AI: {msg['ai']}")
            continue
        if user_input.lower() == "stats":
            stats = engine.mm.get_statistics()
            print(f"  Total words: {stats.get('total_words')}")
            continue
        if user_input.lower() == "validate":
            report = engine.mm.validate_brain()
            print(f"  Validation: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
            continue

        response = engine.process_message(user_input)
        print(f"AI: {response.reply_text}")
        if response.learning_triggered:
            unknown_count = len(engine.current_context.get("last_unknown_count", [])) if isinstance(engine.current_context.get("last_unknown_count"), list) else engine.current_context.get("last_unknown_count", 0)
            print(f"  📚 I found {unknown_count} unknown words. Please teach me!")
