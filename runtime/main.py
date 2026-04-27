#!/usr/bin/env python3
"""
DISKNORY-FOR-AI — Main Entry Point
===================================
Run this file to start the AI chat interface.

Usage:
    python runtime/main.py

Commands:
    exit, help, stats, validate, history, learn, backup, clear, search, add, edit
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Allow direct execution from repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.chat_engine import ChatEngine
from runtime.memory_manager import MemoryManager
from runtime.validator import validate_all_files

BRAIN_PATH = "brain/"
SESSION_PATH = os.path.join("runtime", "session_history.json")
WELCOME_MESSAGE = """
🧠 ══════════════════════════════════════════
   DISKNORY-FOR-AI v1.0 — Offline Learning AI
   ═══════════════════════════════════════════
   Type 'help' for commands, 'exit' to quit
   ═══════════════════════════════════════════
"""


class AIApplication:
    """Main application controller for system startup and command loop."""

    def __init__(self) -> None:
        self.mm: Optional[MemoryManager] = None
        self.engine: Optional[ChatEngine] = None
        self.session_start: Optional[datetime] = None
        self.message_count: int = 0
        self.running: bool = False
        self.session_events: List[Dict[str, Any]] = []

    def initialize(self) -> bool:
        """Initialize memory manager, chat engine, and startup validation."""
        try:
            print("🔧 Initializing AI brain...")
            self.mm = MemoryManager(BRAIN_PATH)
            print("  ✅ Memory Manager loaded")

            self.engine = ChatEngine(self.mm)
            print("  ✅ Chat Engine loaded")

            print("  🔍 Running validation...")
            report = validate_all_files(BRAIN_PATH)
            if report.overall_passed:
                print("  ✅ Validation PASSED")
            else:
                print(f"  ⚠️ Validation failed ({report.total_errors} errors, {report.critical_errors} critical)")
                if report.critical_errors > 0:
                    print("  ⚠️ Some features may not work correctly")

            stats = self.mm.get_statistics()
            print(f"  📊 Total words in brain: {stats.get('total_words', 0)}")

            self._load_previous_session()
            self.session_start = datetime.now()
            print("  ✅ AI initialized successfully!\n")
            return True
        except Exception as exc:  # pragma: no cover - defensive startup guard
            print(f"  ❌ Initialization failed: {exc}")
            return False

    def show_welcome(self) -> None:
        """Display startup banner."""
        print(WELCOME_MESSAGE)

    def show_help(self) -> None:
        """Display command help."""
        help_text = """
📋 AVAILABLE COMMANDS:
═══════════════════════════════════════
exit, quit, bye    → Exit the AI
help               → Show this help message
stats              → Show brain statistics
validate           → Run full brain validation
history            → Show conversation history
learn              → Show unknown words queue
backup             → Create brain backup
clear              → Clear conversation context
search <word>      → Search for word meaning
add <word>         → Add new word manually
edit <word>        → Edit existing word
═══════════════════════════════════════
"""
        print(help_text)

    def show_stats(self) -> None:
        """Display memory/index/session stats."""
        if not self.mm:
            print("  ❌ Memory manager not initialized")
            return
        stats = self.mm.get_statistics()
        print("\n📊 BRAIN STATISTICS:")
        print(f"  Total words: {stats.get('total_words', 0)}")
        print(f"  Total nouns: {stats.get('total_nouns', 0)}")
        print(f"  Total verbs: {stats.get('total_verbs', 0)}")
        print(f"  Total adjectives: {stats.get('total_adjectives', 0)}")
        print(f"  Common words: {stats.get('total_common', 0)}")
        print(f"  Technical words: {stats.get('total_technical', 0)}")
        print(f"  Average confidence: {float(stats.get('avg_confidence', 0.0)):.2f}")
        print(f"  Session messages: {self.message_count}")
        print()

    def show_history(self) -> None:
        """Display last 10 turns from current in-memory history."""
        if not self.engine:
            print("  ❌ Chat engine not initialized")
            return
        history = self.engine.get_conversation_history(limit=10)
        if not history:
            print("  📭 No conversation history yet")
            return

        print("\n📜 CONVERSATION HISTORY:")
        for i, msg in enumerate(history, 1):
            print(f"  {i}. You: {msg.get('user', '')[:80]}")
            print(f"     AI: {msg.get('ai', '')[:80]}")
        print()

    def show_learn_queue(self) -> None:
        """Display unknown words collected by chat engine."""
        if not self.engine:
            print("  ❌ Chat engine not initialized")
            return
        unknown = self.engine.unknown_words
        if not unknown:
            print("  🎉 No unknown words queued")
            return

        print(f"\n📚 UNKNOWN WORDS ({len(unknown)}):")
        for word in unknown[:10]:
            print(f"  - {word}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")
        print()

    def create_backup(self) -> None:
        """Create timestamped backup under backups/ directory."""
        if not self.mm:
            print("  ❌ Memory manager not initialized")
            return
        backup_path = f"backups/brain_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.mm.backup_brain(backup_path):
            print(f"  ✅ Backup created: {backup_path}")
        else:
            print("  ❌ Backup failed")
        print()

    def search_word(self, word: str) -> None:
        """Lookup and print word details from memory manager."""
        if not self.mm:
            print("  ❌ Memory manager not initialized")
            return
        result = self.mm.get_word(word)
        if result:
            print(f"\n📖 WORD: {result.get('word', '')}")
            print(f"  Hindi: {result.get('hindi_meaning', 'N/A')}")
            print(f"  English: {result.get('english_meaning', 'N/A')}")
            print(f"  Part of Speech: {result.get('part_of_speech', 'N/A')}")
            if result.get("example_sentence_en"):
                print(f"  Example: {result.get('example_sentence_en')}")
        else:
            print(f"  ❌ Word '{word}' not found in brain")
            print("  💡 Use 'add <word>' to teach me")
        print()

    def add_word_interactive(self, word: str) -> None:
        """Prompt user for fields and create new word entry."""
        if not self.mm:
            print("  ❌ Memory manager not initialized")
            return
        print(f"\n📝 Adding new word: {word}")
        hindi = input("  Hindi meaning: ").strip()
        english = input("  English meaning: ").strip()
        pos = input("  Part of speech (noun/verb/adjective): ").strip() or "noun"
        example = input("  Example sentence: ").strip()

        entry = {
            "word": word,
            "lowercase": word.lower(),
            "hindi_meaning": hindi or f"{word} (हिंदी अर्थ जोड़ना बाकी)",
            "english_meaning": english or f"Meaning of {word}",
            "part_of_speech": pos,
            "example_sentence_en": example or f"I used the word {word} in a sentence.",
        }
        result = self.mm.add_word(entry)
        if result.success:
            print(f"  ✅ Word added successfully! ID: {result.word_id}")
        else:
            print(f"  ❌ Failed to add word: {result.errors}")
        print()

    def edit_word_interactive(self, word: str) -> None:
        """Prompt user for updates and apply to existing word."""
        if not self.mm:
            print("  ❌ Memory manager not initialized")
            return

        existing = self.mm.get_word(word)
        if not existing:
            print(f"  ❌ Word '{word}' not found")
            return

        print(f"\n✏️ Editing word: {existing.get('word')} ({existing.get('word_id')})")
        print("Press Enter to keep existing value")
        hindi = input(f"  Hindi meaning [{existing.get('hindi_meaning', '')}]: ").strip()
        english = input(f"  English meaning [{existing.get('english_meaning', '')}]: ").strip()
        pos = input(f"  Part of speech [{existing.get('part_of_speech', '')}]: ").strip()
        example = input(f"  Example sentence [{existing.get('example_sentence_en', '')}]: ").strip()

        updates: Dict[str, Any] = {}
        if hindi:
            updates["hindi_meaning"] = hindi
        if english:
            updates["english_meaning"] = english
        if pos:
            updates["part_of_speech"] = pos
        if example:
            updates["example_sentence_en"] = example

        if not updates:
            print("  ℹ️ No updates provided")
            return

        result = self.mm.edit_word(existing["word_id"], updates)
        if result.success:
            print(f"  ✅ Word updated successfully! Version: {result.version}")
        else:
            print(f"  ❌ Failed to edit word: {result.errors}")
        print()

    def handle_command(self, user_input: str) -> Optional[bool]:
        """Handle commands. Returns True=handled, False=not command, None=exit."""
        parts = user_input.strip().split()
        if not parts:
            return True

        command = parts[0].lower()
        args = parts[1:]

        if command in {"exit", "quit", "bye"}:
            return None
        if command == "help":
            self.show_help()
            return True
        if command == "stats":
            self.show_stats()
            return True
        if command == "validate":
            print("  🔍 Running full validation...")
            report = validate_all_files(BRAIN_PATH)
            print(f"  Result: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
            print(f"  Files checked: {report.total_files}")
            print(f"  Files failed: {report.files_failed}")
            print(f"  Errors: {report.total_errors}")
            print(f"  Warnings: {report.total_warnings}")
            return True
        if command == "history":
            self.show_history()
            return True
        if command == "learn":
            self.show_learn_queue()
            return True
        if command == "backup":
            self.create_backup()
            return True
        if command == "clear":
            if self.engine:
                self.engine.clear_context()
            print("  ✅ Conversation context cleared")
            return True
        if command == "search" and args:
            self.search_word(" ".join(args))
            return True
        if command == "add" and args:
            self.add_word_interactive(" ".join(args))
            return True
        if command == "edit" and args:
            self.edit_word_interactive(" ".join(args))
            return True

        return False

    def run_chat_loop(self) -> None:
        """Run interactive chat loop until exit/interrupt."""
        if not self.engine:
            raise RuntimeError("Chat engine not initialized")

        self.running = True
        self.show_welcome()

        while self.running:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue

                handled = self.handle_command(user_input)
                if handled is None:
                    break
                if handled:
                    continue

                response = self.engine.process_message(user_input)
                print(f"🤖 AI: {response.reply_text}")
                if response.learning_triggered:
                    print("  📚 I found unknown words. Run 'learn' to review them.")

                self.message_count += 1
                self.session_events.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "user": user_input,
                        "ai": response.reply_text,
                    }
                )

            except KeyboardInterrupt:
                print("\n\n  ⚠️ Interrupted by user")
                break
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"\n  ❌ Error: {exc}")
                print("  💡 Try 'validate' command to check brain health")

        self.shutdown()

    def _load_previous_session(self) -> None:
        """Load previous session metadata if available."""
        if not os.path.exists(SESSION_PATH):
            return
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            last = data.get("last_session", {})
            if last:
                print(
                    f"  ℹ️ Last session: {last.get('message_count', 0)} messages, ended at {last.get('ended_at', 'unknown')}"
                )
        except Exception:
            # Non-fatal; continue startup.
            pass

    def _save_session(self) -> None:
        """Persist current session metadata and recent turns to disk."""
        os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        now = datetime.now()
        payload = {
            "last_session": {
                "started_at": self.session_start.isoformat(timespec="seconds") if self.session_start else None,
                "ended_at": now.isoformat(timespec="seconds"),
                "message_count": self.message_count,
                "duration_seconds": int((now - self.session_start).total_seconds()) if self.session_start else 0,
            },
            "recent_events": self.session_events[-50:],
        }
        with open(SESSION_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def shutdown(self) -> None:
        """Gracefully shutdown and print session summary."""
        print("\n\n🧠 Shutting down AI...")
        print("  💾 Saving session data...")
        self._save_session()

        if self.session_start:
            duration = datetime.now() - self.session_start
            print("  📊 Session summary:")
            print(f"     Duration: {duration}")
            print(f"     Messages: {self.message_count}")

        print("  ✅ Goodbye! Come back soon! 👋\n")


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("DISKNORY-FOR-AI — Starting...")
    print("=" * 60 + "\n")

    app = AIApplication()
    if not app.initialize():
        print("\n❌ Failed to initialize AI. Please check brain files.")
        sys.exit(1)

    app.run_chat_loop()


if __name__ == "__main__":
    main()
