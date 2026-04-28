#!/usr/bin/env python3
"""DISKNORY-FOR-AI - Main entry point.

Just type and chat. Hindi / Hinglish / English sab chalega.
Optional slash commands: /stats /backup /history /search /teach /unknown /exit
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)

from memory_manager import MemoryManager
from chat_engine import ChatEngine
from learning_loop import LearningLoop

WELCOME = """
=============================================================
  DISKNORY-FOR-AI  v1.2   (natural chat, no commands needed)
  Bas type kar aur baat kar. Hindi / Hinglish / English chalega.
  Optional: /stats /backup /history /search <w> /teach /unknown /exit
=============================================================
"""


def _parse_pipe(s: str):
    return [p.strip() for p in s.split("|")]


def main():
    print(WELCOME)
    print("[boot] Loading brain...")
    mm = MemoryManager("brain")
    engine = ChatEngine(mm)
    learner = LearningLoop(mm)
    total = mm.lex_index.get("total", 0)
    print(f"[boot] Brain ready — {total} words loaded.\n")

    started = datetime.now(timezone.utc)
    msg_count = 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not line:
            continue

        # ---------- optional slash commands ----------
        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower() if parts else ""
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in {"exit", "quit", "bye"}:
                print("[bye]")
                break

            elif cmd == "stats":
                s = mm.stats()
                print(f"  total words : {s['total_words']}")
                for k, v in s["by_language"].items():
                    print(f"  {k:10s}: {v}")
                print(f"  cache size  : {s['cache_size']}")
                print(f"  messages    : {msg_count}")

            elif cmd == "backup":
                path = mm.backup()
                print(f"  backup saved: {path}")

            elif cmd == "history":
                for h in engine.get_conversation_history(10):
                    print(f"  you: {h['user']}")
                    print(f"  ai : {h['ai']}")

            elif cmd == "search":
                word = arg.strip()
                entry = mm.get_word(word)
                if not entry:
                    print(f"  '{word}' not found in brain.")
                else:
                    print(f"  word    : {entry['word']}")
                    print(f"  hindi   : {entry.get('hindi_meaning', '')}")
                    print(f"  english : {entry.get('english_meaning', '')}")
                    print(f"  example : {entry.get('example_sentence_en', '')}")

            elif cmd == "teach":
                p = _parse_pipe(arg)
                if len(p) < 2:
                    print("  usage: /teach <word> | <hindi> | <english> | <example>")
                else:
                    word = p[0]
                    hi   = p[1] if len(p) > 1 else ""
                    en   = p[2] if len(p) > 2 else ""
                    ex   = p[3] if len(p) > 3 else ""
                    r = learner.teach(word, hi, en, ex)
                    print("  ok:" if r.success else "  err:", r.message or r.error)

            elif cmd == "unknown":
                items = mm.list_unknown(20)
                if not items:
                    print("  (no unknown words queued)")
                else:
                    for item in items:
                        print(f"  {item['word']:20s} seen {item['count']}x")

            elif cmd == "help":
                print("  /stats  /backup  /history  /search <word>")
                print("  /teach <word>|<hindi>|<english>|<example>")
                print("  /unknown  /exit")

            else:
                print(f"  unknown command: /{cmd}  (try /help)")

            continue

        # ---------- normal chat ----------
        resp = engine.process_message(line)
        msg_count += 1
        print(f"ai > {resp.reply_text}")

    dur = datetime.now(timezone.utc) - started
    print(f"session: {dur} | messages: {msg_count}")


if __name__ == "__main__":
    main()
