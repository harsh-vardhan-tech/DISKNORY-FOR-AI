

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

WELCOME = 
=============================================================
  DISKNORY-FOR-AI  v1.1   (natural chat, no commands needed)
  Bas type kar aur baat kar. Hindi / Hinglish / English chalega.
  (slash commands optional: /stats /backup /history /search /teach /exit)
=============================================================



def parse_pipe(s: str):
    return [p.strip() for p in s.split(\"|\")]


def main():
    print(WELCOME)
    print(\"[boot] loading brain...\")
    mm = MemoryManager(\"brain\")
    engine = ChatEngine(mm)
    learner = LearningLoop(mm)
    print(f\"[boot] words loaded: {mm.lex_index.get('total', 0)}\")
    started = datetime.now(timezone.utc)
    msg_count = 0

    while True:
        try:
            line = input(\"\nyou> \").strip()
        except (EOFError, KeyboardInterrupt):
            print(\"\n[bye]\")
            break
        if not line:
            continue

        # ---------- optional slash commands ----------
        if line.startswith(\"/\"):
            cmd_parts = line[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower() if cmd_parts else \"\"
            arg = cmd_parts[1] if len(cmd_parts) > 1 else \"\"

            if cmd in {\"exit\", \"quit\", \"bye\"}:
                print(\"[bye]\")
                break
            if cmd == \"stats\":
                s = mm.stats()
                print(f\"  total words : {s['total_words']}\")
                for k, v in s[\"by_language\"].items():
                    print(f\"  {k:8s}: {v}\")
                print(f\"  cache size  : {s['cache_size']}\")
                print(f\"  messages    : {msg_count}\")
                continue
            if cmd == \"backup\":
                print(f\"  backup at: {mm.backup()}\")
                continue
            if cmd == \"history\":
                for h in engine.get_conversation_history(10):
                    print(f\"  you: {h['user']}\")
                    print(f\"  ai : {h['ai']}\")
                continue
            if cmd == \"search\":
                e = mm.get_word(arg.strip())
                if not e:
                    print(f\"  '{arg}' not in brain.\")
                else:
                    print(f\"  word    : {e['word']}\")
                    print(f\"  hindi   : {e.get('hindi_meaning','')}\")
                    print(f\"  english : {e.get('english_meaning','')}\")
                    print(f\"  example : {e.get('example_sentence_en','')}\")
                continue
            if cmd == \"teach\":
                parts = parse_pipe(arg)
                if len(parts) < 2:
                    print(\"  usage: /teach <word> | <hindi> | <english> | <example>\")
                    continue
                w = parts[0]
                hi = parts[1] if len(parts) > 1 else \"\"
                en = parts[2] if len(parts) > 2 else \"\"
                ex = parts[3] if len(parts) > 3 else \"\"
                r = learner.teach(w, hi, en, ex)
                print(\"  ok:\" if r.success else \"  err:\", r.message or r.error)
                continue
            if cmd == \"help\":
                print(\"  /stats  /backup  /history  /search <w>  /teach <w>|<hi>|<en>|<ex>  /exit\")
                continue
            print(f\"  unknown slash command: /{cmd}  (try /help)\")
            continue

        # ---------- normal chat ----------
        resp = engine.process_message(line)
        msg_count += 1
        print(f\"ai > {resp.reply_text}\")

    dur = datetime.now(timezone.utc) - started
    print(f\"session: {dur} | messages: {msg_count}\")


if __name__ == \"__main__\":
    main()
"
Observation: Overwrite successful: /app/DISKNORY-FOR-AI/runtime/main.py