# DISKNORY-FOR-AI

Self-learning, **offline** AI brain in pure Python — supports **English, Hindi, Hinglish**.
No internet, no API keys, no external libraries. Just Python 3.8+.

---

## Quick Start

### Windows
1. Extract the ZIP anywhere (e.g. `C:\DISKNORY-FOR-AI\`)
2. Double-click **`runner.bat`**
3. First run will build the dataset, validate it, then drop you into the chat.

### Linux / Mac
```bash
chmod +x runner.sh
./runner.sh
```

### Manual
```bash
python tools/build_dataset.py    # only first time
python tools/rebuild_indexes.py  # only first time
python runtime/main.py           # chat
```

---

## Folder Structure

```
DISKNORY-FOR-AI/
├── brain/                          # ALL data lives here (the "brain")
│   ├── data/
│   │   ├── english_core.jsonl      # >=1000 English words
│   │   ├── hindi_core.jsonl        # >=100 Hindi words (auto-grows)
│   │   └── hinglish_core.jsonl     # >=80 Hinglish words (auto-grows)
│   ├── schema/
│   │   ├── brain_schema_v1.json
│   │   └── memory_event_schema_v1.json
│   ├── indexes/
│   │   ├── lexeme_index.json       # word -> location (O(1) lookup)
│   │   └── prefix_index.json
│   ├── journal/
│   │   └── events.log              # every add/edit/delete logged
│   ├── backups/                    # snapshot folder
│   └── learning_queue.jsonl        # unknown words AI saw
├── runtime/                        # the engine
│   ├── memory_manager.py           # CRUD + journal + index
│   ├── chat_engine.py              # tokenize, intent, reply
│   ├── learning_loop.py            # teach / correct / reinforce
│   ├── validator.py                # schema enforcement
│   └── main.py                     # interactive CLI
├── tools/
│   ├── build_dataset.py            # generates seed data
│   ├── validate_brain.py           # validates all data
│   ├── rebuild_indexes.py          # rebuilds indexes
│   └── backup_brain.py             # snapshot
├── runner.bat                      # Windows one-click
├── runner.sh                       # Linux/Mac one-click
├── requirements.txt
└── README.md
```

---

## CLI Commands (inside the chat)

```
help                                show all commands
stats                               brain statistics
validate                            validate all files
search <word>                       look up a word
learn <word> | <hindi> | <english> | <example>
correct <word_id> | field=value [| field=value]
delete <word_id>                    archive a word (recoverable)
unknown                             list unknown-word queue
history                             last 10 messages
backup                              snapshot brain folder
rebuild                             rebuild indexes
clear                               clear chat context
exit                                leave
```

### Examples
```
you> hello
you> search love
you> learn jugaad | तरकीब | clever workaround | He found a jugaad.
you> stats
you> backup
```

---

## Why this design is "human-like" + crash-safe

1. **JSONL one-word-per-line** → editing one word can never corrupt another.
2. **Schema-validated writes** → garbage data is rejected before disk.
3. **Atomic file replace** → power loss never leaves a half-written file.
4. **Journal log** of every event → full audit + future rollback.
5. **Index files** keep lookup ~O(1) so even 1M words reply in <2 sec.
6. **Cache** of hot words for instant repeat lookups.
7. **Self-learning loop** → unknown words queued, user can teach with `learn`.
8. **Soft delete** → words go to archive, never truly gone.
9. **Versioning** on every entry → corrections track history.
10. **Future-proof** → unknown extra fields preserved, schema version field set.

---

## Add a new language

1. Create `brain/data/<lang>_core.jsonl`
2. Add the language prefix to `brain/schema/brain_schema_v1.json` → `language_prefix`
3. Tell `MemoryManager.data_files` about it (one line in `runtime/memory_manager.py`)
4. Run `python tools/rebuild_indexes.py`

That's it — the rest of the engine doesn't need to change.

---

## Roadmap (you can extend safely)

- [ ] N-gram model on top of dictionary for sentence prediction
- [ ] Embedding cache for synonyms / fuzzy match
- [ ] WebSocket server wrapper to use brain from a UI
- [ ] Auto-import from PDF / TXT (drop file, brain ingests + queues unknowns)
- [ ] Multi-user separate memories

---

## License
MIT — use, fork, modify freely.
