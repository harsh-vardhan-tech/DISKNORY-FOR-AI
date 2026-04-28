# INSTALL & USE GUIDE — DISKNORY-FOR-AI

## 1. Download
Tujhe ek hi file download karni hai:

**`DISKNORY-FOR-AI.zip`**

Yeh project ke root me bani hui hai. Iska poora path:
```
/app/DISKNORY-FOR-AI.zip
```

## 2. Extract karne ke baad ka structure
ZIP ko jab extract karega, tujhe yeh dikhega:

```
DISKNORY-FOR-AI/
├── runner.bat              <-- WINDOWS pe IS PE DOUBLE-CLICK
├── runner.sh               <-- LINUX/MAC ke liye
├── README.md               <-- full documentation
├── requirements.txt
├── INSTALL_GUIDE.md        <-- yeh file
├── brain/                  <-- AI ka brain (data + schema + journal)
├── runtime/                <-- engine (chat, memory, learning)
└── tools/                  <-- build/validate/backup scripts
```

## 3. Pehli baar chalane ka tareeqa

### Windows
1. ZIP ko `C:\DISKNORY-FOR-AI\` jaisi jagah extract kar.
2. Python 3.8+ install hona chahiye (https://python.org se).
3. `runner.bat` pe **double-click** kar.
4. Pehli baar dataset build hoga (~5 sec), phir chat khulega.

### Linux / Mac
```bash
unzip DISKNORY-FOR-AI.zip
cd DISKNORY-FOR-AI
chmod +x runner.sh
./runner.sh
```

## 4. Konsi file kahan rakhni hai (agar manually shift karna ho)

| File / Folder | Final location | Use |
|---|---|---|
| `runtime/main.py` | `<root>/runtime/main.py` | Entry point |
| `runtime/memory_manager.py` | `<root>/runtime/` | Brain CRUD |
| `runtime/chat_engine.py` | `<root>/runtime/` | Chat logic |
| `runtime/learning_loop.py` | `<root>/runtime/` | Self-learning |
| `runtime/validator.py` | `<root>/runtime/` | Schema check |
| `brain/schema/*.json` | `<root>/brain/schema/` | Strict schema |
| `brain/data/*.jsonl` | `<root>/brain/data/` | Word data |
| `brain/indexes/*.json` | `<root>/brain/indexes/` | Fast lookup |
| `brain/journal/events.log` | `<root>/brain/journal/` | Audit log |
| `tools/build_dataset.py` | `<root>/tools/` | First-time data builder |
| `runner.bat` / `runner.sh` | `<root>/` | One-click launcher |

## 5. Pehli baar chalane par AI bolega
```
[boot] loading brain...
[boot] words loaded: 1XXX
you>
```

Tu type kar:
- `help` — saare commands
- `search love` — word dhoond
- `learn jugaad | तरकीब | clever workaround | He found a jugaad.` — naya word sikha
- `stats` — kitne words hain
- `backup` — snapshot
- `exit` — bahar

## 6. Naye words add karne ke 2 tareeqe

### A) AI ko sikha (recommended)
```
you> learn awesome | शानदार | excellent | That movie was awesome.
```

### B) Bulk add (programmatic)
`tools/build_dataset.py` me apne tuples add kar (word, hindi, english, example, pos), phir:
```
python tools/build_dataset.py
python tools/rebuild_indexes.py
```

## 7. Backup / Rollback
```
you> backup
```
Backup `brain/backups/backup_YYYYMMDD_HHMMSS/` me chala jaata hai. Restore karna ho to us folder ko `brain/` ke andar copy paste kar de — done.

## 8. Validation
Kabhi shak ho ki data corrupt ho gaya:
```
python tools/validate_brain.py
```
Yeh JSON me detail report dega — kaun si line, kya error.

## 9. Performance
- Currently dataset choti hai → reply <100ms.
- 100k+ words tak `lexeme_index.json` ke wajah se reply <2 sec rahega.
- Hot words cache me chale jaate hain → repeat queries instant.

Bas! Sab ready hai. Run karo aur baat karo apne brain se. 🧠
