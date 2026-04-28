@echo off
title DISKNORY-FOR-AI
echo ================================================
echo   DISKNORY-FOR-AI  -  Starting...
echo ================================================
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "brain\data\english_core.jsonl" (
    echo [first run] building dataset...
    python tools\build_dataset.py
    python tools\rebuild_indexes.py
    python tools\validate_brain.py
)

python runtime\main.py
pause
