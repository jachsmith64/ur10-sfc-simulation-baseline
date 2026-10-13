@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "TASK_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "TASK_PYTHON=%~dp0.venv\Scripts\python.exe"
"%TASK_PYTHON%" tools\snapshot.py copy
if errorlevel 1 echo FAILED: snapshot not copied.
pause
