@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "TASK_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "TASK_PYTHON=%~dp0.venv\Scripts\python.exe"
"%TASK_PYTHON%" tools\engineering.py new
if errorlevel 1 echo FAILED: no successful result or fallback.
pause
