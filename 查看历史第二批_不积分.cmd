@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "TASK_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "TASK_PYTHON=%~dp0.venv\Scripts\python.exe"
"%TASK_PYTHON%" tools\engineering.py show --run-id 20261012T123042308025Z_05af114f
if errorlevel 1 echo FAILED: explicit history unavailable.
pause
