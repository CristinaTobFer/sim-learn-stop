@echo off
setlocal

REM Pass all args to the Python runner.
REM irace typically calls: target-runner <config-id> <instance-id> <seed> <instance> <params...>

set THIS_DIR=%~dp0
set PYTHON_EXE=C:\TeamOriented\.venv\Scripts\python.exe

"%PYTHON_EXE%" "%THIS_DIR%main.py" %*

endlocal
