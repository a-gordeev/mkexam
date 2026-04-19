@echo off
setlocal

set PORT=%~1
if "%PORT%"=="" set PORT=5000

echo.
echo  mkexam launcher
echo  ===============
echo.

:: Check Python
python --version >/dev/null 2>&1
if errorlevel 1 goto no_python

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found: %PYVER%
echo.

:: Check .env
if exist .env goto check_venv
if exist .env.example copy .env.example .env >nul
if not exist .env echo GEMINI_API_KEY=> .env
echo  Created .env -- open it and set your GEMINI_API_KEY, then run again.
echo.
pause
exit /b 1

:check_venv
if exist .venv goto install_packages

echo  The following packages will be installed into a local virtual
echo  environment (.venv\) inside this folder. Nothing will be installed
echo  system-wide.
echo.
type requirements.txt
echo.
set /p CONFIRM= Allow installation? [Y/N]: 
if /i "%CONFIRM%"=="Y" goto create_venv
echo  Aborted.
pause
exit /b 0

:create_venv
echo.
echo  Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto venv_failed

:install_packages
echo  Installing/updating packages...
.venv\Scripts\python -m pip install -q -r requirements.txt
if errorlevel 1 goto pip_failed

:: Launch
echo.
echo  Starting mkexam at http://localhost:%PORT%
echo  Press Ctrl+C to stop.
echo.
start "" http://localhost:%PORT%
.venv\Scripts\python app.py --port %PORT%
exit /b 0

:no_python
echo  Python is not installed or not on PATH.
echo.
echo  Please install Python 3.11 or later from https://www.python.org/downloads/
echo  Make sure to check "Add python.exe to PATH" during installation.
echo.
pause
exit /b 1

:venv_failed
echo  Failed to create virtual environment.
pause
exit /b 1

:pip_failed
echo.
echo  Package installation failed. Check your internet connection and try again.
pause
exit /b 1
