@echo off
REM Double-click this to launch Leti's interface in its own window.
REM
REM First run: creates leti_env\ in this folder and installs everything from
REM requirements.txt automatically - no manual "pip install" needed. Every
REM run after that just launches, unless requirements.txt has changed, in
REM which case it re-syncs the venv first.
setlocal enabledelayedexpansion

REM %~dp0 is always this .bat file's own folder, regardless of where it was
REM double-clicked from - so this works no matter where the project lives.
cd /d "%~dp0"

echo Leti - starting the interface
echo (from: %cd%)
echo.

set "VENV_DIR=leti_env"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "HASH_MARKER=%VENV_DIR%\.requirements_hash"
set "PLAYWRIGHT_MARKER=%VENV_DIR%\.playwright_installed"

where python >nul 2>nul
if not %errorlevel%==0 (
    echo ERROR: No Python found on this system. Install Python 3.10+ from python.org first,
    echo making sure to check "Add python.exe to PATH" during setup, then re-run this.
    pause
    exit /b 1
)

REM --- First run (or a deleted venv): create it -----------------------------
if not exist "%VENV_PYTHON%" (
    echo First run detected - setting up leti_env\ ^(this only happens once^)...
    python -m venv "%VENV_DIR%"
    if not !errorlevel!==0 (
        echo ERROR: Could not create the virtual environment. Make sure Python was installed
        echo with the standard installer from python.org ^(the "venv" module ships with it^).
        pause
        exit /b 1
    )
    echo Virtual environment created.
    echo.
)

REM --- Install/update dependencies, but only when requirements.txt actually
REM changed since the last successful install (hash-gated via PowerShell's
REM Get-FileHash, built into Windows 10+). ---
for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash requirements.txt -Algorithm SHA256).Hash" 2^>nul`) do set "CURRENT_HASH=%%H"

if "%CURRENT_HASH%"=="" (
    echo ERROR: Could not read requirements.txt - is this really the Leti project folder?
    pause
    exit /b 1
)

set "STORED_HASH="
if exist "%HASH_MARKER%" (
    for /f "usebackq delims=" %%H in ("%HASH_MARKER%") do set "STORED_HASH=%%H"
)

if not "%CURRENT_HASH%"=="%STORED_HASH%" (
    echo Installing dependencies into leti_env\ ^(first run, or requirements.txt changed^)...
    echo This can take a few minutes the first time - it's downloading everything Leti needs.
    echo.
    "%VENV_PYTHON%" -m pip install --upgrade pip --quiet
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if not !errorlevel!==0 (
        echo ERROR: Dependency installation failed - see the pip output above for which
        echo package failed and why. Fix that, then re-run this launcher - it will pick up
        echo where it left off.
        pause
        exit /b 1
    )
    > "%HASH_MARKER%" echo %CURRENT_HASH%
    echo Dependencies installed.
    echo.
)

REM --- Playwright's browser binary is a separate download from the pip
REM package itself, and only needs doing once per venv. ---
if not exist "%PLAYWRIGHT_MARKER%" (
    echo Downloading the Playwright browser ^(needed for web browsing/automation^)...
    "%VENV_PYTHON%" -m playwright install chromium
    if !errorlevel!==0 (
        echo. > "%PLAYWRIGHT_MARKER%"
    ) else (
        echo Warning: Playwright browser install failed - browser automation won't work until
        echo you run: leti_env\Scripts\python.exe -m playwright install chromium
    )
    echo.
)

call :ensure_ollama
if not !errorlevel!==0 (
    pause
    exit /b 1
)

"%VENV_PYTHON%" main.py --mode gui
set "STATUS=%errorlevel%"

echo.
if not "%STATUS%"=="0" (
    echo Leti exited with an error (code %STATUS%^) - see the output above for details.
) else (
    echo Leti closed.
)
pause
exit /b 0

REM ---------------------------------------------------------------------------
REM Ensures Ollama is installed, running, and has the models Leti needs. This is
REM what actually fixes "couldn't reach Ollama" instead of just warning about it:
REM installs it if missing, starts it as a background service if it's not
REM already running (survives after this window closes via Start-Process, not
REM tied to this cmd session), and pulls whatever models config/settings.yaml
REM asks for (ollama pull is cheap/no-op if already present, so this is safe to
REM run on every launch, not just the first).
REM ---------------------------------------------------------------------------
:ensure_ollama
where ollama >nul 2>nul
if not !errorlevel!==0 (
    echo Ollama not found - installing it now ^(one-time^)...
    where winget >nul 2>nul
    if !errorlevel!==0 (
        winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
        if not !errorlevel!==0 (
            echo ERROR: winget install failed. Install manually from https://ollama.com/download,
            echo then re-run this launcher.
            exit /b 1
        )
        echo Ollama installed.
        where ollama >nul 2>nul
        if not !errorlevel!==0 (
            echo Ollama was installed but this window's PATH hasn't picked it up yet.
            echo Close this window and run the launcher again - it'll be found next time.
            exit /b 1
        )
    ) else (
        echo ERROR: winget not found, so Ollama can't be auto-installed here. Install manually
        echo from https://ollama.com/download, then re-run this launcher.
        exit /b 1
    )
    echo.
)

curl -s -m 2 http://localhost:11434 >nul 2>nul
if not !errorlevel!==0 (
    echo Starting Ollama in the background ^(it'll keep running after this window closes^)...
    powershell -NoProfile -Command "Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden" >nul 2>nul
    set "OLLAMA_READY=0"
    for /l %%i in (1,1,30) do (
        curl -s -m 2 http://localhost:11434 >nul 2>nul
        if !errorlevel!==0 (
            set "OLLAMA_READY=1"
        )
        if "!OLLAMA_READY!"=="1" goto :ollama_ready_check
        timeout /t 1 >nul
    )
    :ollama_ready_check
    if not "!OLLAMA_READY!"=="1" (
        echo ERROR: Ollama still isn't responding at http://localhost:11434 after 30s.
        exit /b 1
    )
    echo Ollama is up.
    echo.
)

echo Checking Leti's configured models ^(skips anything already downloaded^)...
echo First run can take a while and needs several GB of disk space - ollama shows its own progress below.
for /f "usebackq delims=" %%M in (`"%VENV_PYTHON%" -c "import yaml; cfg=yaml.safe_load(open('config/settings.yaml')); o=cfg.get('ollama',{}); [print(m) for m in [o.get('reasoning_model'), o.get('fallback_reasoning_model'), o.get('vision_model'), o.get('embedding_model')] if m]"`) do (
    echo   - %%M
    ollama pull "%%M"
    if not !errorlevel!==0 echo     Warning: failed to pull '%%M' - Leti may not work correctly until this succeeds.
)
echo.
exit /b 0
