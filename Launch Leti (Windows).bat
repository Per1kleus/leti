@echo off
REM ===========================================================================
REM  Double-click this to start Leti. That is the whole instruction.
REM
REM  First run: prepares a private Python environment in this folder, installs
REM  everything requirements.txt asks for, and makes sure Ollama is running with
REM  the models Leti is configured for. Nothing is installed system-wide, no
REM  administrator prompt appears, and PATH is not touched.
REM
REM  If this machine has no Python at all, one is fetched into leti_runtime\.
REM  Your own Python, if you have one, is never written to.
REM
REM  Every run after the first is a quick check and then Leti starts.
REM
REM  Two jobs, deliberately split:
REM    - this file finds SOME Python to run, because it is what runs when there
REM      is none. Batch is the only language guaranteed to be here.
REM    - launcher\bootstrap.py does everything after that, because it is also
REM      what Leti.exe uses, and one copy of that logic is the point.
REM
REM  Developers: nothing here is required. `python main.py --mode gui` from a
REM  prepared checkout works exactly as it always did.
REM ===========================================================================
setlocal enabledelayedexpansion

REM %~dp0 is this file's own folder, wherever it was double-clicked from.
cd /d "%~dp0"

echo Leti
echo.

set "RUNTIME_DIR=leti_runtime"
set "RUNTIME_PYTHON=%RUNTIME_DIR%\python.exe"
set "VENV_PYTHON=leti_env\Scripts\python.exe"
set "PY_VERSION=3.11.9"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-embed-amd64.zip"

if not exist "requirements.txt" (
    echo Leti could not find its own files.
    echo   What failed: requirements.txt is not in this folder.
    echo   It was trying to: work out which packages Leti needs.
    echo   Keep this launcher inside the Leti folder it came with.
    echo.
    pause
    exit /b 2
)

REM --- Find a Python to run the setup with ----------------------------------
REM Order: one this launcher prepared earlier, then the system's, then fetch.
REM Only enough to run launcher\bootstrap.py - which then decides properly, and
REM will build a venv from a system Python when there is a good one.
set "SETUP_PYTHON="
if exist "%VENV_PYTHON%"    set "SETUP_PYTHON=%VENV_PYTHON%"
if not defined SETUP_PYTHON if exist "%RUNTIME_PYTHON%" set "SETUP_PYTHON=%RUNTIME_PYTHON%"

if not defined SETUP_PYTHON (
    for %%P in (py.exe py python.exe python python3.exe python3) do (
        if not defined SETUP_PYTHON (
            where %%P >nul 2>nul
            if !errorlevel!==0 set "SETUP_PYTHON=%%P"
        )
    )
)

if not defined SETUP_PYTHON (
    echo Preparing Leti...
    echo   No Python on this machine - fetching Leti's own ^(about 11 MB^).
    echo   Nothing is installed system-wide and PATH is not changed.
    echo.
    REM PowerShell ships with Windows 10 and later, so this needs nothing that
    REM is not already here. TLS 1.2 is set explicitly because the default on
    REM older builds is too old for python.org.
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
      "New-Item -ItemType Directory -Force -Path '%RUNTIME_DIR%' | Out-Null;" ^
      "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%RUNTIME_DIR%\python-embed.zip' -UseBasicParsing;" ^
      "Expand-Archive -Path '%RUNTIME_DIR%\python-embed.zip' -DestinationPath '%RUNTIME_DIR%' -Force;" ^
      "Remove-Item '%RUNTIME_DIR%\python-embed.zip' -Force"
    if not !errorlevel!==0 (
        echo Leti could not finish setting itself up.
        echo   What failed: Python could not be downloaded.
        echo   It was trying to: fetch a private copy of Python, because this
        echo                     machine has none Leti can use.
        echo   Trying again is safe - setup carries on from the last step that finished.
        echo   If this keeps happening, check whether a firewall or proxy is
        echo   blocking https://www.python.org, or install Python 3.11+ from
        echo   python.org and run this launcher again.
        echo.
        pause
        exit /b 1
    )
    if not exist "%RUNTIME_PYTHON%" (
        echo Leti could not finish setting itself up.
        echo   What failed: the downloaded Python is not where it was expected.
        echo   It was trying to: unpack Python into the Leti folder.
        echo   Trying again is safe - delete the leti_runtime folder first.
        echo.
        pause
        exit /b 1
    )
    set "SETUP_PYTHON=%RUNTIME_PYTHON%"
    echo   Python fetched.
    echo.
)

REM --- Everything else is launcher\bootstrap.py ------------------------------
REM It opens up the fetched runtime's path file, gives it pip, installs whatever
REM is missing from requirements.txt, and starts main.py. Leti.exe runs the same
REM module, so there is one description of what a prepared machine looks like.
REM --print-python prepares the machine and prints the interpreter Leti should
REM run on. Asked for rather than worked out here: this file cannot tell a venv
REM built from a good system Python from a runtime fetched because the system one
REM was too old, and guessing wrong runs Leti on the interpreter that was
REM rejected. Progress goes to stderr in this mode, so it is still on screen.
set "LETI_PYTHON="
for /f "usebackq delims=" %%P in (`"%SETUP_PYTHON%" launcher\leti_launcher.py --print-python`) do set "LETI_PYTHON=%%P"

if not defined LETI_PYTHON (
    echo.
    echo Setup did not finish. Nothing is broken - running this launcher again
    echo carries on from the last step that completed. If it keeps stopping here,
    echo look in logs\ in this folder.
    echo.
    pause
    exit /b 1
)
if not exist "%LETI_PYTHON%" (
    echo.
    echo Leti could not finish setting itself up.
    echo   What failed: the prepared Python is not where setup said it was.
    echo   It was trying to: start Leti with the environment it just prepared.
    echo   Trying again is safe - delete leti_env and leti_runtime first.
    echo.
    pause
    exit /b 1
)

call :ensure_ollama
if not !errorlevel!==0 (
    pause
    exit /b 1
)

REM Appended rather than replaced, and without a stray separator when it was
REM empty - an empty entry on PYTHONPATH is the current directory, which is
REM harmless here and confusing everywhere else.
if defined PYTHONPATH (
    set "PYTHONPATH=%cd%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%cd%"
)
"%LETI_PYTHON%" main.py --mode gui
set "STATUS=%errorlevel%"

echo.
if not "%STATUS%"=="0" (
    echo Leti exited with an error ^(code %STATUS%^) - see above, and logs\ in this folder.
    pause
) else (
    echo Leti closed.
)
exit /b 0

REM ---------------------------------------------------------------------------
REM Ollama: installed if missing, started if not running, and asked for whatever
REM models config/settings.yaml names. Left in this file rather than moved into
REM bootstrap.py because it is Windows package management rather than Python
REM environment management, and because core/model_setup.py already owns the
REM question of WHICH model this machine should run - this only makes sure the
REM server is there for it to ask.
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

REM Skipped once the models have been confirmed present, so a normal launch does
REM not shell out to ollama four times. The marker is removed by changing
REM config/settings.yaml, which is the only thing that changes the answer.
set "MODEL_MARKER=data\.models_pulled"
if exist "%MODEL_MARKER%" (
    for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash 'config\settings.yaml' -Algorithm SHA256).Hash" 2^>nul`) do set "CFG_HASH=%%H"
    set "STORED_CFG="
    for /f "usebackq delims=" %%H in ("%MODEL_MARKER%") do set "STORED_CFG=%%H"
    if "!CFG_HASH!"=="!STORED_CFG!" exit /b 0
)

echo Checking Leti's configured models ^(skips anything already downloaded^)...
echo First run can take a while and needs several GB of disk space - ollama shows its own progress below.
set "PULL_OK=1"
for /f "usebackq delims=" %%M in (`"%LETI_PYTHON%" -c "import yaml; cfg=yaml.safe_load(open('config/settings.yaml')); o=cfg.get('ollama',{}); [print(m) for m in [o.get('reasoning_model'), o.get('fallback_reasoning_model'), o.get('vision_model'), o.get('embedding_model')] if m]" 2^>nul`) do (
    echo   - %%M
    ollama pull "%%M"
    if not !errorlevel!==0 (
        echo     Warning: failed to pull '%%M' - Leti may not work correctly until this succeeds.
        set "PULL_OK=0"
    )
)
if "!PULL_OK!"=="1" (
    for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash 'config\settings.yaml' -Algorithm SHA256).Hash" 2^>nul`) do (
        if not exist "data" mkdir "data"
        > "%MODEL_MARKER%" echo %%H
    )
)
echo.
exit /b 0
