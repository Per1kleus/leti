@echo off
REM ===========================================================================
REM  Double-click this to build Leti.exe.
REM
REM  You do NOT need this to use Leti. "Launch Leti (Windows).bat" already runs
REM  it. This is for when you would rather have a real Leti.exe to double-click
REM  and to pin to the taskbar - the two start exactly the same Leti.
REM
REM  It needs the Python that "Launch Leti (Windows).bat" already prepared, so
REM  run that once first. Everything it downloads (PyInstaller) goes into that
REM  same private environment; nothing is installed system-wide.
REM
REM  The result is Leti.exe next to this file. Your Desktop and Start Menu
REM  shortcuts will point at it the next time they are placed or repaired -
REM  scripts\install_windows_launcher.ps1 does that immediately.
REM ===========================================================================
setlocal

cd /d "%~dp0"

echo Building Leti.exe
echo.

if not exist "launcher\build_exe.py" (
    echo Leti could not find its own files.
    echo   What failed: launcher\build_exe.py is not in this folder.
    echo   It was trying to: build Leti.exe.
    echo   Keep this file inside the Leti folder it came with.
    echo.
    pause
    exit /b 2
)

REM The environment a launcher already prepared. Building needs the same Python
REM Leti runs on, and asking for it rather than guessing is the same rule
REM "Launch Leti (Windows).bat" follows.
set "BUILD_PYTHON="
if exist "leti_env\Scripts\python.exe"  set "BUILD_PYTHON=leti_env\Scripts\python.exe"
if not defined BUILD_PYTHON if exist "leti_runtime\python.exe" set "BUILD_PYTHON=leti_runtime\python.exe"

if not defined BUILD_PYTHON (
    echo Leti has not been set up on this computer yet.
    echo   What failed: there is no prepared Python in this folder.
    echo   It was trying to: build Leti.exe with the same Python Leti runs on.
    echo   Double-click "Launch Leti (Windows).bat" once first - it prepares
    echo   everything - then run this again.
    echo.
    pause
    exit /b 1
)

"%BUILD_PYTHON%" launcher\build_exe.py
set "STATUS=%errorlevel%"

echo.
if not "%STATUS%"=="0" (
    echo The build did not finish ^(code %STATUS%^) - the reason is above.
    echo Leti still works: double-click "Launch Leti (Windows).bat".
    pause
    exit /b %STATUS%
)

echo Done. Double-click Leti.exe, or run this to move your shortcuts onto it:
echo   powershell -ExecutionPolicy Bypass -File scripts\install_windows_launcher.ps1
echo.
pause
exit /b 0
