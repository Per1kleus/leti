# Put Leti on your Desktop and in your Start Menu.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_launcher.ps1
#
# You do not normally need this. Both launchers place the shortcuts themselves the
# first time they set the folder up. Run it when one has been deleted or has
# stopped working, or after building Leti.exe for a folder that only had the .bat
# - which repairs the existing shortcut to point at the executable rather than
# leaving you with two.
#
# It decides nothing itself. launcher/shortcuts.py knows which paths, which
# target, which icon and whether anything needs doing, because that is also what
# the launchers use, and one answer to "where does Leti's shortcut go" is the
# point. This file finds an interpreter to ask.
#
# Per-user throughout: your own Desktop, your own Start Menu. No administrator
# prompt, no registry, nothing in Program Files.
#
# The Linux equivalent is scripts/install_linux_launcher.sh; macOS is
# scripts/install_macos_icon.sh.

$ErrorActionPreference = 'Stop'

# This script's own folder's parent is the project root, wherever it was invoked
# from. -LiteralPath throughout, so a folder with a bracket in its name is a
# folder and not a pattern.
$ProjectDir = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir 'main.py'))) {
    throw "Can't find main.py in '$ProjectDir' - run this from inside the Leti project folder."
}

# The interpreter one of the launchers already prepared, or any system one. Only
# needed to run launcher/shortcuts.py, which is standard library only - so an
# old Python that Leti itself could not use is still fine for this.
$Candidates = @(
    (Join-Path $ProjectDir 'leti_env\Scripts\python.exe'),
    (Join-Path $ProjectDir 'leti_runtime\python.exe')
)
$Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $Python) {
    foreach ($name in @('py.exe', 'python.exe', 'python3.exe')) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { $Python = $found.Source; break }
    }
}

if (-not $Python) {
    Write-Host "No Python here yet, so there is nothing to ask."
    Write-Host ""
    Write-Host "Double-click 'Launch Leti (Windows).bat' once - it prepares everything"
    Write-Host "and places these shortcuts itself. Then you will not need this script."
    exit 1
}

# & with an argument list, not an interpolated command string: the project path
# can contain spaces, an apostrophe or an ampersand, and none of those should
# become syntax.
& $Python (Join-Path $ProjectDir 'launcher\leti_launcher.py') '--install-shortcuts'
$status = $LASTEXITCODE

if ($status -ne 0) {
    Write-Host ""
    Write-Host "Some shortcuts could not be written. Everything else about Leti is"
    Write-Host "unaffected - the launchers in this folder still work."
    exit $status
}

Write-Host ""
Write-Host "Done. Double-click 'Leti' on your Desktop, or search for it in the Start Menu."
Write-Host "The first launch prepares Python and the packages, so it takes a while;"
Write-Host "after that it opens straight into the interface."
