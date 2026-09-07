# Run this ONCE after extracting/cloning Leti on Windows, from this folder:
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_launcher.ps1
#
# A .bat file can't carry an icon - Windows always draws the generic script
# icon for one, no matter what you do to the file. The icon has to live on a
# shortcut (.lnk) pointing at it. This creates those shortcuts, on the Desktop
# and in the Start Menu, with Leti's icon and the project folder baked in as
# the working directory.
#
# The Linux equivalent is scripts/install_linux_launcher.sh; macOS is
# scripts/install_macos_icon.sh.

$ErrorActionPreference = 'Stop'

# This script's own folder's parent is the project root, regardless of where it
# was invoked from.
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Target     = Join-Path $ProjectDir 'Launch Leti (Windows).bat'
$IconPath   = Join-Path $ProjectDir 'gui\icons\leti.ico'

if (-not (Test-Path $Target))   { throw "Can't find '$Target' - run this from inside the Leti project folder." }
if (-not (Test-Path $IconPath)) { throw "Can't find '$IconPath'. Regenerate it with: python scripts\build_icons.py" }

function New-LetiShortcut {
    param([string]$LinkPath)

    $shell    = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath       = $Target
    $shortcut.WorkingDirectory = $ProjectDir
    $shortcut.IconLocation     = "$IconPath,0"
    $shortcut.Description      = "Launch Leti's interface"
    $shortcut.Save()
    Write-Host "Created: $LinkPath"
}

$desktop = [Environment]::GetFolderPath('Desktop')
if ($desktop) { New-LetiShortcut (Join-Path $desktop 'Leti.lnk') }

$startMenu = Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'Microsoft\Windows\Start Menu\Programs'
if (Test-Path $startMenu) { New-LetiShortcut (Join-Path $startMenu 'Leti.lnk') }

Write-Host ''
Write-Host "Done. Double-click 'Leti' on your Desktop, or search for it in the Start Menu."
Write-Host 'The first launch sets up leti_env\ and downloads models, so it takes a while;'
Write-Host 'after that it opens straight into the interface.'
