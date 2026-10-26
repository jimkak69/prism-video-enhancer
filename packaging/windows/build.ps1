# Builds the Windows installer and a portable zip into dist\:
#   PrismVideoEnhancer-<version>-windows-x64-setup.exe     (per-user installer, no admin rights)
#   PrismVideoEnhancer-<version>-windows-x64-portable.zip  (unzip and run prism-app.exe)
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
#
# FFmpeg is not bundled; the installer checks for it and shows the winget command. ONNX Runtime
# is not bundled either: Windows 11 ships one with DirectML, which Prism uses for GPU inference.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value

cargo build --profile dist -p prism-app -p prism-cli -p prism-installer
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$bin = Join-Path $root 'target\dist'

$stage = Join-Path $root 'dist\windows-payload'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item "$bin\prism-app.exe", "$bin\prism.exe", 'README.md' $stage
# Both programs are windowed (GUI subsystem): PowerShell would not wait for them with `&`.
function Invoke-Gui($exe, [string[]]$arguments) {
  $p = Start-Process -FilePath $exe -ArgumentList $arguments -Wait -PassThru -NoNewWindow
  if ($p.ExitCode -ne 0) { throw "$exe $arguments failed ($($p.ExitCode))" }
}
$icons = Join-Path $root 'dist\icons'
Invoke-Gui "$bin\prism-app.exe" @('--write-icons', "`"$icons`"")
Copy-Item "$icons\prism.ico" $stage

$setup = Join-Path $root "dist\PrismVideoEnhancer-$version-windows-x64-setup.exe"
Invoke-Gui "$bin\prism-setup.exe" @('--make-installer', "`"$stage`"", "`"$setup`"")

$zip = Join-Path $root "dist\PrismVideoEnhancer-$version-windows-x64-portable.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip

Get-Item $setup, $zip | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
