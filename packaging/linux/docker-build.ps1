# Builds the Linux packages from Windows with Docker Desktop, on Debian 11 (glibc 2.31) so they
# run on most current distributions. Output goes to dist\.
#   powershell -ExecutionPolicy Bypass -File packaging\linux\docker-build.ps1
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
docker version --format '{{.Server.Version}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker is not running. Start Docker Desktop and try again.' }
# Named volumes keep the cargo registry and Linux build cache between runs, away from the
# Windows target folder.
docker run --rm `
  -v "${root}:/src" `
  -v prism-linux-cargo:/usr/local/cargo/registry `
  -v prism-linux-target:/target `
  -e CARGO_TARGET_DIR=/target `
  -w /src `
  rust:1-bullseye `
  bash packaging/linux/build.sh
if ($LASTEXITCODE -ne 0) { throw "Linux build failed ($LASTEXITCODE)" }
