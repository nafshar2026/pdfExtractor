# ===========================================================================
# pdf-extractor - Local mode launcher
# ===========================================================================
#
# DEPRECATION NOTICE: Use .\run.ps1 instead (unified local + Azure launcher)
#
# This script is kept for backwards compatibility with existing workflows.
# It delegates to the unified run.ps1 script with -Mode local.
#
# USAGE:
#   .\run-local.ps1                              # Interactive menu
#   .\run-local.ps1 -FileName Sample-1.pdf       # Single file
#   .\run-local.ps1 -All                         # All files in batch
#   .\run-local.ps1 -FileName Sample-1.pdf -WithOptIn  # + opt-in extraction
#
# NEW WAY (preferred):
#   .\run.ps1                                    # Choose mode interactively
#   .\run.ps1 -Mode local                        # Force local mode
#   .\run.ps1 -Mode local -File Sample-1.pdf -OptIn
#
# ===========================================================================

param(
    [string]$FileName,
    [switch]$All,
    [switch]$WithOptIn
)

# Delegate to unified run.ps1 with local mode
$args = @("-Mode", "local")

if ($FileName) {
    $args += @("-File", $FileName)
}

if ($WithOptIn) {
    $args += "-OptIn"
}

# Note: -All flag will be handled by interactive menu when no -File is specified
# The unified script's batch processing can be accessed via menu option 2 in local mode

Write-Host "Launching pdf-extractor in LOCAL mode..." -ForegroundColor Cyan
Write-Host ""

& ".\run.ps1" @args
