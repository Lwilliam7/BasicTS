param(
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
$Remote = "origin"
$Branch = "master"
$PromptPath = "docs/codex_tasks/chatgpt_to_codex.md"
$StateRoot = Join-Path $env:LOCALAPPDATA "BasicTSCodexWatcher"
$StateFile = Join-Path $StateRoot "last_processed_blob.txt"
$LogDir = Join-Path $StateRoot "logs"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $output = & git -C $Repo @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }

    return $output
}

Require-Command "git"
Require-Command "codex"

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    throw "This script must be run from the BasicTS Git repository."
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Watching GitHub prompt inbox:"
Write-Host "  ${Remote}/${Branch}:$PromptPath"
Write-Host "Polling every $PollSeconds seconds."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    try {
        Invoke-Git fetch $Remote $Branch | Out-Null

        $remoteRef = "${Remote}/${Branch}:$PromptPath"
        $blobSha = (Invoke-Git rev-parse $remoteRef | Select-Object -First 1).Trim()

        $lastProcessed = ""
        if (Test-Path $StateFile) {
            $lastProcessed = (Get-Content $StateFile -Raw).Trim()
        }

        if ($blobSha -ne $lastProcessed) {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] New prompt detected: $blobSha"

            $taskText = (Invoke-Git show $remoteRef) -join [Environment]::NewLine

            # Mark the prompt as consumed before launching Codex so a partial or failed
            # run is not started repeatedly. Editing the GitHub inbox again retriggers it.
            Set-Content -Path $StateFile -Value $blobSha -Encoding utf8

            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $logFile = Join-Path $LogDir "codex_$timestamp.log"

            $launcherPrompt = @"
A new task was received through the ChatGPT-to-Codex prompt inbox.

Treat the stdin block as the complete active task specification.

Requirements:
1. Inspect the current BasicTS repository and git diff before editing.
2. Preserve unrelated user changes and existing working routers.
3. Implement the task instead of only describing a plan.
4. Run relevant available tests and report their real results.
5. Do not claim completion for work or tests that were not performed.
6. Never use the final test split for tuning.
7. Keep forecasting experts frozen when the task concerns router training.
8. At the end, summarize every changed file, command run, test result, and blocker.
"@

            Write-Host "Launching Codex. Log: $logFile"

            $taskText |
                & codex exec -s workspace-write -C $Repo $launcherPrompt 2>&1 |
                Tee-Object -FilePath $logFile

            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0) {
                Write-Host "Codex finished successfully."
            }
            else {
                Write-Warning "Codex exited with code $exitCode. Edit the GitHub prompt inbox to trigger another run."
            }
        }
    }
    catch {
        Write-Warning $_
    }

    Start-Sleep -Seconds $PollSeconds
}
