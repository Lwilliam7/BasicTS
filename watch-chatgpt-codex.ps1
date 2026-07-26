param(
    [int]$PollSeconds = 10,
    [double]$LoopHours = 2,
    [int]$MaxIterations = 20
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

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & git -C $Repo @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Get-RemotePrompt {
    Invoke-Git fetch $Remote $Branch | Out-Null
    $remotePromptRef = "$Remote/$Branch" + ":" + $PromptPath
    $sha = (Invoke-Git rev-parse $remotePromptRef | Select-Object -First 1).Trim()
    $text = (Invoke-Git show $remotePromptRef) -join [Environment]::NewLine
    return @{ Sha = $sha; Text = $text }
}

function Invoke-CodexTask {
    param(
        [string]$TaskText,
        [int]$Iteration,
        [datetime]$Deadline
    )

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $LogDir "codex_$($timestamp)_iteration_$Iteration.log"
    $beforeHead = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()

    $launcherPrompt = @"
You are iteration $Iteration of a bounded COSTAR-TS research loop.
The loop deadline is $($Deadline.ToString("o")). Do one focused task only.

Treat the stdin block as the complete active task specification.

Requirements:
1. Inspect the current BasicTS repository, recent COSTAR-TS commits, git status, and git diff before editing.
2. Preserve unrelated user changes and existing working routers.
3. Implement the task instead of only describing a plan.
4. Run relevant available tests and report their real results.
5. Never use the final test split for tuning.
6. Keep forecasting experts frozen when the task concerns router training.
7. Prefer correctness, leakage prevention, baselines, ablations, multi-seed evidence, and cost-accuracy evaluation over adding architecture.
8. Make only one bounded, evidence-driven improvement in this iteration.
9. If no safe useful improvement can be completed, stop without making a commit and explain the blocker.
10. After implementation and testing, review the exact diff, stage only task-related files, commit with a task-specific message, and push to origin $Branch.
11. Never force-push. If authentication, conflicts, tests, or branch protection block completion, report the exact error and stop.
12. End with changed files, commands, test results, commit hash, pushed branch, and blockers.
"@

    Write-Host "Launching Codex iteration $Iteration. Log: $logFile"
    $TaskText |
        & codex exec -s workspace-write -C $Repo $launcherPrompt 2>&1 |
        Tee-Object -FilePath $logFile

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Warning "Codex iteration $Iteration exited with code $exitCode."
        return $false
    }

    $afterHead = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()
    if ($afterHead -eq $beforeHead) {
        Write-Warning "Codex iteration $Iteration created no commit. Stopping the loop."
        return $false
    }

    Invoke-Git fetch $Remote $Branch | Out-Null
    $remoteHead = (Invoke-Git rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()
    if ($remoteHead -ne $afterHead) {
        Write-Warning "Iteration $Iteration was not pushed to $Remote/$Branch. Stopping the loop."
        return $false
    }

    Write-Host "Codex iteration $Iteration completed and pushed commit $afterHead."
    return $true
}

Require-Command "git"
Require-Command "codex"

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    throw "This script must be run from the BasicTS Git repository."
}
if ($LoopHours -le 0) {
    throw "LoopHours must be greater than zero."
}
if ($MaxIterations -le 0) {
    throw "MaxIterations must be greater than zero."
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Watching GitHub prompt inbox:"
Write-Host "  ${Remote}/${Branch}:$PromptPath"
Write-Host "Polling every $PollSeconds seconds."
Write-Host "A detected prompt starts a completion-driven loop lasting up to $LoopHours hours."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    try {
        $prompt = Get-RemotePrompt
        $lastProcessed = ""
        if (Test-Path $StateFile) {
            $lastProcessed = (Get-Content $StateFile -Raw).Trim()
        }

        if ($prompt.Sha -ne $lastProcessed) {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] New prompt detected: $($prompt.Sha)"
            Set-Content -Path $StateFile -Value $prompt.Sha -Encoding utf8

            $deadline = (Get-Date).AddHours($LoopHours)
            $iteration = 1
            $taskText = $prompt.Text

            while ((Get-Date) -lt $deadline -and $iteration -le $MaxIterations) {
                $completed = Invoke-CodexTask -TaskText $taskText -Iteration $iteration -Deadline $deadline
                if (-not $completed) {
                    break
                }
                if ((Get-Date) -ge $deadline) {
                    break
                }

                $iteration++
                $taskText = @"
Inspect the latest pushed COSTAR-TS changes and their real test or experiment evidence.
Identify the single biggest remaining problem that can be safely addressed in one bounded iteration.
Implement and test that improvement now.

Do not assume more architecture is better. Check first for correctness bugs, data leakage,
train-validation-test misuse, finalizer mismatch, weak or missing baselines, stopping-policy
failures, cost accounting errors, missing ablations, seed instability, and absent robustness
evidence. Choose the largest evidence-backed weakness. Do not use the final test split for
selection or tuning. If the next useful step requires major compute, unavailable data, or a
research choice from the user, make no commit and report the blocker.
"@
            }

            Write-Host "Research loop ended after $iteration iteration(s)."
            Write-Host "It stops between tasks at the deadline; it does not terminate a Codex process mid-task."
        }
    }
    catch {
        Write-Warning $_
    }

    Start-Sleep -Seconds $PollSeconds
}
