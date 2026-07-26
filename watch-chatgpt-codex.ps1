param(
    [int]$PollSeconds = 10,
    [double]$LoopHours = 6,
    [int]$MaxIterations = 40
)

$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
$Remote = "origin"
$Branch = "master"
$PromptPath = "docs/codex_tasks/chatgpt_to_codex.md"
$StateRoot = Join-Path $env:LOCALAPPDATA "BasicTSCodexWatcher"
$StateFile = Join-Path $StateRoot "last_processed_blob.txt"
$LockFile = Join-Path $StateRoot "watcher.lock"
$RepairPromptFile = Join-Path $StateRoot "repair_watcher_prompt.txt"
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

function Test-GitAncestor {
    param(
        [string]$Ancestor,
        [string]$Descendant
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & git -C $Repo merge-base --is-ancestor $Ancestor $Descendant 2>&1 | Out-Null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -eq 0) {
        return $true
    }
    if ($exitCode -eq 1) {
        return $false
    }
    throw "git merge-base --is-ancestor failed with exit code $exitCode."
}

function Sync-LocalBranch {
    $currentBranch = (Invoke-Git branch --show-current | Select-Object -First 1).Trim()
    if ($currentBranch -ne $Branch) {
        throw "Expected local branch '$Branch', but found '$currentBranch'."
    }

    $status = @(Invoke-Git status --porcelain)
    if ($status.Count -gt 0) {
        throw "The working tree is not clean. Preserve or commit these changes before running the watcher:`n$($status -join [Environment]::NewLine)"
    }

    Invoke-Git fetch $Remote $Branch | Out-Null
    $localHead = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()
    $remoteHead = (Invoke-Git rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()

    if ($localHead -eq $remoteHead) {
        return
    }
    if (-not (Test-GitAncestor -Ancestor $localHead -Descendant $remoteHead)) {
        throw "Local $Branch is ahead of or diverged from $Remote/$Branch. Refusing to merge or overwrite commits."
    }

    Invoke-Git merge --ff-only "$Remote/$Branch" | Out-Null
    $syncedHead = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()
    if ($syncedHead -ne $remoteHead) {
        throw "Fast-forward verification failed: local HEAD $syncedHead does not equal remote HEAD $remoteHead."
    }
}

function Get-RemotePrompt {
    Invoke-Git fetch $Remote $Branch | Out-Null
    $remotePromptRef = "$Remote/$Branch" + ":" + $PromptPath
    $sha = (Invoke-Git rev-parse $remotePromptRef | Select-Object -First 1).Trim()
    $text = (Invoke-Git show $remotePromptRef) -join [Environment]::NewLine
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "The remote prompt file is empty."
    }
    return @{ Sha = $sha; Text = $text }
}

function Write-RepairPrompt {
    param(
        [string]$Reason,
        [string]$LogFile = "No Codex log was created."
    )

    $repairLines = @(
        "Repair the BasicTS completion-driven COSTAR-TS watcher failure.",
        "",
        "Observed failure:",
        $Reason,
        "",
        "Relevant Codex log:",
        $LogFile,
        "",
        "Repository: $Repo",
        "Watcher: watch-chatgpt-codex.ps1",
        "Target branch: $Branch",
        "",
        "Inspect the watcher, repository status, current branch, origin/$Branch, the prompt inbox, and the relevant log.",
        "Find the exact root cause. Make the smallest safe fix. Preserve unrelated changes.",
        "Parse-check the complete PowerShell file and test the failing control-flow path without launching a long research run.",
        "Do not bypass authentication, force-push, discard changes, or claim success without evidence.",
        "If the stop was intentional because no useful research change was possible, explain that and do not fabricate a commit.",
        "If a valid fix is made, run focused tests, stage only repair files, commit with a specific message, and push to origin $Branch.",
        "Report the root cause, changed files, tests, commit hash, pushed branch, and remaining blocker."
    )

    Set-Content -Path $RepairPromptFile -Value $repairLines -Encoding utf8
    Write-Warning "Repair prompt written to: $RepairPromptFile"
    Write-Host "Run it with:"
    Write-Host "  Get-Content `"$RepairPromptFile`" -Raw | codex exec -s workspace-write -C `"$Repo`""
}

function Invoke-CodexTask {
    param(
        [string]$TaskText,
        [int]$Iteration,
        [datetime]$Deadline
    )

    Sync-LocalBranch
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
        $reason = "Codex iteration $Iteration exited with code $exitCode."
        Write-Warning "$reason Log: $logFile"
        Write-RepairPrompt -Reason $reason -LogFile $logFile
        return $false
    }

    $afterHead = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()
    if ($afterHead -eq $beforeHead) {
        $reason = "Codex iteration $Iteration created no commit."
        Write-Warning "$reason Stopping the loop. Log: $logFile"
        Write-RepairPrompt -Reason $reason -LogFile $logFile
        return $false
    }

    Invoke-Git fetch $Remote $Branch | Out-Null
    $remoteHead = (Invoke-Git rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()
    if ($remoteHead -ne $afterHead) {
        $reason = "Iteration $Iteration was not pushed to $Remote/$Branch. Local: $afterHead Remote: $remoteHead."
        Write-Warning "$reason Log: $logFile"
        Write-RepairPrompt -Reason $reason -LogFile $logFile
        return $false
    }

    $remainingChanges = @(Invoke-Git status --porcelain)
    if ($remainingChanges.Count -gt 0) {
        $reason = "Iteration $Iteration left uncommitted changes:`n$($remainingChanges -join [Environment]::NewLine)"
        Write-Warning "$reason`nStopping to preserve them."
        Write-RepairPrompt -Reason $reason -LogFile $logFile
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
if ($PollSeconds -le 0) {
    throw "PollSeconds must be greater than zero."
}
if ($LoopHours -le 0) {
    throw "LoopHours must be greater than zero."
}
if ($MaxIterations -le 0) {
    throw "MaxIterations must be greater than zero."
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

try {
    $lockStream = [System.IO.File]::Open(
        $LockFile,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another BasicTS Codex watcher is already running. Stop it before starting a new watcher."
}

try {
    Write-Host "Watching GitHub prompt inbox:"
    Write-Host "  ${Remote}/${Branch}:$PromptPath"
    Write-Host "Polling every $PollSeconds seconds."
    Write-Host "A detected prompt starts a completion-driven loop lasting up to $LoopHours hours."
    Write-Host "Only one watcher and one Codex iteration may run at a time."
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

                Sync-LocalBranch
                Set-Content -Path $StateFile -Value $prompt.Sha -Encoding utf8

                $deadline = (Get-Date).AddHours($LoopHours)
                $iteration = 1
                $taskText = $prompt.Text

                Write-Host "Loop deadline: $($deadline.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
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

                $finishedAt = Get-Date
                Write-Host "Research loop ended after $iteration iteration(s) at $($finishedAt.ToString('yyyy-MM-dd HH:mm:ss zzz'))."
                Write-Host "Deadline was $($deadline.ToString('yyyy-MM-dd HH:mm:ss zzz'))."
                Write-Host "It stops between tasks at the deadline; it does not terminate a Codex process mid-task."
            }
        }
        catch {
            Write-Warning $_
            Write-RepairPrompt -Reason $_.Exception.Message
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
}
