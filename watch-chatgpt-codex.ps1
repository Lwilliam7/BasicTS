param(
    [int]$PollSeconds = 10,
    [double]$LoopHours = 6,
    [int]$MaxIterations = 40,
    [string]$Remote = "origin",
    [string]$Branch = "master",
    [string]$PromptPath = "docs/codex_tasks/chatgpt_to_codex.md",
    [string]$StateRoot = (Join-Path $env:LOCALAPPDATA "BasicTSCodexWatcher"),
    [string]$RunnerRoot = "",
    [string]$RemoteUrl = "",
    [string]$CodexCommand = "codex",
    [switch]$RunOnce
)

$ErrorActionPreference = "Stop"

$LauncherRepo = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RunnerRoot)) {
    $RunnerRoot = Join-Path $StateRoot "runner"
}

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

function Invoke-GitCommand {
    param(
        [string]$WorkingDirectory,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & git -C $WorkingDirectory @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "git -C `"$WorkingDirectory`" $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Invoke-RunnerGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    return Invoke-GitCommand $RunnerRoot @Arguments
}

function Test-GitAncestor {
    param(
        [string]$WorkingDirectory,
        [string]$Ancestor,
        [string]$Descendant
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & git -C $WorkingDirectory merge-base --is-ancestor $Ancestor $Descendant 2>&1 | Out-Null
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
    throw "git -C `"$WorkingDirectory`" merge-base --is-ancestor $Ancestor $Descendant failed with exit code $exitCode."
}

function Get-RunnerStatus {
    return @(Invoke-RunnerGit status --porcelain)
}

function Assert-CleanRunner {
    param([string]$Reason)

    $status = Get-RunnerStatus
    if ($status.Count -gt 0) {
        throw "$Reason Runner checkout is dirty; refusing to overwrite uncommitted work in '$RunnerRoot':`n$($status -join [Environment]::NewLine)"
    }
}

function Get-RemoteUrl {
    if (-not [string]::IsNullOrWhiteSpace($RemoteUrl)) {
        return $RemoteUrl
    }

    if (-not (Test-Path (Join-Path $LauncherRepo ".git"))) {
        throw "The launcher directory '$LauncherRepo' is not a Git repository. Pass -RemoteUrl explicitly."
    }

    return (Invoke-GitCommand $LauncherRepo remote get-url $Remote | Select-Object -First 1).Trim()
}

function Ensure-RunnerClone {
    $runnerGitDir = Join-Path $RunnerRoot ".git"
    if (Test-Path $runnerGitDir) {
        return
    }

    if (Test-Path $RunnerRoot) {
        $existingEntries = @(Get-ChildItem -Force -LiteralPath $RunnerRoot)
        if ($existingEntries.Count -gt 0) {
            throw "Runner path '$RunnerRoot' exists but is not a Git repository. Move it aside or pass a different -RunnerRoot."
        }
    }

    $runnerParent = Split-Path -Parent $RunnerRoot
    New-Item -ItemType Directory -Force -Path $runnerParent | Out-Null

    $url = Get-RemoteUrl
    Write-Host "Creating isolated runner clone:"
    Write-Host "  $RunnerRoot"
    Write-Host "  source: $url"
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & git clone --branch $Branch $url $RunnerRoot 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "git clone --branch $Branch $url `"$RunnerRoot`" failed:`n$($output -join [Environment]::NewLine)"
    }
}

function Sync-RunnerBranch {
    Ensure-RunnerClone

    Invoke-RunnerGit remote set-url $Remote (Get-RemoteUrl) | Out-Null
    Invoke-RunnerGit fetch --prune $Remote $Branch | Out-Null

    $currentBranch = (Invoke-RunnerGit branch --show-current | Select-Object -First 1).Trim()
    if ($currentBranch -ne $Branch) {
        Assert-CleanRunner "Cannot switch runner branch from '$currentBranch' to '$Branch'."
        $localBranch = @(Invoke-RunnerGit branch --list $Branch)
        if ($localBranch.Count -gt 0) {
            Invoke-RunnerGit switch $Branch | Out-Null
        }
        else {
            Invoke-RunnerGit switch --track -c $Branch "$Remote/$Branch" | Out-Null
        }
    }

    $localHead = (Invoke-RunnerGit rev-parse HEAD | Select-Object -First 1).Trim()
    $remoteHead = (Invoke-RunnerGit rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()

    if ($localHead -eq $remoteHead) {
        Assert-CleanRunner "Runner is on $Branch at $Remote/$Branch, but cannot start."
        return
    }

    if (Test-GitAncestor $RunnerRoot $localHead $remoteHead) {
        Assert-CleanRunner "Cannot fast-forward runner $Branch to $Remote/$Branch."
        Invoke-RunnerGit merge --ff-only "$Remote/$Branch" | Out-Null
        $syncedHead = (Invoke-RunnerGit rev-parse HEAD | Select-Object -First 1).Trim()
        if ($syncedHead -ne $remoteHead) {
            throw "Fast-forward verification failed in runner: local HEAD $syncedHead does not equal $Remote/$Branch $remoteHead."
        }
        return
    }

    if (Test-GitAncestor $RunnerRoot $remoteHead $localHead) {
        throw "Runner $Branch is ahead of $Remote/$Branch at $localHead. Refusing to overwrite or force-push unpushed commits in '$RunnerRoot'."
    }

    throw "Runner $Branch has diverged from $Remote/$Branch. Refusing to merge, reset, or overwrite '$RunnerRoot'."
}

function Get-QueuedPrompt {
    Sync-RunnerBranch

    $promptFile = Join-Path $RunnerRoot $PromptPath
    if (-not (Test-Path $promptFile)) {
        throw "The queued prompt file does not exist in the runner: $promptFile"
    }

    $sha = (Invoke-RunnerGit rev-parse "HEAD:$PromptPath" | Select-Object -First 1).Trim()
    $text = Get-Content -Path $promptFile -Raw
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "The queued prompt file is empty: $PromptPath"
    }

    return @{ Sha = $sha; Text = $text }
}

function Write-RepairPrompt {
    param(
        [string]$Reason,
        [string]$LogFile = "No Codex log was created."
    )

    New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
    $repairLines = @(
        "Repair the BasicTS completion-driven COSTAR-TS watcher failure.",
        "",
        "Exact error:",
        $Reason,
        "",
        "Exact log path:",
        $LogFile,
        "",
        "Launcher checkout:",
        $LauncherRepo,
        "",
        "Isolated runner checkout:",
        $RunnerRoot,
        "",
        "Watcher:",
        "watch-chatgpt-codex.ps1",
        "",
        "Target branch:",
        "$Remote/$Branch",
        "",
        "Prompt inbox:",
        $PromptPath,
        "",
        "Inspect the watcher, runner status, current branch, $Remote/$Branch, the queued prompt, and the relevant log.",
        "Find the exact root cause. Make the smallest safe fix. Preserve unrelated changes in both checkouts.",
        "Parse-check the complete PowerShell file and test the failing control-flow path without launching a long research run.",
        "Do not bypass authentication, force-push, discard changes, run git reset --hard, run git clean, or claim success without evidence.",
        "If the stop was intentional because no useful research change was possible, explain that and do not fabricate a commit.",
        "If a valid fix is made, run focused tests, stage only repair files, commit with a specific message, and push to origin $Branch.",
        "Report the root cause, changed files, tests, commit hash, pushed branch, and remaining blocker."
    )

    Set-Content -Path $RepairPromptFile -Value $repairLines -Encoding utf8
    Write-Warning "Repair prompt written to: $RepairPromptFile"
    Write-Host "Retry with:"
    Write-Host "  Get-Content `"$RepairPromptFile`" -Raw | codex exec -s workspace-write -C `"$RunnerRoot`""
}

function Fail-Iteration {
    param(
        [string]$Reason,
        [string]$LogFile
    )

    Write-Warning "$Reason Log: $LogFile"
    Write-RepairPrompt -Reason $Reason -LogFile $LogFile
    return $false
}

function Quote-ProcessArgument {
    param([AllowNull()][string]$Argument)

    if ($null -eq $Argument) {
        return '""'
    }
    if ($Argument.Length -eq 0) {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append('\' * ($backslashCount * 2 + 1))
            [void]$builder.Append('"')
            $backslashCount = 0
            continue
        }
        if ($backslashCount -gt 0) {
            [void]$builder.Append('\' * $backslashCount)
            $backslashCount = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashCount -gt 0) {
        [void]$builder.Append('\' * ($backslashCount * 2))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Get-PowerShellExecutable {
    $currentProcessPath = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    if (-not [string]::IsNullOrWhiteSpace($currentProcessPath) -and (Test-Path $currentProcessPath)) {
        return $currentProcessPath
    }

    $windowsPowerShell = Join-Path $PSHOME "powershell.exe"
    if (Test-Path $windowsPowerShell) {
        return $windowsPowerShell
    }

    $command = Get-Command "powershell.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    throw "Could not find a PowerShell executable to launch '$CodexCommand'."
}

function Invoke-CodexProcess {
    param(
        [string]$TaskText,
        [string]$LauncherPrompt,
        [string]$LogFile
    )

    $commandInfo = Get-Command $CodexCommand -ErrorAction Stop | Select-Object -First 1
    $commandPath = $commandInfo.Source
    if ([string]::IsNullOrWhiteSpace($commandPath)) {
        $commandPath = $commandInfo.Path
    }
    if ([string]::IsNullOrWhiteSpace($commandPath)) {
        $commandPath = $CodexCommand
    }

    $codexArguments = @("exec", "-s", "workspace-write", "-C", $RunnerRoot, $LauncherPrompt)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    if ([System.IO.Path]::GetExtension($commandPath) -ieq ".ps1") {
        $psi.FileName = Get-PowerShellExecutable
        $psi.Arguments = (@(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $commandPath
        ) + $codexArguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " "
    }
    else {
        $psi.FileName = $commandPath
        $psi.Arguments = ($codexArguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " "
    }
    $psi.WorkingDirectory = $RunnerRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.Write($TaskText)
        $process.StandardInput.Close()
        $process.WaitForExit()
        $nativeExitCode = $process.ExitCode

        $combinedOutput = @()
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        if (-not [string]::IsNullOrEmpty($stdout)) {
            $combinedOutput += $stdout.TrimEnd("`r", "`n")
        }
        if (-not [string]::IsNullOrEmpty($stderr)) {
            $combinedOutput += $stderr.TrimEnd("`r", "`n")
        }

        Set-Content -Path $LogFile -Value $combinedOutput -Encoding utf8
        foreach ($line in $combinedOutput) {
            if (-not [string]::IsNullOrEmpty($line)) {
                Write-Host $line
            }
        }

        return $nativeExitCode
    }
    finally {
        if ($null -ne $process) {
            $process.Dispose()
        }
    }
}

function Invoke-CodexTask {
    param(
        [string]$TaskText,
        [int]$Iteration,
        [datetime]$Deadline
    )

    Sync-RunnerBranch
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $LogDir "codex_$($timestamp)_iteration_$Iteration.log"
    $beforeHead = (Invoke-RunnerGit rev-parse HEAD | Select-Object -First 1).Trim()

    $launcherPrompt = @"
You are iteration $Iteration of a bounded COSTAR-TS research loop.
The loop deadline is $($Deadline.ToString("o")). Do one focused task only.

Treat the stdin block as the complete active task specification.

Requirements:
1. Inspect the current BasicTS runner repository, recent COSTAR-TS commits, git status, and git diff before editing.
2. Preserve unrelated user changes and existing working routers.
3. Implement the task instead of only describing a plan.
4. Run relevant available tests and report their real results.
5. Never use the final test split for tuning.
6. Keep forecasting experts frozen when the task concerns router training.
7. Prefer correctness, leakage prevention, baselines, ablations, multi-seed evidence, and cost-accuracy evaluation over adding architecture.
8. Make only one bounded, evidence-driven improvement in this iteration.
9. If no safe useful improvement can be completed, stop without making a commit and explain the blocker.
10. After implementation and testing, review the exact diff, stage only task-related files, commit with a task-specific message, and push to $Remote $Branch.
11. Never force-push. If authentication, conflicts, tests, or branch protection block completion, report the exact error and stop.
12. End with changed files, commands, test results, commit hash, pushed branch, and blockers.
"@

    Write-Host "Launching Codex iteration $Iteration in isolated runner."
    Write-Host "Runner: $RunnerRoot"
    Write-Host "Log: $logFile"
    $exitCode = Invoke-CodexProcess -TaskText $TaskText -LauncherPrompt $launcherPrompt -LogFile $logFile
    if ($exitCode -ne 0) {
        return Fail-Iteration "Codex iteration $Iteration exited with code $exitCode." $logFile
    }

    $afterHead = (Invoke-RunnerGit rev-parse HEAD | Select-Object -First 1).Trim()
    if ($afterHead -eq $beforeHead) {
        return Fail-Iteration "Codex iteration $Iteration created no commit." $logFile
    }

    Invoke-RunnerGit fetch $Remote $Branch | Out-Null
    $remoteHead = (Invoke-RunnerGit rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()
    if ($remoteHead -ne $afterHead) {
        return Fail-Iteration "Iteration $Iteration was not pushed to $Remote/$Branch. Local: $afterHead Remote: $remoteHead." $logFile
    }

    $remainingChanges = Get-RunnerStatus
    if ($remainingChanges.Count -gt 0) {
        return Fail-Iteration "Iteration $Iteration left uncommitted runner changes:`n$($remainingChanges -join [Environment]::NewLine)" $logFile
    }

    Write-Host "Codex iteration $Iteration completed and pushed commit $afterHead."
    return $true
}

function Invoke-PromptLoop {
    param([hashtable]$Prompt)

    $deadline = (Get-Date).AddHours($LoopHours)
    $iteration = 1
    $completedIterations = 0
    $taskText = $Prompt.Text

    Write-Host "Loop deadline: $($deadline.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
    while ((Get-Date) -lt $deadline -and $iteration -le $MaxIterations) {
        $completed = Invoke-CodexTask -TaskText $taskText -Iteration $iteration -Deadline $deadline
        if (-not $completed) {
            Write-Warning "Prompt $($Prompt.Sha) was not marked processed and will be retried."
            return $false
        }

        $completedIterations++
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

    Set-Content -Path $StateFile -Value $Prompt.Sha -Encoding utf8
    $finishedAt = Get-Date
    Write-Host "Research loop ended after $completedIterations completed iteration(s) at $($finishedAt.ToString('yyyy-MM-dd HH:mm:ss zzz'))."
    Write-Host "Deadline was $($deadline.ToString('yyyy-MM-dd HH:mm:ss zzz'))."
    Write-Host "The watcher stops between tasks at the deadline; it does not terminate a Codex process mid-task."
    return $true
}

function Invoke-WatcherCycle {
    $prompt = Get-QueuedPrompt
    $lastProcessed = ""
    if (Test-Path $StateFile) {
        $lastProcessed = (Get-Content $StateFile -Raw).Trim()
    }

    if ($prompt.Sha -eq $lastProcessed) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] No new prompt detected."
        return $true
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] New prompt detected: $($prompt.Sha)"
    return Invoke-PromptLoop $prompt
}

Require-Command "git"
Require-Command $CodexCommand

if ($PollSeconds -le 0) {
    throw "PollSeconds must be greater than zero."
}
if ($LoopHours -le 0) {
    throw "LoopHours must be greater than zero."
}
if ($MaxIterations -le 0) {
    throw "MaxIterations must be greater than zero."
}
if ([string]::IsNullOrWhiteSpace($StateRoot)) {
    throw "StateRoot must not be empty."
}
if ([string]::IsNullOrWhiteSpace($RunnerRoot)) {
    throw "RunnerRoot must not be empty."
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$lockStream = $null
$hadFailure = $false
try {
    $lockStream = [System.IO.File]::Open(
        $LockFile,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another BasicTS Codex watcher is already running for state root '$StateRoot'. Stop it before starting a new watcher."
}

try {
    Write-Host "Watching prompt inbox from isolated runner:"
    Write-Host "  ${Remote}/${Branch}:$PromptPath"
    Write-Host "Runner:"
    Write-Host "  $RunnerRoot"
    Write-Host "Polling every $PollSeconds seconds."
    Write-Host "A detected prompt starts a completion-driven loop lasting up to $LoopHours hours."
    Write-Host "Only one watcher and one Codex iteration may run at a time."
    Write-Host "Press Ctrl+C to stop."
    Write-Host ""

    while ($true) {
        try {
            $cycleSucceeded = Invoke-WatcherCycle
            if (-not $cycleSucceeded) {
                $hadFailure = $true
                break
            }
        }
        catch {
            $hadFailure = $true
            $reason = $_.Exception.Message
            Write-Warning $reason
            Write-RepairPrompt -Reason $reason
            break
        }

        if ($RunOnce) {
            break
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
}

if ($RunOnce) {
    if ($hadFailure) {
        exit 1
    }
    exit 0
}

if ($hadFailure) {
    exit 1
}
