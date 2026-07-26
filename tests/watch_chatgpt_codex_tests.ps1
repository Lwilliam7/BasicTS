$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WatcherPath = Join-Path $RepoRoot "watch-chatgpt-codex.ps1"
$PromptPath = "docs/codex_tasks/chatgpt_to_codex.md"
$PowerShellExe = (Get-Process -Id $PID).Path

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-Git {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments
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

function Invoke-GitRaw {
    param([string[]]$Arguments)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & git @Arguments 2>&1
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

function New-Fixture {
    $root = Join-Path ([System.IO.Path]::GetTempPath()) "basicts-watcher-tests-$([guid]::NewGuid().ToString('N'))"
    $origin = Join-Path $root "origin.git"
    $seed = Join-Path $root "seed"
    $state = Join-Path $root "state"
    $fakeCodex = Join-Path $root "fake-codex.ps1"
    $fakeLog = Join-Path $root "fake-codex.log"

    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Invoke-GitRaw @("init", "--bare", $origin) | Out-Null
    Invoke-GitRaw @("clone", $origin, $seed) | Out-Null

    Invoke-Git $seed @("checkout", "-b", "master") | Out-Null
    Invoke-Git $seed @("config", "user.email", "watcher-test@example.com") | Out-Null
    Invoke-Git $seed @("config", "user.name", "Watcher Test") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $seed "docs/codex_tasks") | Out-Null
    Set-Content -Path (Join-Path $seed $PromptPath) -Value "initial queued prompt" -Encoding utf8
    Invoke-Git $seed @("add", $PromptPath) | Out-Null
    Invoke-Git $seed @("commit", "-m", "seed prompt") | Out-Null
    Invoke-Git $seed @("push", "origin", "master") | Out-Null

    $fakeCodexBody = @'
$ErrorActionPreference = "Continue"

$repoIndex = [Array]::IndexOf($args, "-C")
if ($repoIndex -lt 0 -or $repoIndex + 1 -ge $args.Count) {
    Write-Host "fake codex did not receive -C"
    exit 64
}

$repo = $args[$repoIndex + 1]
Add-Content -Path $env:FAKE_CODEX_LOG -Value "start $repo $(Get-Date -Format o)"

if ($env:FAKE_CODEX_MODE -eq "fail") {
    Write-Host "fake codex failure requested"
    exit 42
}

if (-not [string]::IsNullOrWhiteSpace($env:FAKE_CODEX_SLEEP_MS)) {
    Start-Sleep -Milliseconds ([int]$env:FAKE_CODEX_SLEEP_MS)
}

& git -C $repo config user.email "watcher-test@example.com" | Out-Null
& git -C $repo config user.name "Watcher Test" | Out-Null

$existing = @(Get-ChildItem -Path $repo -Filter "fake_iteration_*.txt" -ErrorAction SilentlyContinue).Count
$iteration = $existing + 1
$fileName = "fake_iteration_$iteration.txt"
Set-Content -Path (Join-Path $repo $fileName) -Value "fake iteration $iteration" -Encoding utf8
& git -C $repo add $fileName | Out-Null
& git -C $repo commit -m "fake codex iteration $iteration" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "fake codex commit failed"
    exit 65
}
& git -C $repo push origin master | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "fake codex push failed"
    exit 66
}

Add-Content -Path $env:FAKE_CODEX_LOG -Value "finish $iteration $(Get-Date -Format o)"
exit 0
'@
    Set-Content -Path $fakeCodex -Value $fakeCodexBody -Encoding utf8

    return @{
        Root = $root
        Origin = $origin
        Seed = $seed
        State = $state
        Runner = (Join-Path $state "runner")
        FakeCodex = $fakeCodex
        FakeLog = $fakeLog
    }
}

function Update-RemotePrompt {
    param(
        [hashtable]$Fixture,
        [string]$Text
    )

    Invoke-Git $Fixture.Seed @("pull", "--ff-only", "origin", "master") | Out-Null
    Set-Content -Path (Join-Path $Fixture.Seed $PromptPath) -Value $Text -Encoding utf8
    Invoke-Git $Fixture.Seed @("add", $PromptPath) | Out-Null
    Invoke-Git $Fixture.Seed @("commit", "-m", "update prompt") | Out-Null
    Invoke-Git $Fixture.Seed @("push", "origin", "master") | Out-Null
    return (Invoke-Git $Fixture.Seed @("rev-parse", "HEAD:$PromptPath") | Select-Object -First 1).Trim()
}

function Invoke-Watcher {
    param(
        [hashtable]$Fixture,
        [string]$Mode = "success",
        [double]$LoopHours = 0.01,
        [int]$MaxIterations = 1,
        [int]$SleepMs = 0
    )

    $env:FAKE_CODEX_MODE = $Mode
    $env:FAKE_CODEX_LOG = $Fixture.FakeLog
    if ($SleepMs -gt 0) {
        $env:FAKE_CODEX_SLEEP_MS = [string]$SleepMs
    }
    else {
        Remove-Item Env:\FAKE_CODEX_SLEEP_MS -ErrorAction SilentlyContinue
    }

    $output = & $PowerShellExe -NoProfile -ExecutionPolicy Bypass -File $WatcherPath `
        -RunOnce `
        -PollSeconds 1 `
        -LoopHours $LoopHours `
        -MaxIterations $MaxIterations `
        -RemoteUrl $Fixture.Origin `
        -StateRoot $Fixture.State `
        -RunnerRoot $Fixture.Runner `
        -CodexCommand $Fixture.FakeCodex 2>&1
    $exitCode = $LASTEXITCODE

    Remove-Item Env:\FAKE_CODEX_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:\FAKE_CODEX_LOG -ErrorAction SilentlyContinue
    Remove-Item Env:\FAKE_CODEX_SLEEP_MS -ErrorAction SilentlyContinue

    return @{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object { $_.ToString() })
    }
}

function Get-StateValue {
    param([hashtable]$Fixture)

    $stateFile = Join-Path $Fixture.State "last_processed_blob.txt"
    if (-not (Test-Path $stateFile)) {
        return ""
    }
    return (Get-Content -Path $stateFile -Raw).Trim()
}

function Get-FakeIterationCount {
    param([hashtable]$Fixture)

    if (-not (Test-Path $Fixture.Runner)) {
        return 0
    }
    return @(Get-ChildItem -Path $Fixture.Runner -Filter "fake_iteration_*.txt" -ErrorAction SilentlyContinue).Count
}

function Remove-Fixture {
    param([hashtable]$Fixture)

    $tempRoot = [System.IO.Path]::GetTempPath().TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $resolved = Resolve-Path -LiteralPath $Fixture.Root -ErrorAction SilentlyContinue
    if ($null -ne $resolved -and $resolved.Path.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolved.Path -Recurse -Force
    }
}

function Test-Parse {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($WatcherPath, [ref]$tokens, [ref]$errors) | Out-Null
    Assert-True ($errors.Count -eq 0) "PowerShell parse errors: $($errors | Out-String)"
}

function Test-BranchSync {
    $fixture = New-Fixture
    try {
        $first = Invoke-Watcher $fixture
        Assert-True ($first.ExitCode -eq 0) "First branch-sync run failed: $($first.Output -join [Environment]::NewLine)"
        Assert-True ((Get-FakeIterationCount $fixture) -eq 1) "Expected first fake iteration."

        $newPromptSha = Update-RemotePrompt $fixture "second queued prompt"
        $second = Invoke-Watcher $fixture
        Assert-True ($second.ExitCode -eq 0) "Second branch-sync run failed: $($second.Output -join [Environment]::NewLine)"
        Assert-True ((Get-FakeIterationCount $fixture) -eq 2) "Runner did not fast-forward and process the second prompt."
        Assert-True ((Get-StateValue $fixture) -eq $newPromptSha) "Processed prompt state was not updated after branch sync."
    }
    finally {
        Remove-Fixture $fixture
    }
}

function Test-DirtyCheckout {
    $fixture = New-Fixture
    try {
        $first = Invoke-Watcher $fixture
        Assert-True ($first.ExitCode -eq 0) "Initial dirty-checkout setup run failed."
        $oldState = Get-StateValue $fixture
        $newPromptSha = Update-RemotePrompt $fixture "prompt blocked by dirty runner"
        Set-Content -Path (Join-Path $fixture.Runner "dirty-runner-file.txt") -Value "do not overwrite" -Encoding utf8

        $blocked = Invoke-Watcher $fixture
        Assert-True ($blocked.ExitCode -ne 0) "Dirty runner checkout should block the watcher."
        Assert-True ((Get-FakeIterationCount $fixture) -eq 1) "Watcher launched fake Codex despite dirty runner."
        Assert-True ((Get-StateValue $fixture) -eq $oldState) "Dirty-runner failure should retain the new prompt for retry."

        $repair = Get-Content -Path (Join-Path $fixture.State "repair_watcher_prompt.txt") -Raw
        Assert-True ($repair.Contains("Runner checkout is dirty")) "Repair prompt did not capture dirty-checkout error."
        Assert-True (-not $repair.Contains($newPromptSha)) "Repair prompt should not mark the blocked prompt as processed."
    }
    finally {
        Remove-Fixture $fixture
    }
}

function Test-FailureRetry {
    $fixture = New-Fixture
    try {
        $failed = Invoke-Watcher $fixture -Mode "fail"
        Assert-True ($failed.ExitCode -ne 0) "Fake Codex failure should fail the watcher cycle."
        Assert-True ((Get-StateValue $fixture) -eq "") "Failed prompt should not be marked processed."

        $repairPath = Join-Path $fixture.State "repair_watcher_prompt.txt"
        $repair = Get-Content -Path $repairPath -Raw
        Assert-True ($repair.Contains("Codex iteration 1 exited with code 42.")) "Repair prompt did not include exact exit code."
        Assert-True ($repair.Contains("Exact log path:")) "Repair prompt did not include log path header."

        $retried = Invoke-Watcher $fixture
        Assert-True ($retried.ExitCode -eq 0) "Retry after failure did not succeed: $($retried.Output -join [Environment]::NewLine)"
        Assert-True ((Get-FakeIterationCount $fixture) -eq 1) "Retry did not launch exactly one successful fake iteration."
        Assert-True ((Get-StateValue $fixture) -ne "") "Successful retry did not mark prompt processed."
    }
    finally {
        Remove-Fixture $fixture
    }
}

function Test-Deadline {
    $fixture = New-Fixture
    try {
        $result = Invoke-Watcher $fixture -LoopHours 0.0003 -MaxIterations 5 -SleepMs 1500
        Assert-True ($result.ExitCode -eq 0) "Deadline run failed: $($result.Output -join [Environment]::NewLine)"
        Assert-True ((Get-FakeIterationCount $fixture) -eq 1) "Watcher launched another iteration after the deadline."
    }
    finally {
        Remove-Fixture $fixture
    }
}

Test-Parse
Test-BranchSync
Test-DirtyCheckout
Test-FailureRetry
Test-Deadline

Write-Host "watch_chatgpt_codex_tests.ps1 passed"
