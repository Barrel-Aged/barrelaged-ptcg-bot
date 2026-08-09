$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\jhos4\AppData\Local\Programs\Python\Python312\python.exe"
$Collector = Join-Path $PSScriptRoot "collect_target80_raw_action_milestone_v1r.py"
$OutputRoot = Join-Path $ProjectRoot "work\target80_raw_action_milestone_v1r_pilot"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing Python runtime: $Python"
}
if (-not (Test-Path -LiteralPath $Collector -PathType Leaf)) {
    throw "Missing collector: $Collector"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to reuse pilot directory: $OutputRoot"
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$Cells = @(
    @{ Stem = "01_marnie_first"; Opponent = "work\arena_bundles\AshV178MarnieFullBest_20260801_extracted"; Seat = "first"; Seed = 806101 },
    @{ Stem = "02_marnie_second"; Opponent = "work\arena_bundles\AshV178MarnieFullBest_20260801_extracted"; Seat = "second"; Seed = 806102 },
    @{ Stem = "03_mist_first"; Opponent = "work\arena_bundles\SparkStarmieMistFlatV20_20260801_extracted"; Seat = "first"; Seed = 806201 },
    @{ Stem = "04_mist_second"; Opponent = "work\arena_bundles\SparkStarmieMistFlatV20_20260801_extracted"; Seat = "second"; Seed = 806202 },
    @{ Stem = "05_fros_first"; Opponent = "work\arena_bundles\SparkStarmieFroslassCFV13_extracted"; Seat = "first"; Seed = 806301 },
    @{ Stem = "06_fros_second"; Opponent = "work\arena_bundles\SparkStarmieFroslassCFV13_extracted"; Seat = "second"; Seed = 806302 },
    @{ Stem = "07_rocket_first"; Opponent = "work\arena_bundles\SparkRocketSpidopsV1_extracted"; Seat = "first"; Seed = 806401 },
    @{ Stem = "08_rocket_second"; Opponent = "work\arena_bundles\SparkRocketSpidopsV1_extracted"; Seat = "second"; Seed = 806402 },
    @{ Stem = "09_purerl_first"; Opponent = "work\arena_bundles_20260804\SoldierPureRLV45_extracted"; Seat = "first"; Seed = 806501 },
    @{ Stem = "10_purerl_second"; Opponent = "work\arena_bundles_20260804\SoldierPureRLV45_extracted"; Seat = "second"; Seed = 806502 },
    @{ Stem = "11_alakazam_first"; Opponent = "work\arena_bundles_20260804\VioletAlakazamMKV93_extracted"; Seat = "first"; Seed = 806601 },
    @{ Stem = "12_alakazam_second"; Opponent = "work\arena_bundles_20260804\VioletAlakazamMKV93_extracted"; Seat = "second"; Seed = 806602 },
    @{ Stem = "13_crustle_first"; Opponent = "work\arena_bundles_20260804\SparkTrueCrustleCrispinGalvV71_extracted"; Seat = "first"; Seed = 806701 },
    @{ Stem = "14_crustle_second"; Opponent = "work\arena_bundles_20260804\SparkTrueCrustleCrispinGalvV71_extracted"; Seat = "second"; Seed = 806702 }
)

Set-Location -LiteralPath $ProjectRoot
foreach ($Cell in $Cells) {
    $Output = Join-Path $OutputRoot ($Cell.Stem + ".jsonl")
    Write-Output ("starting {0}" -f $Cell.Stem)
    & $Python $Collector `
        --opponent $Cell.Opponent `
        --output $Output `
        --roots 2 `
        --seed $Cell.Seed `
        --seat $Cell.Seat `
        --capture-roots-per-game 8 `
        --candidates 3 `
        --minimum-alternative-support 0.02 `
        --particles 8 `
        --coin-rollouts 2 `
        --earliest-turn 2 `
        --latest-turn 8 `
        --max-source-attempts 240 `
        --max-live-decisions 2000 `
        --max-rollout-decisions 150 `
        --root-budget-seconds 180 `
        --threads 1
    if ($LASTEXITCODE -ne 0) {
        throw "Collector failed for $($Cell.Stem) with exit code $LASTEXITCODE"
    }
    Write-Output ("completed {0}" -f $Cell.Stem)
}
Write-Output "pilot_complete"
