param()
$ErrorActionPreference = 'Stop'
$overviewRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $overviewRoot
$index = Get-Content -LiteralPath (Join-Path $overviewRoot 'project-index.json') -Raw -Encoding UTF8 | ConvertFrom-Json
function Encode-Html([string]$Value) { [System.Net.WebUtility]::HtmlEncode($Value) }

$seenIds = @{}
$cards = foreach ($entry in $index.experiments) {
    if ($entry.id -notmatch '^[a-z][a-z0-9_]*$' -or $seenIds.ContainsKey($entry.id)) { throw "Invalid or duplicate id: $($entry.id)" }
    $seenIds[$entry.id] = $true
    foreach ($field in @('report', 'protocol', 'archive', 'code')) {
        $target = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $entry.$field))
        if (-not $target.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Path outside project: $target" }
        if (-not (Test-Path -LiteralPath $target)) { throw "Missing $field for $($entry.id): $target" }
    }
    $number = [array]::IndexOf($index.experiments, $entry) + 1
    $expanded = if ($entry.id -eq 'repeated') { ' open' } else { '' }
    $links = foreach ($pair in @(@('report','Отчёт'), @('protocol','Протокол'), @('archive','Архив'), @('code','Код'))) {
        '<a href="../' + (Encode-Html $entry.($pair[0])) + '">' + $pair[1] + '</a>'
    }
    @"
<details class="experiment" id="exp-$(Encode-Html $entry.id)"$expanded>
  <summary><span class="number">$('{0:00}' -f $number)</span><span class="experiment-heading"><span class="experiment-title">$(Encode-Html $entry.title)</span><span class="outcome">$(Encode-Html $entry.status)</span></span><time datetime="$(Encode-Html $entry.date)">$([datetime]::ParseExact($entry.date,'yyyy-MM-dd',$null).ToString('dd.MM'))</time></summary>
  <div class="experiment-body"><dl>
    <dt>Вопрос</dt><dd>$(Encode-Html $entry.question)</dd>
    <dt>С чем сравнивали</dt><dd>$(Encode-Html $entry.baseline)</dd>
    <dt>Что получили</dt><dd>$(Encode-Html $entry.result)</dd>
    <dt>Что это меняет</dt><dd>$(Encode-Html $entry.decision)</dd>
  </dl><div class="source-links" aria-label="Источники опыта">$($links -join '')</div></div>
</details>
"@
}
$template = Get-Content -LiteralPath (Join-Path $overviewRoot 'map-template.html') -Raw -Encoding UTF8
$html = $template.Replace('<!-- EXPERIMENTS -->', ($cards -join "`n")).Replace('<!-- REVIEW_DATE -->', (Encode-Html $index.review_date)).Replace('<!-- CURRENT -->', (Encode-Html $index.current))
[System.IO.File]::WriteAllText((Join-Path $overviewRoot 'index.html'), $html, [System.Text.UTF8Encoding]::new($false))
Write-Output "Built research_overview/index.html: $($index.experiments.Count) experiments; source paths verified."
