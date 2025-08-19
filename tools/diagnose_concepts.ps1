# tools/diagnose_concepts.ps1 — PowerShell: Simple concept diagnostic
<#
.SYNOPSIS
    Diagnose concept extraction results from JSON files.

.DESCRIPTION
    This PowerShell script analyzes semantic_concepts.json files to check
    concept count, confidence scores, and metadata completeness. It provides
    a Windows-friendly diagnostic tool for the TORI concept ingestion pipeline.

.PARAMETER FilePath
    Path to the semantic_concepts.json file to analyze

.PARAMETER Directory
    Directory containing multiple concept JSON files to batch analyze

.PARAMETER ShowSamples
    Number of sample concepts to display (default: 5)

.PARAMETER NoSamples
    Skip showing sample concepts

.EXAMPLE
    .\diagnose_concepts.ps1 "semantic_concepts.json"
    Analyze a single concept file

.EXAMPLE
    .\diagnose_concepts.ps1 -Directory ".\concept_outputs\"
    Analyze all concept files in a directory

.EXAMPLE
    .\diagnose_concepts.ps1 -FilePath "concepts.json" -ShowSamples 10
    Analyze file and show 10 sample concepts
#>

param (
    [string]$FilePath = "semantic_concepts.json",
    [string]$Directory = "",
    [int]$ShowSamples = 5,
    [switch]$NoSamples
)

function Test-ConceptFile {
    param([string]$Path)
    
    if (!(Test-Path $Path)) {
        Write-Host "❌ File not found: $Path" -ForegroundColor Red
        return $false
    }
    
    try {
        $content = Get-Content $Path -Raw -ErrorAction Stop
        $null = $content | ConvertFrom-Json -ErrorAction Stop
        return $true
    }
    catch {
        Write-Host "❌ Invalid JSON in file: $Path" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

function Get-ConceptData {
    param([string]$Path)
    
    try {
        $content = Get-Content $Path -Raw | ConvertFrom-Json
        
        # Handle different JSON structures
        if ($content -is [array]) {
            return $content
        }
        elseif ($content.concepts) {
            return $content.concepts
        }
        else {
            Write-Host "⚠️  Warning: Unexpected JSON structure in $Path" -ForegroundColor Yellow
            return @()
        }
    }
    catch {
        Write-Host "❌ Error loading $Path`: $($_.Exception.Message)" -ForegroundColor Red
        return @()
    }
}

function Analyze-Concepts {
    param([array]$Concepts)
    
    if (-not $Concepts -or $Concepts.Count -eq 0) {
        return @{ Error = "No concepts to analyze" }
    }
    
    $totalCount = $Concepts.Count
    
    # Confidence analysis
    $confidences = @()
    foreach ($concept in $Concepts) {
        if ($concept.confidence -is [double] -or $concept.confidence -is [int]) {
            $confidences += [double]$concept.confidence
        }
    }
    
    $confidenceStats = @{}
    if ($confidences.Count -gt 0) {
        $confidenceStats = @{
            Min = ($confidences | Measure-Object -Minimum).Minimum
            Max = ($confidences | Measure-Object -Maximum).Maximum
            Mean = ($confidences | Measure-Object -Average).Average
            Count = $confidences.Count
        }
    }
    
    # Method analysis
    $methods = @{}
    foreach ($concept in $Concepts) {
        $method = if ($concept.method) { $concept.method } else { "unknown" }
        $methods[$method] = ($methods[$method] ?? 0) + 1
    }
    
    # Source analysis
    $sources = @{}
    foreach ($concept in $Concepts) {
        $source = "unknown_source"
        if ($concept.source) {
            if ($concept.source.page) {
                $source = "page_$($concept.source.page)"
            }
            elseif ($concept.source.segment) {
                $source = "segment_$($concept.source.segment)"
            }
            else {
                $source = $concept.source.ToString()
            }
        }
        $sources[$source] = ($sources[$source] ?? 0) + 1
    }
    
    # Quality tiers
    $qualityTiers = @{ High = 0; Medium = 0; Low = 0; Missing = 0 }
    foreach ($concept in $Concepts) {
        if ($null -eq $concept.confidence) {
            $qualityTiers.Missing++
        }
        elseif ($concept.confidence -ge 0.8) {
            $qualityTiers.High++
        }
        elseif ($concept.confidence -ge 0.6) {
            $qualityTiers.Medium++
        }
        else {
            $qualityTiers.Low++
        }
    }
    
    # Field presence
    $requiredFields = @("name", "confidence", "method", "source")
    $optionalFields = @("context", "embedding", "eigenfunction_id")
    $fieldPresence = @{}
    
    foreach ($field in ($requiredFields + $optionalFields)) {
        $present = 0
        foreach ($concept in $Concepts) {
            if ($concept.$field -and $concept.$field -ne "") {
                $present++
            }
        }
        $fieldPresence[$field] = @{
            Count = $present
            Percentage = if ($totalCount -gt 0) { ($present / $totalCount) * 100 } else { 0 }
        }
    }
    
    return @{
        TotalCount = $totalCount
        ConfidenceStats = $confidenceStats
        MethodDistribution = $methods
        SourceDistribution = $sources
        QualityTiers = $qualityTiers
        FieldPresence = $fieldPresence
        RequiredFields = $requiredFields
        OptionalFields = $optionalFields
    }
}

function Show-Analysis {
    param(
        [string]$FilePath,
        [hashtable]$Analysis
    )
    
    Write-Host "`n📄 Analyzing: $FilePath" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Gray
    
    if ($Analysis.Error) {
        Write-Host "❌ $($Analysis.Error)" -ForegroundColor Red
        return
    }
    
    # Basic info
    Write-Host "✅ Total Concepts: $($Analysis.TotalCount)" -ForegroundColor Green
    
    # Confidence statistics
    if ($Analysis.ConfidenceStats.Count -gt 0) {
        $conf = $Analysis.ConfidenceStats
        Write-Host "`n📊 Confidence Statistics:" -ForegroundColor Blue
        Write-Host "   Range: $($conf.Min.ToString('F3')) - $($conf.Max.ToString('F3'))"
        Write-Host "   Mean:  $($conf.Mean.ToString('F3'))"
        Write-Host "   Valid: $($conf.Count)/$($Analysis.TotalCount)"
    }
    
    # Quality distribution
    $quality = $Analysis.QualityTiers
    Write-Host "`n🏆 Quality Distribution:" -ForegroundColor Magenta
    Write-Host "   High (≥0.8):   $($quality.High.ToString().PadLeft(3)) concepts"
    Write-Host "   Medium (≥0.6): $($quality.Medium.ToString().PadLeft(3)) concepts"
    Write-Host "   Low (<0.6):    $($quality.Low.ToString().PadLeft(3)) concepts"
    if ($quality.Missing -gt 0) {
        Write-Host "   Missing conf:  $($quality.Missing.ToString().PadLeft(3)) concepts" -ForegroundColor Yellow
    }
    
    # Method distribution
    Write-Host "`n🔧 Extraction Methods:" -ForegroundColor DarkCyan
    $sortedMethods = $Analysis.MethodDistribution.GetEnumerator() | Sort-Object Value -Descending
    foreach ($method in $sortedMethods) {
        $percentage = if ($Analysis.TotalCount -gt 0) { ($method.Value / $Analysis.TotalCount) * 100 } else { 0 }
        $methodName = $method.Key.PadRight(20)
        Write-Host "   ${methodName}: $($method.Value.ToString().PadLeft(3)) ($($percentage.ToString('F1').PadLeft(5))%)"
    }
    
    # Top sources
    Write-Host "`n📍 Top Sources:" -ForegroundColor DarkGreen
    $sortedSources = $Analysis.SourceDistribution.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5
    foreach ($source in $sortedSources) {
        $percentage = if ($Analysis.TotalCount -gt 0) { ($source.Value / $Analysis.TotalCount) * 100 } else { 0 }
        $sourceName = $source.Key.PadRight(20)
        Write-Host "   ${sourceName}: $($source.Value.ToString().PadLeft(3)) ($($percentage.ToString('F1').PadLeft(5))%)"
    }
    
    # Metadata completeness
    Write-Host "`n📋 Metadata Completeness:" -ForegroundColor DarkYellow
    Write-Host "   Required Fields:"
    foreach ($field in $Analysis.RequiredFields) {
        $stats = $Analysis.FieldPresence[$field]
        $status = if ($stats.Percentage -ge 95) { "✅" } elseif ($stats.Percentage -ge 80) { "⚠️ " } else { "❌" }
        $fieldName = $field.PadRight(15)
        Write-Host "   $status ${fieldName}: $($stats.Count.ToString().PadLeft(3))/$($Analysis.TotalCount) ($($stats.Percentage.ToString('F1').PadLeft(5))%)"
    }
    
    Write-Host "   Optional Fields:"
    foreach ($field in $Analysis.OptionalFields) {
        $stats = $Analysis.FieldPresence[$field]
        $status = if ($stats.Percentage -ge 50) { "✅" } else { "📝" }
        $fieldName = $field.PadRight(15)
        Write-Host "   $status ${fieldName}: $($stats.Count.ToString().PadLeft(3))/$($Analysis.TotalCount) ($($stats.Percentage.ToString('F1').PadLeft(5))%)"
    }
}

function Show-SampleConcepts {
    param(
        [array]$Concepts,
        [int]$Count = 5
    )
    
    if (-not $Concepts -or $Concepts.Count -eq 0) {
        return
    }
    
    $sampleCount = [Math]::Min($Count, $Concepts.Count)
    Write-Host "`n🔍 Sample Concepts (showing $sampleCount):" -ForegroundColor Cyan
    Write-Host ("-" * 80) -ForegroundColor Gray
    
    for ($i = 0; $i -lt $sampleCount; $i++) {
        $concept = $Concepts[$i]
        $name = if ($concept.name) { $concept.name } else { "Unnamed" }
        $confidence = if ($concept.confidence) { $concept.confidence.ToString('F3') } else { "?" }
        $method = if ($concept.method) { $concept.method } else { "unknown" }
        $source = if ($concept.source) { $concept.source | ConvertTo-Json -Compress } else { "{}" }
        $context = if ($concept.context) { 
            if ($concept.context.Length -gt 60) { 
                $concept.context.Substring(0, 60) + "..." 
            } else { 
                $concept.context 
            }
        } else { 
            "No context" 
        }
        
        Write-Host "  $($i + 1).".PadRight(4) -NoNewline
        Write-Host $name -ForegroundColor White
        Write-Host "      Confidence: $confidence | Method: $method" -ForegroundColor Gray
        Write-Host "      Source: $source" -ForegroundColor Gray
        Write-Host "      Context: $context" -ForegroundColor DarkGray
        Write-Host ""
    }
}

function Process-Directory {
    param([string]$DirPath)
    
    if (!(Test-Path $DirPath)) {
        Write-Host "❌ Directory not found: $DirPath" -ForegroundColor Red
        return
    }
    
    # Find concept JSON files
    $jsonFiles = @()
    $jsonFiles += Get-ChildItem -Path $DirPath -Filter "*concept*.json" -Recurse
    $jsonFiles += Get-ChildItem -Path $DirPath -Filter "*semantic*.json" -Recurse
    $jsonFiles = $jsonFiles | Sort-Object Name | Get-Unique -AsString
    
    if ($jsonFiles.Count -eq 0) {
        Write-Host "📂 No concept JSON files found in $DirPath" -ForegroundColor Yellow
        return
    }
    
    Write-Host "📂 Processing $($jsonFiles.Count) files in $DirPath" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Gray
    
    $totalConcepts = 0
    $successfulFiles = 0
    
    foreach ($file in $jsonFiles) {
        $concepts = Get-ConceptData -Path $file.FullName
        if ($concepts -and $concepts.Count -gt 0) {
            $totalConcepts += $concepts.Count
            $successfulFiles++
            
            # Brief analysis
            $analysis = Analyze-Concepts -Concepts $concepts
            $avgConf = if ($analysis.ConfidenceStats.Mean) { $analysis.ConfidenceStats.Mean } else { 0 }
            
            $fileName = $file.Name.PadRight(40)
            $conceptCount = $concepts.Count.ToString().PadLeft(4)
            Write-Host "✅ $fileName $conceptCount concepts (avg conf: $($avgConf.ToString('F2')))" -ForegroundColor Green
        }
        else {
            $fileName = $file.Name.PadRight(40)
            Write-Host "❌ $fileName failed to load or empty" -ForegroundColor Red
        }
    }
    
    Write-Host "`n📋 Summary: $successfulFiles/$($jsonFiles.Count) files processed, $totalConcepts total concepts" -ForegroundColor Cyan
}

# Main execution
if ($Directory) {
    Process-Directory -DirPath $Directory
}
elseif (Test-Path $FilePath -PathType Container) {
    Process-Directory -DirPath $FilePath
}
else {
    if (Test-ConceptFile -Path $FilePath) {
        $concepts = Get-ConceptData -Path $FilePath
        if ($concepts -and $concepts.Count -gt 0) {
            $analysis = Analyze-Concepts -Concepts $concepts
            Show-Analysis -FilePath $FilePath -Analysis $analysis
            
            if (-not $NoSamples) {
                Show-SampleConcepts -Concepts $concepts -Count $ShowSamples
            }
        }
        else {
            Write-Host "❌ No concepts found in $FilePath" -ForegroundColor Red
        }
    }
}

Write-Host "`n🎯 Concept diagnosis complete!" -ForegroundColor Green
