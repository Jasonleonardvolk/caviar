# Build-WowPack.ps1
# Encodes ProRes .mov masters to HEVC HDR10 + AV1 + H.264 SDR
# Properly maps Rec.709 to BT.2020 PQ for HDR10
#
# Usage:
#   .\Build-WowPack.ps1 -Basename holo_flux_loop -Input "..\..\content\wowpack\input\holo_flux_loop.mov" -Framerate 60 -DoSDR
#   .\Build-WowPack.ps1 -Basename mach_lightfield -Input "..\..\content\wowpack\input\mach_lightfield.mov" -Framerate 60 -DoSDR -MakeHLS

param(
  [Parameter(Mandatory=$true)][string]$Basename,
  [Parameter(Mandatory=$true)][string]$Input,
  [int]$Framerate = 60,
  [switch]$DoSDR,
  [switch]$MakeHLS
)

$ErrorActionPreference = "Stop"

function NeedTool($name) {
  $p = Get-Command $name -ErrorAction SilentlyContinue
  if (-not $p) { throw "Required tool '$name' not found in PATH." }
}

NeedTool ffmpeg
NeedTool ffprobe

$root      = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$repo      = Split-Path -Parent $root
$content   = Join-Path $repo "content\wowpack"
$outHDR10  = Join-Path $content "video\hdr10"
$outAV1    = Join-Path $content "video\av1"
$staticWow = Join-Path $repo "tori_ui_svelte\static\media\wow"
$hlsDir    = Join-Path $repo "tori_ui_svelte\static\media\hls\$Basename"

# Create directories
New-Item $outHDR10 -ItemType Directory -Force | Out-Null
New-Item $outAV1   -ItemType Directory -Force | Out-Null
New-Item $staticWow -ItemType Directory -Force | Out-Null
if ($MakeHLS) { New-Item $hlsDir -ItemType Directory -Force | Out-Null }

# Define output paths
$hdr10Archive = Join-Path $outHDR10 "${Basename}_hdr10.mp4"
$av1Archive   = Join-Path $outAV1   "${Basename}_av1.mp4"
$sdrArchive   = Join-Path $outHDR10 "${Basename}_sdr.mp4"

# Runtime serving paths
$hdr10Static = Join-Path $staticWow "${Basename}_hdr10.mp4"
$av1Static   = Join-Path $staticWow "${Basename}_av1.mp4"
$sdrStatic   = Join-Path $staticWow "${Basename}_sdr.mp4"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "WOW Pack Encoder - $Basename" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor White
Write-Host "Input: $Input" -ForegroundColor Gray
Write-Host ""

# Analyze input to verify it's proper ProRes
Write-Host "Analyzing input file..." -ForegroundColor Yellow
$probeJson = & ffprobe -v quiet -print_format json -show_streams "$Input" | ConvertFrom-Json
$videoStream = $probeJson.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1

Write-Host "  Codec: $($videoStream.codec_name)" -ForegroundColor Gray
Write-Host "  Pixel Format: $($videoStream.pix_fmt)" -ForegroundColor Gray
Write-Host "  Resolution: $($videoStream.width)x$($videoStream.height)" -ForegroundColor Gray
if ($videoStream.color_primaries) {
    Write-Host "  Color Primaries: $($videoStream.color_primaries)" -ForegroundColor Gray
}
Write-Host ""

# HEVC HDR10 encoding with proper Rec.709 to BT.2020 PQ mapping
Write-Host "[1/3] Encoding HEVC HDR10..." -ForegroundColor Cyan
Write-Host "      Mapping Rec.709 -> BT.2020 PQ" -ForegroundColor Gray

$hevcCmd = @"
ffmpeg -y -i "$Input" `
  -vf "zscale=rin=full:pin=bt709:tin=bt709,zscale=t=bt2020nc:m=bt2020nc:r=tv,zscale=transfer=smpte2084" `
  -r $Framerate `
  -c:v libx265 `
  -preset medium `
  -crf 18 `
  -pix_fmt yuv420p10le `
  -x265-params "hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400:aud=1:keyint=120:min-keyint=24" `
  -tag:v hvc1 `
  -movflags +faststart `
  -an `
  "$hdr10Archive"
"@

Invoke-Expression $hevcCmd
if ($LASTEXITCODE -ne 0) { throw "HEVC encoding failed" }

Copy-Item $hdr10Archive $hdr10Static -Force
Write-Host "  [OK] HEVC HDR10 complete" -ForegroundColor Green
Write-Host ""

# AV1 10-bit encoding with HDR signaling
Write-Host "[2/3] Encoding AV1 10-bit..." -ForegroundColor Cyan

$av1Cmd = @"
ffmpeg -y -i "$Input" `
  -vf "scale=out_color_matrix=bt2020nc:out_h_chr_pos=0:out_v_chr_pos=0,format=yuv420p10le" `
  -r $Framerate `
  -c:v libsvtav1 `
  -preset 6 `
  -crf 30 `
  -pix_fmt yuv420p10le `
  -color_primaries bt2020 `
  -color_trc smpte2084 `
  -colorspace bt2020nc `
  -svtav1-params "enable-hdr=1:mastering-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):content-light=1000,400" `
  -movflags +faststart `
  -tag:v av01 `
  -an `
  "$av1Archive"
"@

Invoke-Expression $av1Cmd
if ($LASTEXITCODE -ne 0) { throw "AV1 encoding failed" }

Copy-Item $av1Archive $av1Static -Force
Write-Host "  [OK] AV1 10-bit complete" -ForegroundColor Green
Write-Host ""

# H.264 SDR fallback with proper tonemapping
if ($DoSDR) {
  Write-Host "[3/3] Encoding H.264 SDR fallback..." -ForegroundColor Cyan
  Write-Host "      Tonemapping HDR -> SDR" -ForegroundColor Gray
  
  $sdrCmd = @"
ffmpeg -y -i "$Input" `
  -vf "zscale=t=linear:npl=100,tonemap=mobius:param=0.5,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" `
  -r $Framerate `
  -c:v libx264 `
  -preset slow `
  -crf 18 `
  -profile:v high `
  -level 4.2 `
  -pix_fmt yuv420p `
  -color_primaries bt709 `
  -color_trc bt709 `
  -colorspace bt709 `
  -movflags +faststart `
  -an `
  "$sdrArchive"
"@

  Invoke-Expression $sdrCmd
  if ($LASTEXITCODE -ne 0) { throw "SDR encoding failed" }
  
  Copy-Item $sdrArchive $sdrStatic -Force
  Write-Host "  [OK] H.264 SDR complete" -ForegroundColor Green
  Write-Host ""
}

# Optional HLS generation
if ($MakeHLS) {
  Write-Host "[+] Generating HLS segments..." -ForegroundColor Cyan
  
  $hlsCmd = @"
ffmpeg -y -i "$hdr10Static" `
  -c copy `
  -hls_time 4 `
  -hls_playlist_type vod `
  -hls_flags independent_segments `
  -hls_segment_type fmp4 `
  -hls_segment_filename "$hlsDir\segment_%03d.m4s" `
  -hls_fmp4_init_filename "init.mp4" `
  "$hlsDir\playlist.m3u8"
"@

  Invoke-Expression $hlsCmd
  Write-Host "  [OK] HLS segments generated" -ForegroundColor Green
  Write-Host ""
}

# Update manifest
Write-Host "Updating manifest..." -ForegroundColor Yellow
$manifestPath = Join-Path $staticWow "wow.manifest.json"

function Probe($path) {
  $json = ffprobe -v quiet -print_format json -show_streams "$path" | ConvertFrom-Json
  $v = $json.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1
  return @{
    width = $v.width
    height = $v.height
    codec = $v.codec_name
    pix_fmt = $v.pix_fmt
    r_frame_rate = $v.r_frame_rate
    color_primaries = $v.color_primaries
    color_transfer = $v.color_transfer
  }
}

$entry = @{
  id     = $Basename
  label  = ($Basename -replace '_',' ' -creplace '\b(\w)', { $args[0].Value.ToUpper() })
  sources = @()
}

# Add HEVC source
$hevcInfo = Probe $hdr10Static
$entry.sources += @{
  type = 'hevc'
  url = "/media/wow/${Basename}_hdr10.mp4"
  codecs = "hvc1.2.4.L120.B0"
  meta = $hevcInfo
}

# Add AV1 source
$av1Info = Probe $av1Static
$entry.sources += @{
  type = 'av1'
  url = "/media/wow/${Basename}_av1.mp4"
  codecs = "av01.0.08M.10"
  meta = $av1Info
}

# Add SDR source if generated
if ($DoSDR) {
  $sdrInfo = Probe $sdrStatic
  $entry.sources += @{
    type = 'h264'
    url = "/media/wow/${Basename}_sdr.mp4"
    codecs = "avc1.640028"
    meta = $sdrInfo
  }
}

# Add HLS if generated
if ($MakeHLS) {
  $entry.hls = "/media/hls/$Basename/playlist.m3u8"
}

# Load existing manifest or create new
if (Test-Path $manifestPath) {
  $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
  # Remove existing entry with same ID
  $clips = @($manifest.clips | Where-Object { $_.id -ne $Basename })
  $clips += $entry
  $manifest.clips = $clips
  $manifest.updated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
} else {
  $manifest = @{
    version = 1
    updated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    clips = @($entry)
  }
}

$manifest | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $manifestPath

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Encoding Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files created:" -ForegroundColor Yellow
Write-Host "  Archive: $hdr10Archive" -ForegroundColor Gray
Write-Host "  Archive: $av1Archive" -ForegroundColor Gray
if ($DoSDR) {
  Write-Host "  Archive: $sdrArchive" -ForegroundColor Gray
}
Write-Host "  Runtime: $hdr10Static" -ForegroundColor Gray
Write-Host "  Runtime: $av1Static" -ForegroundColor Gray
if ($DoSDR) {
  Write-Host "  Runtime: $sdrStatic" -ForegroundColor Gray
}
if ($MakeHLS) {
  Write-Host "  HLS: $hlsDir" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Manifest updated: $manifestPath" -ForegroundColor Green
Write-Host ""
Write-Host "Test at: http://localhost:3000/hologram?clip=$Basename" -ForegroundColor Cyan
