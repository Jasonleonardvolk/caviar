# Build-SocialPack.ps1
# Input: an SDR 709 master (e.g., *_sdr.mp4) or a 709 ProRes .mov
# Output:
#   D:\Dev\kha\content\socialpack\out\tiktok\<base>_tiktok_1080x1920.mp4
#   D:\Dev\kha\content\socialpack\out\snap\<base>_snap_1080x1920.mp4
#   + thumbnails in ...\thumbs\<base>_NN.jpg
# Notes:
#   - 9:16 safe export (1080×1920), H.264 High@4.2, AAC 48k
#   - Bitrate targets: 10–12 Mb/s @60fps (or ~8 Mb/s @30fps)
#   - Caps file size if you ask for it

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Input,
  [int]$Framerate = 60,
  [int]$VideoBitrateMbps = 10,
  [int]$MaxFileSizeMB = 250,
  [switch]$NoThumbs
)

$ErrorActionPreference = 'Stop'
function Need($exe){ if(-not (Get-Command $exe -ErrorAction SilentlyContinue)){ throw "Missing tool: $exe" } }
Need ffmpeg; Need ffprobe

$repo   = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$rootIn = Join-Path $repo "content\socialpack\input"
$rootOut= Join-Path $repo "content\socialpack\out"
$thumbs = Join-Path $repo "content\socialpack\thumbs"
$outTik = Join-Path $rootOut "tiktok"
$outSnap= Join-Path $rootOut "snap"
$static = Join-Path $repo  "tori_ui_svelte\static\social"

New-Item $rootOut -ItemType Directory -Force | Out-Null
New-Item $thumbs  -ItemType Directory -Force | Out-Null
New-Item $outTik  -ItemType Directory -Force | Out-Null
New-Item $outSnap -ItemType Directory -Force | Out-Null
New-Item $static  -ItemType Directory -Force | Out-Null
New-Item (Join-Path $static "tiktok") -ItemType Directory -Force | Out-Null
New-Item (Join-Path $static "snap")   -ItemType Directory -Force | Out-Null

$inPath = (Resolve-Path $Input).Path
$base   = [IO.Path]::GetFileNameWithoutExtension($inPath)

# Detect orientation for smarter crop (but keep it simple + reliable)
$probe = ffprobe -v quiet -print_format json -show_streams "$inPath" | ConvertFrom-Json
$video = $probe.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1
$srcW  = [int]$video.width; $srcH = [int]$video.height

# 9:16 framing → scale to 1920 tall, then center crop 1080 width
$vf = 'scale=-2:1920:flags=lanczos,crop=1080:1920,setsar=1,format=yuv420p'

# Core encoder settings
$br   = ($VideoBitrateMbps * 1000000)
$mx   = ([int]($VideoBitrateMbps*1.2) * 1000000)
$buf  = ([int]($VideoBitrateMbps*2) * 1000000)
$g    = ($Framerate * 2)

$common = @(
  '-r', $Framerate,
  '-c:v','libx264','-profile:v','high','-level','4.2','-pix_fmt','yuv420p',
  '-b:v',$br, '-maxrate',$mx, '-bufsize',$buf, '-g', $g,
  '-c:a','aac','-b:a','160k','-ar','48000',
  '-movflags','+faststart',
  '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709'
)

# Outputs
$outTikFile  = Join-Path $outTik  "${base}_tiktok_1080x1920.mp4"
$outSnapFile = Join-Path $outSnap "${base}_snap_1080x1920.mp4"
$statTik     = Join-Path $static  "tiktok\${base}_tiktok_1080x1920.mp4"
$statSnap    = Join-Path $static  "snap\${base}_snap_1080x1920.mp4"

Write-Host "==> Social encode: $base" -ForegroundColor Cyan

# TikTok
& ffmpeg -y -i "$inPath" -vf $vf @common "$outTikFile"
if($LASTEXITCODE){ throw "TikTok encode failed" }

# Snapchat (same recipe; keep independent files for tracking)
& ffmpeg -y -i "$inPath" -vf $vf @common "$outSnapFile"
if($LASTEXITCODE){ throw "Snap encode failed" }

# Size cap re-encode if needed
function Cap-Size($path,[int]$limitMB){
  $szMB = [math]::Round((Get-Item $path).Length/1MB,2)
  if($szMB -le $limitMB){ return }
  $scale = [math]::Max(0.6, [math]::Round($limitMB / $szMB,2))  # don't crush it too hard
  $newBR = [int]($br * $scale); $newMX=[int]($mx*$scale); $newBUF=[int]($buf*$scale)
  $tmp = "$path.tmp.mp4"
  & ffmpeg -y -i "$path" -c:v libx264 -b:v $newBR -maxrate $newMX -bufsize $newBUF `
           -c:a copy -movflags +faststart "$tmp"
  if(-not $LASTEXITCODE){ Move-Item -Force "$tmp" "$path" }
}
Cap-Size $outTikFile  $MaxFileSizeMB
Cap-Size $outSnapFile $MaxFileSizeMB

# Thumbs
if(-not $NoThumbs){
  New-Item (Join-Path $thumbs $base) -ItemType Directory -Force | Out-Null
  & ffmpeg -y -i "$outTikFile" -vf "fps=1/2,scale=540:-2" (Join-Path $thumbs "${base}\${base}_%02d.jpg")
}

# Copy to static for QR/device testing
Copy-Item $outTikFile  $statTik  -Force
Copy-Item $outSnapFile $statSnap -Force

# Report
$tik = [math]::Round((Get-Item $outTikFile).Length/1MB,2)
$sna = [math]::Round((Get-Item $outSnapFile).Length/1MB,2)
Write-Host ("  TikTok: {0} MB → {1}" -f $tik, $outTikFile) -ForegroundColor Green
Write-Host ("  Snap  : {0} MB → {1}" -f $sna, $outSnapFile) -ForegroundColor Green
Write-Host ("Test on device: http://<LAN-IP>:3000/social/tiktok/{0}" -f (Split-Path $statTik -Leaf)) -ForegroundColor Yellow