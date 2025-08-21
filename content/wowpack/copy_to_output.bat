@echo off
echo Copying encoded videos to output directory...

REM Copy H264/MP4 versions from input (if they exist)
copy "D:\Dev\kha\content\wowpack\input\*.mp4" "D:\Dev\kha\content\wowpack\output\" 2>nul

REM Copy AV1 versions
copy "D:\Dev\kha\content\wowpack\video\av1\*.mp4" "D:\Dev\kha\content\wowpack\output\" 2>nul

REM Copy HDR10 and SDR versions
copy "D:\Dev\kha\content\wowpack\video\hdr10\*.mp4" "D:\Dev\kha\content\wowpack\output\" 2>nul

echo Done! Files copied to output directory.
dir "D:\Dev\kha\content\wowpack\output\"