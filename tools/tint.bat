@echo off
if "%1"=="--version" echo tint version 1.0.0-fake
if "%1"=="--format" (
  echo Fake tint: skipping shader compilation
  type nul > "%4"
)
exit 0