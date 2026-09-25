@echo off
REM run.bat - one-shot launcher for the Falcon Eye / Eagle Eye recognition scanner.
REM Usage:  run.bat        (uses external camera, index 1)
REM         run.bat 0      (built-in camera instead)

set FALCON_CAM_INDEX=%1
if "%FALCON_CAM_INDEX%"=="" set FALCON_CAM_INDEX=1

echo Using camera index %FALCON_CAM_INDEX%
py -m src.recognize
