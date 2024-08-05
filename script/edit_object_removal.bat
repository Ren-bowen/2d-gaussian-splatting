@echo off

REM Check if the user provided two arguments
if "%~2"=="" (
    echo Usage: %0 ^<output_folder^> ^<config_file^>
    exit /b 1
)

set output_folder=%1
set config_file=%2

if not exist "%output_folder%" (
    echo Error: Folder '%output_folder%' does not exist.
    exit /b 2
)

REM Remove the selected object
python edit_object_removal.py -m %output_folder% --config_file %config_file%
