@echo off

REM Check if the user provided two arguments
if "%~2"=="" (
    echo Usage: %~0 dataset_name scale
    exit /b 1
)

set dataset_name=%~1
set scale=%~2
set dataset_folder=data\%dataset_name%

if not exist "%dataset_folder%" (
    echo Error: Folder '%dataset_folder%' does not exist.
    exit /b 2
)

REM Gaussian Grouping training
python train.py -s %dataset_folder% -r %scale% -m output\%dataset_name% --config_file config\gaussian_dataset\train.json

REM Segmentation rendering using trained model
REM python render.py -m output\%dataset_name% --num_classes 256
