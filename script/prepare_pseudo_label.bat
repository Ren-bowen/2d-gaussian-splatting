@echo off

REM Check if the user provided an argument
if "%~2"=="" (
    echo Usage: %~nx0 ^<dataset_name^> ^<scale^>
    exit /b 1
)

set dataset_name=%~1
set scale=%~2
set dataset_folder=data\%dataset_name%

if not exist "%dataset_folder%" (
    echo Error: Folder '%dataset_folder%' does not exist.
    exit /b 2
)

REM 1. DEVA anything mask
cd T

if "%scale%"=="1" (
    set img_path=..\data\%dataset_name%\images
) else (
    set img_path=..\data\%dataset_name%\images_%scale%
)

REM colored mask for visualization check
python demo\demo_automatic.py ^
  --chunk_size 4 ^
  --img_path "%img_path%" ^
  --amp ^
  --temporal_setting semionline ^
  --size 480 ^
  --output ".\example\output_gaussian_dataset\%dataset_name%" ^
  --suppress_small_objects ^
  --SAM_PRED_IOU_THRESHOLD 0.7

move .\example\output_gaussian_dataset\%dataset_name%\Annotations .\example\output_gaussian_dataset\%dataset_name%\Annotations_color

REM gray mask for training
python demo\demo_automatic.py ^
  --chunk_size 4 ^
  --img_path "%img_path%" ^
  --amp ^
  --temporal_setting semionline ^
  --size 480 ^
  --output ".\example\output_gaussian_dataset\%dataset_name%" ^
  --use_short_id ^
  --suppress_small_objects ^
  --SAM_PRED_IOU_THRESHOLD 0.7
  
REM 2. copy gray mask to the corresponding data path
xcopy /E /I /H /Y .\example\output_gaussian_dataset\%dataset_name%\Annotations ..\data\%dataset_name%\object_mask
cd ..
