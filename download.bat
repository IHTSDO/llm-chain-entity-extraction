@echo off
setlocal enabledelayedexpansion

echo Enter the URL from email:
set /p PRESIGNED_URL=
echo.
echo Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all:
set /p MODEL_SIZE=
set TARGET_FOLDER=.

if not exist %TARGET_FOLDER% mkdir %TARGET_FOLDER%

if "%MODEL_SIZE%"=="" set MODEL_SIZE=7B,13B,70B,7B-chat,13B-chat,70B-chat

echo Downloading LICENSE and Acceptable Usage Policy
powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'=LICENSE! -OutFile %TARGET_FOLDER%\LICENSE"
powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'=USE_POLICY.md! -OutFile %TARGET_FOLDER%\USE_POLICY.md"

echo Downloading tokenizer
powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'=tokenizer.model! -OutFile %TARGET_FOLDER%\tokenizer.model"
powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'=tokenizer_checklist.chk! -OutFile %TARGET_FOLDER%\tokenizer_checklist.chk"

REM Check md5sum
REM Note: This will require you to have a Windows md5sum equivalent.

for %%m in (!MODEL_SIZE:,= !) do (
    if "%%m"=="7B" (
        set SHARD=0
        set MODEL_PATH=llama-2-7b
    ) else if "%%m"=="7B-chat" (
        set SHARD=0
        set MODEL_PATH=llama-2-7b-chat
    ) else if "%%m"=="13B" (
        set SHARD=1
        set MODEL_PATH=llama-2-13b
    ) else if "%%m"=="13B-chat" (
        set SHARD=1
        set MODEL_PATH=llama-2-13b-chat
    ) else if "%%m"=="70B" (
        set SHARD=7
        set MODEL_PATH=llama-2-70b
    ) else if "%%m"=="70B-chat" (
        set SHARD=7
        set MODEL_PATH=llama-2-70b-chat
    )

    echo Downloading !MODEL_PATH!
    if not exist %TARGET_FOLDER%\!MODEL_PATH! mkdir %TARGET_FOLDER%\!MODEL_PATH!

    for /l %%s in (0,1,!SHARD!) do (
        powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'='!MODEL_PATH!/consolidated.%%s.pth'! -OutFile %TARGET_FOLDER%\!MODEL_PATH!\consolidated.%%s.pth"
    )

    powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'='!MODEL_PATH!/params.json'! -OutFile %TARGET_FOLDER%\!MODEL_PATH!\params.json"
    powershell -Command "Invoke-WebRequest -Uri !PRESIGNED_URL:'*'='!MODEL_PATH!/checklist.chk'! -OutFile %TARGET_FOLDER%\!MODEL_PATH!\checklist.chk"

    echo Checking checksums
    REM Note: This will require you to have a Windows md5sum equivalent.
)

endlocal
